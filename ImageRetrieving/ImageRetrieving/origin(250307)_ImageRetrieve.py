import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import SwinModel, RobertaModel, RobertaTokenizer
from datasets import load_dataset

import math

# -----------------------------------------------------------------------------
# 1. Global Settings: Seed and Tensor Precision
# -----------------------------------------------------------------------------
pl.seed_everything(42)
torch.set_float32_matmul_precision('medium')  # Optimize matmul precision for CUDA TensorCores

# -----------------------------------------------------------------------------
# 2. Flickr30K Multi-Caption Dataset
#    Each image record from the HuggingFace dataset is split into as many examples 
#    as there are captions (i.e. 5 captions per image). Images are transformed and
#    text is tokenized using Roberta-large.
# -----------------------------------------------------------------------------
class Flickr30KMultiCaptionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_transform, max_length=64):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.index_map = []  # Map each example as (record_idx, caption_idx)
        for rec_idx, record in enumerate(self.hf_dataset):
            captions = record["caption"]
            for cap_idx in range(len(captions)):
                self.index_map.append((rec_idx, cap_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        record_idx, caption_idx = self.index_map[idx]
        record = self.hf_dataset[record_idx]
        pil_image = record["image"]  # PIL Image
        caption = record["caption"][caption_idx]

        # Image transformation: Resize, ToTensor, and Normalize
        image = self.image_transform(pil_image)

        # Tokenize text using Roberta tokenizer: output tensors with fixed max_length
        tokenized = self.tokenizer(
            caption,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)          # (max_length,)
        attention_mask = tokenized["attention_mask"].squeeze(0)  # (max_length,)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "img_id": record_idx  # Same image gets the same id
        }

# -----------------------------------------------------------------------------
# 3. DataModule: Loads train/val/test splits from Flickr30k
# -----------------------------------------------------------------------------
class Flickr30KDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dataset_hf,
                 valid_dataset_hf,
                 test_dataset_hf,
                 batch_size=128,
                 num_workers=4,
                 max_length=64):
        super().__init__()
        self.train_dataset_hf = train_dataset_hf
        self.valid_dataset_hf = valid_dataset_hf
        self.test_dataset_hf = test_dataset_hf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length

        # Define image transformation: Resize → ToTensor → Normalize
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))
        ])
        # Use Roberta-large tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Flickr30KMultiCaptionDataset(
                self.train_dataset_hf,
                tokenizer=self.tokenizer,
                image_transform=self.image_transform,
                max_length=self.max_length
            )
            self.valid_dataset = Flickr30KMultiCaptionDataset(
                self.valid_dataset_hf,
                tokenizer=self.tokenizer,
                image_transform=self.image_transform,
                max_length=self.max_length
            )
        if stage == "test" or stage is None:
            self.test_dataset = Flickr30KMultiCaptionDataset(
                self.test_dataset_hf,
                tokenizer=self.tokenizer,
                image_transform=self.image_transform,
                max_length=self.max_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

# -----------------------------------------------------------------------------
# 4. Image-Text Lightning Module
#    The model uses:
#      • A frozen Swin image encoder and frozen RoBERTa text encoder.
#      • Projection layers to map encoder outputs to a common embedding space.
#      • Mixture tokens with cross-attention to condition image features on the text.
#      • A contrastive loss (with learnable scale 'a' and bias 'b') computed over 
#        the similarity matrix between text and image embeddings.
#
#    Validation and test steps compute recall@K based on cosine similarity.
# -----------------------------------------------------------------------------
class ImageTextLightningModel(pl.LightningModule):
    def __init__(self,
                 image_encoder_name="microsoft/swin-base-patch4-window7-224",
                 text_encoder_name="roberta-large",
                 embed_dim=512,  # 임베딩 차원 증가
                 temperature=0.05,  # 온도 파라미터 조정
                 learning_rate=2e-4,  # 학습률 증가
                 vit_train_layers=3,  # 미세 조정할 이미지 인코더 레이어 수
                 roberta_train_layers=3):  # 미세 조정할 텍스트 인코더 레이어 수
        super().__init__()
        self.save_hyperparameters()

        # -------------------------------
        # 사전 훈련된 인코더 로드 (이미지용 Swin, 텍스트용 RoBERTa)
        # 마지막 몇 개 레이어는 미세 조정을 위해 동결 해제
        # -------------------------------
        self.image_encoder = SwinModel.from_pretrained(image_encoder_name)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_name)
        
        # 인코더 레이어 동결/동결 해제
        self.freeze_image_encoder_layers(vit_train_layers)
        self.freeze_roberta_layers(roberta_train_layers)

        # -------------------------------
        # 프로젝션 레이어: 각 인코더의 출력을 공통 임베딩 차원으로 매핑
        # -------------------------------
        image_hidden_size = self.image_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        
        # 2단계 프로젝션 레이어 (성능 향상)
        self.image_proj = nn.Sequential(
            nn.Linear(image_hidden_size, image_hidden_size),
            nn.LayerNorm(image_hidden_size),
            nn.GELU(),
            nn.Linear(image_hidden_size, embed_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size),
            nn.LayerNorm(text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, embed_dim)
        )

        # -------------------------------
        # 혼합 토큰 및 교차 어텐션:
        # 텍스트 표현과의 교차 어텐션을 통해 문맥화된 시각 특성을
        # 생성하는 혼합 토큰 집합 학습
        # -------------------------------
        self.num_mixture = 64  # 혼합 토큰 수
        
        # Xavier 초기화로 혼합 토큰 초기화
        self.mixture_tokens = nn.Parameter(torch.empty(self.num_mixture, image_hidden_size))
        nn.init.xavier_normal_(self.mixture_tokens)
        
        # 다중 헤드 교차 어텐션
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_hidden_size, 
            num_heads=16,  # 어텐션 헤드 수 증가
            dropout=0.1,   # 드롭아웃 추가
            batch_first=True
        )

        # 온도 파라미터
        self.temperature = temperature
        self.learning_rate = learning_rate

        # 검증/테스트 임베딩 로깅 (리콜 계산용)
        self._val_outputs = []
        self.test_image_embeds = []
        self.test_text_embeds = []

    def freeze_image_encoder_layers(self, train_layers):
        # Freeze all parameters of the image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        # Unfreeze the last 'train_layers' layers if available (Swin encoder layers)
        if hasattr(self.image_encoder.encoder, "layers"):
            layers = self.image_encoder.encoder.layers
        elif hasattr(self.image_encoder.encoder, "layer"):
            layers = self.image_encoder.encoder.layer
        else:
            layers = []
        total_layers = len(layers)
        for layer_idx in range(max(0, total_layers - train_layers), total_layers):
            for param in layers[layer_idx].parameters():
                param.requires_grad = True
        # Also unfreeze any final normalization layers if present
        if hasattr(self.image_encoder, "layernorm"):
            for param in self.image_encoder.layernorm.parameters():
                param.requires_grad = True

    def freeze_roberta_layers(self, train_layers):
        # Freeze all parameters of the text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        total_layers = len(self.text_encoder.encoder.layer)
        for layer_idx in range(max(0, total_layers - train_layers), total_layers):
            for param in self.text_encoder.encoder.layer[layer_idx].parameters():
                param.requires_grad = True
        if hasattr(self.text_encoder, "pooler"):
            for param in self.text_encoder.pooler.parameters():
                param.requires_grad = True

    def forward(self, images, input_ids, attention_mask):
        # -------------------------------
        # Image Encoder: Process images through Swin and get full features
        # -------------------------------
        image_outputs = self.image_encoder(pixel_values=images)
        image_features = image_outputs.last_hidden_state  # (B, N, hidden_size)
        B = images.size(0)
        
        # -------------------------------
        # Text Encoder: Process tokenized captions through RoBERTa
        # Use full sequence output instead of just CLS token
        # -------------------------------
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # (B, L, hidden_size)
        
        # Global average pooling on image features for better representation
        image_cls = image_features.mean(dim=1)  # (B, hidden_size)
        
        # Get contextualized text representation with attention over all tokens
        text_attention_weights = F.softmax(
            torch.matmul(text_features, text_features.transpose(-1, -2)) / 
            math.sqrt(text_features.size(-1)), 
            dim=-1
        )
        text_cls = torch.matmul(text_attention_weights, text_features).mean(dim=1)  # (B, hidden_size)
        
        # -------------------------------
        # Cross-modal interaction with mixture tokens
        # -------------------------------
        # Expand mixture tokens for each sample: (K, hidden_size) → (B, K, hidden_size)
        mixture_tokens = self.mixture_tokens.unsqueeze(0).expand(B, -1, -1)
        
        # Use text representation to attend over mixture tokens
        text_query = text_cls.unsqueeze(1)  # (B, 1, hidden_size)
        attn_output, _ = self.cross_attn(query=text_query, key=mixture_tokens, value=mixture_tokens)
        conditioned_image_feat = attn_output.squeeze(1) + image_cls  # Add residual connection with image features
        
        # -------------------------------
        # Projection and Normalization: Map to embed_dim and L2-normalize
        # -------------------------------
        image_embeds = self.image_proj(conditioned_image_feat)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = self.text_proj(text_cls)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        return image_embeds, text_embeds

    def compute_contrastive_loss(self, image_embeds, text_embeds, img_ids):
        """
        InfoNCE/NT-Xent 손실 함수 구현 + 하드 네거티브 마이닝
        """
        # 코사인 유사도 행렬 계산 
        sim_matrix = torch.matmul(text_embeds, image_embeds.t()) / self.temperature  # (B, B)
        B = sim_matrix.size(0)
        
        # 레이블: 대각선 요소가 양성 쌍 (같은 인덱스의 이미지-텍스트 쌍)
        labels = torch.arange(B, device=sim_matrix.device)
        
        # 텍스트->이미지 방향 손실 (각 텍스트에 대해 올바른 이미지를 찾기)
        t2i_loss = F.cross_entropy(sim_matrix, labels)
        
        # 이미지->텍스트 방향 손실 (각 이미지에 대해 올바른 텍스트를 찾기)
        i2t_loss = F.cross_entropy(sim_matrix.t(), labels)
        
        # 양방향 손실 결합
        loss = (t2i_loss + i2t_loss) / 2.0
        
        # 하드 네거티브: 가장 어려운 네거티브 샘플에 더 가중치 부여
        with torch.no_grad():
            # 양성 쌍 마스크 제거 (대각선 요소)
            neg_mask = 1 - torch.eye(B, device=sim_matrix.device)
            # 가장 유사한 네거티브 샘플 찾기
            hardest_neg_t2i = (sim_matrix * neg_mask).max(dim=1)[0]
            hardest_neg_i2t = (sim_matrix.t() * neg_mask).max(dim=1)[0]
        
        # 양성 쌍 유사도 (대각선 요소)
        pos_sim_t2i = sim_matrix.diag()
        pos_sim_i2t = sim_matrix.t().diag()
        
        # 하드 네거티브 마진 손실 추가
        hard_loss_t2i = F.relu(hardest_neg_t2i - pos_sim_t2i + 0.2).mean()
        hard_loss_i2t = F.relu(hardest_neg_i2t - pos_sim_i2t + 0.2).mean()
        hard_loss = (hard_loss_t2i + hard_loss_i2t) / 2.0
        
        # 최종 손실 = 기본 대조 손실 + 하드 네거티브 마진 손실
        total_loss = loss + 0.5 * hard_loss
        
        return total_loss

    def training_step(self, batch, batch_idx):
        images = batch["image"]               # (B, 3, 224, 224)
        input_ids = batch["input_ids"]          # (B, max_length)
        attention_mask = batch["attention_mask"]# (B, max_length)
        img_ids = batch["img_id"]               # (B,)
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        loss = self.compute_contrastive_loss(image_embeds, text_embeds, img_ids)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch["img_id"]
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        val_loss = self.compute_contrastive_loss(image_embeds, text_embeds, img_ids)
        self.log("val_loss_step", val_loss, prog_bar=False)
        return {
            "val_loss": val_loss,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "img_ids": img_ids
        }

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._val_outputs.append(outputs)

    def on_validation_epoch_end(self):
        # Log average validation loss
        val_losses = torch.stack([o["val_loss"] for o in self._val_outputs])
        avg_val_loss = val_losses.mean()
        self.log("val_loss", avg_val_loss, prog_bar=True)
        # Concatenate embeddings to compute recall metrics (text→image retrieval)
        all_image_embeds = torch.cat([o["image_embeds"] for o in self._val_outputs], dim=0)
        all_text_embeds  = torch.cat([o["text_embeds"] for o in self._val_outputs], dim=0)
        similarity_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
        recall_at_k = self.compute_recall(similarity_matrix, ks=[1, 5, 10])
        for k, v in recall_at_k.items():
            self.log(f"val_recall@{k}", v, prog_bar=True)
        self._val_outputs.clear()

    def test_step(self, batch, batch_idx):
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        return {"image_embeds": image_embeds, "text_embeds": text_embeds}

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_image_embeds.append(outputs["image_embeds"])
        self.test_text_embeds.append(outputs["text_embeds"])

    def on_test_epoch_end(self):
        all_image_embeds = torch.cat(self.test_image_embeds, dim=0)
        all_text_embeds  = torch.cat(self.test_text_embeds, dim=0)
        similarity_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
        recall_at_k = self.compute_recall(similarity_matrix, ks=[1, 5, 10])
        for k, v in recall_at_k.items():
            self.log(f"test_recall@{k}", v, prog_bar=True)
        print(f"[on_test_epoch_end] Test Recall: {recall_at_k}")
        self.test_image_embeds.clear()
        self.test_text_embeds.clear()

    def compute_recall(self, similarity_matrix, ks=[1, 5, 10]):
        # For each query (row), check if the ground-truth index is in the top-k retrieved indices.
        device = similarity_matrix.device
        n = similarity_matrix.size(0)
        ground_truth = torch.arange(n, device=device)
        sorted_indices = similarity_matrix.argsort(dim=1, descending=True)
        recall_scores = {}
        for k in ks:
            top_k = sorted_indices[:, :k]
            match = (top_k == ground_truth.unsqueeze(1)).any(dim=1)
            recall_scores[k] = match.float().mean().item()
        return recall_scores

    # def configure_optimizers(self):
    #     base_lr = self.learning_rate
    #     layerwise_decay = 0.9
    #     optimizer_params = []

    #     # (Optional) Apply layer-wise LR decay to encoder layers if fine-tuning is enabled.
    #     # In this configuration, we keep vit_train_layers and roberta_train_layers = 0 so encoders remain frozen.
    #     # Add projection layers and newly introduced modules with a higher LR.
    #     optimizer_params.append({"params": self.image_proj.parameters(), "lr": base_lr * 5})
    #     optimizer_params.append({"params": self.text_proj.parameters(), "lr": base_lr * 5})
    #     optimizer_params.append({"params": self.mixture_tokens, "lr": base_lr * 5})
    #     optimizer_params.append({"params": self.cross_attn.parameters(), "lr": base_lr * 5})
    #     optimizer_params.append({"params": [self.a, self.b], "lr": base_lr})

    #     optimizer = torch.optim.AdamW(optimizer_params, lr=base_lr, weight_decay=1e-4)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7
    #     )
    #     return [optimizer], [scheduler]
    
    #! layerwise decay
    def configure_optimizers(self):
        base_lr = self.learning_rate
        params_groups = []
        
        # 1. 인코더 미세 조정을 위한 파라미터 그룹화
        if self.hparams.vit_train_layers > 0:
            # 이미지 인코더 (Swin) 레이어
            layers = self.image_encoder.encoder.layers if hasattr(self.image_encoder.encoder, "layers") else self.image_encoder.encoder.layer
            total_layers = len(layers)
            
            # 미세 조정할 레이어만 선택
            for i in range(1, self.hparams.vit_train_layers + 1):
                layer_idx = total_layers - i  # 뒤에서부터 레이어 선택
                layer_lr = base_lr * (1.5 ** i)  # 뒤쪽 레이어일수록 더 높은 학습률
                params_groups.append({"params": layers[layer_idx].parameters(), "lr": layer_lr})
        
        if self.hparams.roberta_train_layers > 0:
            # 텍스트 인코더 (RoBERTa) 레이어
            text_layers = self.text_encoder.encoder.layer
            total_text_layers = len(text_layers)
            
            # 미세 조정할 레이어만 선택
            for i in range(1, self.hparams.roberta_train_layers + 1):
                layer_idx = total_text_layers - i  # 뒤에서부터 레이어 선택
                layer_lr = base_lr * (1.5 ** i)  # 뒤쪽 레이어일수록 더 높은 학습률
                params_groups.append({"params": text_layers[layer_idx].parameters(), "lr": layer_lr})
            
            # 풀러 레이어 추가
            if hasattr(self.text_encoder, "pooler"):
                params_groups.append({"params": self.text_encoder.pooler.parameters(), "lr": base_lr * 3})
        
        # 2. 커스텀 모듈에 대한 파라미터 (더 높은 학습률 적용)
        params_groups.extend([
            {"params": self.image_proj.parameters(), "lr": base_lr * 5},
            {"params": self.text_proj.parameters(), "lr": base_lr * 5},
            {"params": self.mixture_tokens, "lr": base_lr * 5},
            {"params": self.cross_attn.parameters(), "lr": base_lr * 5},
            {"params": [self.a, self.b], "lr": base_lr * 2}
        ])
        
        # 3. 옵티마이저 및 스케줄러 생성
        optimizer = torch.optim.AdamW(params_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7
        )
        
        return [optimizer], [scheduler]

# -----------------------------------------------------------------------------
# 5. Load Flickr30k Dataset and Prepare DataModule
# -----------------------------------------------------------------------------
dataset = load_dataset("nlphuji/flickr30k")["test"]
train_dataset = dataset.filter(lambda x: x["split"] == "train")
valid_dataset = dataset.filter(lambda x: x["split"] == "val")
test_dataset  = dataset.filter(lambda x: x["split"] == "test")

data_module = Flickr30KDataModule(
    train_dataset_hf=train_dataset,
    valid_dataset_hf=valid_dataset,
    test_dataset_hf=test_dataset,
    batch_size=64,
    num_workers=4,
    max_length=64
)
data_module.setup("fit")

# -----------------------------------------------------------------------------
# 6. Trainer, Logger, and Callback Setup
# -----------------------------------------------------------------------------
logger = TensorBoardLogger(
    save_dir="ImageRetrieving_2503",   # Directory for TensorBoard logs
    name="Llip_ver2"                   # Project name (modifiable)
)

# Modify EarlyStopping to monitor recall@1
early_stopping_callback = EarlyStopping(
    monitor="val_recall@1",  # Monitor Recall@1 metric on validation
    patience=5,              # Stop training if no improvement after 5 epochs
    mode="max"
)

# Set up ModelCheckpoint to save the top 5 models based on recall@1
checkpoint_callback = ModelCheckpoint(
    monitor="val_recall@1",  # Monitor Recall@1 metric
    mode="max",               # Save the highest recall@1
    dirpath="checkpoints",    # Directory to save checkpoints
    filename="best-checkpoint-{epoch:02d}-{val_recall@1:.2f}",  # Checkpoint filename format
    save_top_k=5             # Save the top 5 models
)

trainer = pl.Trainer(
    max_epochs=200,           # Total number of epochs (adjust as needed)
    accelerator="gpu",        # Use GPU (or "cpu" if not available)
    devices=1,                # Number of GPUs to use
    precision="16-mixed",     # Mixed precision (16-bit) for memory efficiency
    logger=logger,            # TensorBoardLogger instance
    callbacks=[early_stopping_callback, checkpoint_callback],  # Add checkpoint callback
    # enable_checkpointing=False  # Disable automatic checkpoint saving
)

# -----------------------------------------------------------------------------
# 7. Model Initialization and Training
# -----------------------------------------------------------------------------
model = ImageTextLightningModel(
    image_encoder_name="microsoft/swin-base-patch4-window7-224",
    text_encoder_name="roberta-large",
    embed_dim=512,
    temperature=0.05,
    learning_rate=2e-4,
    vit_train_layers=3,
    roberta_train_layers=3
)

trainer.fit(model, data_module)

# Evaluate on the test dataset and save recall metrics to a file.
test_results = trainer.test(model, datamodule=data_module)

with open("test_recall.txt", "w") as f:
    f.write("Test Recall Metrics:\n")
    for result in test_results:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")

print("테스트 평가 결과가 'test_recall.txt' 파일에 저장되었습니다.")
