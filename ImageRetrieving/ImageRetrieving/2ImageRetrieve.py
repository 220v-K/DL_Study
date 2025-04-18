# image_text_retrieval_wandb.py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from transformers import SwinModel, RobertaModel, RobertaTokenizer
from datasets import load_dataset

# 시드 고정 (재현성을 위해)
pl.seed_everything(42)

# =============================================================================
# 1. Flickr30K Multi-Caption Dataset
#    - HuggingFace 데이터셋을 이용하여 각 이미지 record의 5개 캡션을
#      각각 하나의 예제로 mapping합니다.
#    - 이미지: PIL 이미지 → transform 후 Tensor, 텍스트: Roberta 토크나이저로 tokenization
#    - 반환 dictionary에 "img_id"로 동일 이미지에 대한 인덱스를 포함합니다.
# =============================================================================
class Flickr30KMultiCaptionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_transform, max_length=64):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.index_map = []  # 각 예제가 (record_idx, caption_idx) 형태로 mapping됨.
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

        # 이미지 전처리: 최종 tensor shape → (3, 224, 224)
        image = self.image_transform(pil_image)

        # 텍스트 토큰화: input_ids, attention_mask, 최대 길이 max_length
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
            "img_id": record_idx  # 동일 이미지에 동일 id 부여
        }

# =============================================================================
# 2. DataModule 구성
#    - train, validation, test 셋을 각각 DataLoader로 구성합니다.
#    - HuggingFace load_dataset()를 통해 불러온 데이터셋을 사용합니다.
# =============================================================================
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

        # 이미지 전처리: Resize → ToTensor → Normalize
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))
        ])
        # Roberta-large 토크나이저 사용
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    def setup(self, stage=None):
        # 각 split은 HuggingFace 데이터셋의 "test" 키에 접근 (필요에 따라 수정)
        if stage == "fit" or stage is None:
            self.train_dataset = Flickr30KMultiCaptionDataset(
                self.train_dataset_hf["test"],
                tokenizer=self.tokenizer,
                image_transform=self.image_transform,
                max_length=self.max_length
            )
            self.valid_dataset = Flickr30KMultiCaptionDataset(
                self.valid_dataset_hf["test"],
                tokenizer=self.tokenizer,
                image_transform=self.image_transform,
                max_length=self.max_length
            )
        if stage == "test" or stage is None:
            self.test_dataset = Flickr30KMultiCaptionDataset(
                self.test_dataset_hf["test"],
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

# =============================================================================
# 3. Multi-Positive Contrastive Loss 함수
#    - 배치 내 각 anchor(텍스트 또는 이미지)에 대해, 동일한 img_id를 가진 모든 샘플을 positive로 간주.
#    - similarity_matrix: (B, B), positive_mask: (B, B)
#    - loss = mean( logsumexp(similarity) - log(sum(exp(similarity) over positive)) )
# =============================================================================
def multi_positive_contrastive_loss(similarity_matrix, positive_mask):
    denom = torch.logsumexp(similarity_matrix, dim=1)  # (B,)
    numerator = torch.log((torch.exp(similarity_matrix) * positive_mask).sum(dim=1) + 1e-8)  # (B,)
    loss = (denom - numerator).mean()
    return loss

# =============================================================================
# 4. Image-Text Lightning Module
#    - Image Encoder: Swin (예: microsoft/swin-base-patch4-window7-224)
#      • 입력 이미지: (B, 3, 224, 224)
#      • 출력: last_hidden_state, 평균 풀링하여 (B, hidden_size)
#    - Text Encoder: RoBERTa (roberta-large)
#      • 입력: input_ids (B, max_length), attention_mask (B, max_length)
#      • 출력: last_hidden_state → [CLS] 토큰 위치 → (B, hidden_size)
#    - 각 인코더의 feature를 Projection layer를 거쳐 embed_dim으로 매핑 후 L2 normalize.
#    - 최종 image_embeds, text_embeds의 shape: (B, embed_dim)
#    - compute_contrastive_loss에서는 배치 내 img_id를 활용하여 positive mask 생성 후 양방향 loss 계산.
# =============================================================================
class ImageTextLightningModel(pl.LightningModule):
    def __init__(self,
                 image_encoder_name="microsoft/swin-base-patch4-window7-224",
                 text_encoder_name="roberta-large",
                 embed_dim=256,
                 temperature=0.07,
                 learning_rate=1e-5,
                 vit_train_layers=12,        # Swin 모델의 마지막 fine-tuning할 레이어 수
                 roberta_train_layers=12):   # RoBERTa 모델의 마지막 fine-tuning할 레이어 수
        super().__init__()
        self.save_hyperparameters()

        # 1) 이미지 인코더: Swin 모델 (HuggingFace 제공)
        self.image_encoder = SwinModel.from_pretrained(image_encoder_name)
        # 2) 텍스트 인코더: RoBERTa-large
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_name)

        # 3) Projection layers: 각 인코더의 hidden_size → embed_dim
        image_hidden_size = self.image_encoder.config.hidden_size  # 예: 1024
        text_hidden_size = self.text_encoder.config.hidden_size       # 예: 1024
        self.image_proj = nn.Linear(image_hidden_size, embed_dim)  # (hidden_size -> embed_dim)
        self.text_proj = nn.Linear(text_hidden_size, embed_dim)    # (hidden_size -> embed_dim)

        self.temperature = temperature
        self.learning_rate = learning_rate

        # 검증/테스트 시 임베딩 저장 (recall 계산용)
        self._val_outputs = []
        self.test_image_embeds = []
        self.test_text_embeds = []

        # Freeze를 사용하고 있으므로, 전체 파라미터를 동결한 후 마지막 몇 레이어만 unfreeze합니다.
        self.freeze_image_encoder_layers(train_layers=vit_train_layers)
        self.freeze_roberta_layers(train_layers=roberta_train_layers)

    def freeze_image_encoder_layers(self, train_layers):
        # 전체 이미지 인코더 파라미터 동결 (freeze)
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        # Swin 모델의 encoder는 self.image_encoder.encoder.layers 에 위치합니다.
        if hasattr(self.image_encoder.encoder, "layers"):
            layers = self.image_encoder.encoder.layers
        else:
            layers = []
        total_layers = len(layers)
        # 마지막 train_layers 만큼만 unfreeze
        for layer_idx in range(max(0, total_layers - train_layers), total_layers):
            for param in layers[layer_idx].parameters():
                param.requires_grad = True
        # 추가적으로 layernorm이 있다면 unfreeze (예: self.image_encoder.layernorm)
        if hasattr(self.image_encoder, "layernorm"):
            for param in self.image_encoder.layernorm.parameters():
                param.requires_grad = True

    def freeze_roberta_layers(self, train_layers):
        # 전체 텍스트 인코더 파라미터 동결
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
        # --- 이미지 인코더 ---
        # images: (B, 3, 224, 224)
        image_outputs = self.image_encoder(pixel_values=images)
        # Swin: last_hidden_state shape = (B, seq_len, hidden_size)
        # 평균 풀링하여 (B, hidden_size)로 만듦
        image_feat = image_outputs.last_hidden_state.mean(dim=1)  # (B, hidden_size)
        image_embeds = self.image_proj(image_feat)                # (B, embed_dim)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)       # L2 정규화

        # --- 텍스트 인코더 ---
        # input_ids, attention_mask: (B, max_length)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # RoBERTa: last_hidden_state shape = (B, max_length, hidden_size)
        # [CLS] 토큰에 해당하는 첫 토큰 (B, hidden_size)
        text_feat = text_outputs.last_hidden_state[:, 0, :]
        text_embeds = self.text_proj(text_feat)                   # (B, embed_dim)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        return image_embeds, text_embeds

    def compute_contrastive_loss(self, image_embeds, text_embeds, img_ids):
        # text→image 방향 similarity matrix: (B, B)
        sim_matrix = torch.matmul(text_embeds, image_embeds.t()) / self.temperature
        # img_ids: (B,), positive_mask: (B, B) (동일 img_id이면 1)
        if not torch.is_tensor(img_ids):
            img_ids = torch.tensor(img_ids, device=self.device)
        else:
            img_ids = img_ids.to(self.device)
        positive_mask = (img_ids.unsqueeze(1) == img_ids.unsqueeze(0)).float()
        loss_t2i = multi_positive_contrastive_loss(sim_matrix, positive_mask)
        loss_i2t = multi_positive_contrastive_loss(sim_matrix.t(), positive_mask)
        loss = (loss_t2i + loss_i2t) / 2.0
        return loss

    def training_step(self, batch, batch_idx):
        images = batch["image"]             # (B, 3, 224, 224)
        input_ids = batch["input_ids"]        # (B, max_length)
        attention_mask = batch["attention_mask"]  # (B, max_length)
        img_ids = batch["img_id"]             # (B,)
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
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return {
            "val_loss": val_loss,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "img_ids": img_ids
        }

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._val_outputs.append(outputs)

    def on_validation_epoch_end(self):
        # 전체 validation loss 계산 및 로깅
        val_losses = torch.stack([o["val_loss"] for o in self._val_outputs])
        avg_val_loss = val_losses.mean()
        self.log("avg_val_loss", avg_val_loss, prog_bar=True)

        # 전체 배치의 이미지/텍스트 임베딩을 모아서 recall@K 계산 (여기서는 text→image retrieval)
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
        # similarity_matrix: (N, N), 각 row에 대해 정답이 top-k에 포함되는지 계산
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

    def configure_optimizers(self):
        base_lr = self.learning_rate
        # 아래 layerwise lr decay 부분은 freeze된 encoder를 사용하기 때문에 현재는 주석 처리되어 있습니다.
        # 추후 encoder의 일부를 fine-tuning 할 때 사용하고자 한다면, 주석을 해제하고 설정을 조정하세요.
        #
        # layerwise_decay = 0.9
        # optimizer_params = []
        # # --- 이미지 인코더: layer-wise learning rate decay 적용 ---
        # if hasattr(self.image_encoder.encoder, "layers"):
        #     image_layers = self.image_encoder.encoder.layers
        # elif hasattr(self.image_encoder.encoder, "layer"):
        #     image_layers = self.image_encoder.encoder.layer
        # else:
        #     image_layers = []
        # total_image_layers = len(image_layers)
        # train_image_layers = self.hparams.vit_train_layers
        # for i, layer_idx in enumerate(range(max(0, total_image_layers - train_image_layers), total_image_layers)):
        #     lr = base_lr * (layerwise_decay ** i)
        #     optimizer_params.append({"params": image_layers[layer_idx].parameters(), "lr": lr})
        #
        # # --- 텍스트 인코더: layer-wise learning rate decay 적용 ---
        # total_text_layers = len(self.text_encoder.encoder.layer)
        # train_text_layers = self.hparams.roberta_train_layers
        # for i, layer_idx in enumerate(range(max(0, total_text_layers - train_text_layers), total_text_layers)):
        #     lr = base_lr * (layerwise_decay ** i)
        #     optimizer_params.append({"params": self.text_encoder.encoder.layer[layer_idx].parameters(), "lr": lr})
        #
        # # --- Projection layers: 새로 학습되는 부분은 더 큰 lr 적용 ---
        # optimizer_params.append({"params": self.image_proj.parameters(), "lr": base_lr * 5})
        # optimizer_params.append({"params": self.text_proj.parameters(), "lr": base_lr * 5})
        #
        # optimizer = torch.optim.AdamW(optimizer_params, lr=base_lr, weight_decay=1e-4)

        # 현재 freeze 상태이므로, 전체 모델 파라미터 중 requires_grad=True인 부분만 optimizer에 전달합니다.
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                      lr=base_lr, weight_decay=1e-4)
        # CosineAnnealingLR 스케줄러 사용 (lr decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7
        )
        return [optimizer], [scheduler]

# =============================================================================
# 5. 데이터셋 로드 및 DataModule 생성
#    - HuggingFace load_dataset()를 통해 "nlphuji/flickr30k" 데이터셋을 불러오고,
#      split 필드에 따라 train, val, test 셋으로 분리합니다.
# =============================================================================
dataset = load_dataset("nlphuji/flickr30k")
train_dataset = dataset.filter(lambda x: x["split"] == "train")
valid_dataset = dataset.filter(lambda x: x["split"] == "val")
test_dataset  = dataset.filter(lambda x: x["split"] == "test")

data_module = Flickr30KDataModule(
    train_dataset_hf=train_dataset,
    valid_dataset_hf=valid_dataset,
    test_dataset_hf=test_dataset,
    batch_size=128,
    num_workers=4,
    max_length=64
)
data_module.setup("fit")

# =============================================================================
# 6. Trainer, Logger, Callback 설정
#
# [WandBLogger]
#   - WandB 로그를 저장합니다.
#   - project, entity, 그리고 기타 설정은 WandB 대시보드에서 확인 및 수정 가능합니다.
#
# [ModelCheckpoint]
#   - "val_recall@5"를 기준으로 최대 5개의 체크포인트를 저장합니다.
#   - mode="max"로 설정하여 값이 클수록 좋은 경우에 저장합니다.
#
# [EarlyStopping]
#   - "val_recall@5"가 개선되지 않으면 patience(예: 10 epoch) 후 학습을 중단합니다.
#
# Trainer 설정에서:
#   - max_epochs: 전체 에폭 수 (실험 환경에 맞게 조정)
#   - accelerator 및 devices: GPU 사용 여부
#   - precision: 혼합 정밀도 (16-bit) 사용 여부
# =============================================================================

wandb_logger = WandbLogger(
    project="ImageRetrieve_WandB",  # WandB 프로젝트 이름 (수정 가능)
    log_model=True
)

# checkpoint_callback = ModelCheckpoint(
#     monitor="val_recall@5",       # 검증 시 Recall@5를 기준으로
#     mode="max",                   # 값이 클수록 좋은 metric
#     dirpath="checkpoints_multi",  # 체크포인트 저장 경로
#     filename="best-checkpoint",   # 파일명 (변경 가능)
#     save_top_k=5,                 # 최대 5개 체크포인트 저장
#     save_last=True                # 마지막 에폭 체크포인트도 저장
# )

early_stopping_callback = EarlyStopping(
    monitor="val_recall@5",  # 기준 metric: Recall@5
    patience=5,             # 개선이 없으면 10 epoch 후 중단
    mode="max"
)

trainer = pl.Trainer(
    max_epochs=200,           # 전체 에폭 수 (실험 환경에 따라 조정)
    accelerator="gpu",        # GPU 사용 (없으면 "cpu"로 변경)
    devices=1,                # 사용할 GPU 수
    precision="16-mixed",     # 16-bit 혼합 정밀도 사용 (메모리 효율 개선)
    logger=wandb_logger,      # WandB Logger 사용
    # callbacks=[checkpoint_callback, early_stopping_callback],
    callbacks=[early_stopping_callback],
    enable_checkpointing=False  # Disable automatic checkpoint saving
)

# =============================================================================
# 7. 모델 초기화 및 학습 시작
# =============================================================================
model = ImageTextLightningModel(
    image_encoder_name="microsoft/swin-base-patch4-window7-224",
    text_encoder_name="roberta-large",
    embed_dim=256,
    temperature=0.07,
    learning_rate=1e-5,
    vit_train_layers=12,        # Swin 모델의 마지막 12개 레이어만 fine-tuning
    roberta_train_layers=12     # RoBERTa 모델의 마지막 12개 레이어만 fine-tuning
)

trainer.fit(model, data_module)
