import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from torchvision import transforms
from transformers import SwinModel, RobertaModel, RobertaTokenizer
from datasets import load_dataset

# 재현성을 위한 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 시드 설정 적용
set_seed(42)

# -----------------------------------------------------------------------------
# 1. Global Settings
# -----------------------------------------------------------------------------
pl.seed_everything(42)
torch.set_float32_matmul_precision('medium')

# -----------------------------------------------------------------------------
# 2. Flickr30K Multi-Caption Dataset - 수정된 버전
# -----------------------------------------------------------------------------
class Flickr30KMultiCaptionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_transform, max_length=64):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        
        # 인덱스 매핑 구성
        self.index_map = []
        for rec_idx, record in enumerate(self.hf_dataset):
            captions = record["caption"]
            for cap_idx in range(len(captions)):
                self.index_map.append((rec_idx, cap_idx))
                
        # 이미지 ID -> 인덱스 맵 구성
        self.img_id_to_indices = {}
        for idx, (img_id, _) in enumerate(self.index_map):
            if img_id not in self.img_id_to_indices:
                self.img_id_to_indices[img_id] = []
            self.img_id_to_indices[img_id].append(idx)
                
        # 이미지 전처리
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        record_idx, caption_idx = self.index_map[idx]
        record = self.hf_dataset[record_idx]
        pil_image = record["image"]
        caption = record["caption"][caption_idx]
        
        # 이미지 변환 적용
        image = self.preprocess(pil_image)
        
        # 토큰화
        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "img_id": record_idx,
            "caption_idx": caption_idx,
            "caption": caption
        }

# -----------------------------------------------------------------------------
# 3. DataModule
# -----------------------------------------------------------------------------
class Flickr30KDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dataset_hf,
                 valid_dataset_hf,
                 test_dataset_hf,
                 batch_size=32,
                 num_workers=4,
                 max_length=96):
        super().__init__()
        self.train_dataset_hf = train_dataset_hf
        self.valid_dataset_hf = valid_dataset_hf
        self.test_dataset_hf = test_dataset_hf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length

        # RoBERTa 토크나이저
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        # 학습용 이미지 변환
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        # 평가용 이미지 변환
        self.eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Flickr30KMultiCaptionDataset(
                self.train_dataset_hf,
                tokenizer=self.tokenizer,
                image_transform=self.train_transform,
                max_length=self.max_length
            )
            
            self.valid_dataset = Flickr30KMultiCaptionDataset(
                self.valid_dataset_hf,
                tokenizer=self.tokenizer,
                image_transform=self.eval_transform,
                max_length=self.max_length
            )
            
            print(f"학습 데이터셋 크기: {len(self.train_dataset)}")
            print(f"검증 데이터셋 크기: {len(self.valid_dataset)}")
            
        if stage == "test" or stage is None:
            self.test_dataset = Flickr30KMultiCaptionDataset(
                self.test_dataset_hf,
                tokenizer=self.tokenizer,
                image_transform=self.eval_transform,
                max_length=self.max_length
            )
            print(f"테스트 데이터셋 크기: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

# -----------------------------------------------------------------------------
# 4. 개선된 이미지 텍스트 Lightning 모델
# -----------------------------------------------------------------------------
class ImprovedImageTextModel(pl.LightningModule):
    def __init__(self,
                 image_encoder_name="microsoft/swin-base-patch4-window7-224",
                 text_encoder_name="roberta-large",
                 embed_dim=256,
                 temperature=0.07,
                 learning_rate=1e-4,
                 use_hard_negatives=True,
                 hard_negative_weight=10.0,  # 하드 네거티브에 더 높은 가중치 부여
                 margin=0.2):              # 마진 추가
        super().__init__()
        self.save_hyperparameters()

        # 하이퍼파라미터 저장
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_weight = hard_negative_weight
        self.margin = margin

        # 인코더 초기화
        self.image_encoder = SwinModel.from_pretrained(image_encoder_name)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_name)
        
        # 인코더 동결 (특성 추출기로만 사용)
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 프로젝션 레이어 설정
        image_hidden_size = self.image_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        
        self.image_proj = nn.Linear(image_hidden_size, embed_dim)
        self.text_proj = nn.Linear(text_hidden_size, embed_dim)

        # 검증/테스트 임베딩 로깅
        self._val_outputs = []
        self.test_image_embeds = []
        self.test_text_embeds = []
        
    def forward(self, images, input_ids, attention_mask):
        # 이미지 인코딩
        image_outputs = self.image_encoder(pixel_values=images)
        image_features = image_outputs.last_hidden_state
        
        # 이미지 임베딩 - 단순 평균 풀링
        image_embeds = image_features.mean(dim=1)
        image_embeds = self.image_proj(image_embeds)
        
        # 텍스트 인코딩
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 텍스트 임베딩 - [CLS] 토큰 사용
        text_embeds = text_outputs.last_hidden_state[:, 0]
        text_embeds = self.text_proj(text_embeds)
        
        # L2 정규화
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        return image_embeds, text_embeds

    def compute_multicaption_loss(self, image_embeds, text_embeds, img_ids):
        """
        다중 캡션을 고려한 개선된 대조 손실 계산
        """
        device = image_embeds.device
        batch_size = image_embeds.size(0)
        
        # 유사도 행렬 계산
        logits = torch.matmul(text_embeds, image_embeds.t()) / self.temperature
        
        # 그라운드 트루스 마스크 생성 (같은 이미지 ID를 갖는 샘플들은 positive로 취급)
        # img_ids: [batch_size]의 텐서 (각 샘플의 이미지 ID)
        img_ids = img_ids.view(-1, 1)  # [batch_size, 1]
        pos_mask = (img_ids == img_ids.t()).float()  # [batch_size, batch_size]
        
        # 각 행에 대해 positive 샘플을 제외한 최대 유사도를 가진 hard negative 찾기
        if self.use_hard_negatives:
            neg_mask = 1.0 - pos_mask  # positive가 아닌 모든 샘플은 negative
            
            # 마스킹된 유사도 행렬 (positive는 매우 낮은 값으로 설정)
            masked_logits = logits - pos_mask * 1e9
            
            # 가장 어려운 negative 찾기 (각 행에서 유사도가 가장 높은 negative)
            hardest_negatives_scores, _ = masked_logits.max(dim=1, keepdim=True)  # [batch_size, 1]
            
            # 원래 유사도에 hard negative에 대한 가중치 적용
            hard_negative_mask = (masked_logits == hardest_negatives_scores)
            weighted_logits = logits + hard_negative_mask.float() * self.hard_negative_weight
        else:
            weighted_logits = logits
            
        # InfoNCE 손실 계산 (각 텍스트에 대해)
        # 마진 추가: 마진보다 작은 positive 쌍 유사도에는 패널티 부여하지 않음
        pos_logits = (logits - self.margin) * pos_mask
        pos_logits = torch.clamp(pos_logits, min=0)  # 마진보다 크면 그대로, 작으면 0
        
        # 수정된 유사도 행렬: positive에는 마진 적용, hard negative에는 가중치 적용
        final_logits = pos_logits + weighted_logits * (1.0 - pos_mask)
        
        # 텍스트→이미지 방향 손실
        labels = torch.arange(batch_size, device=device)
        t2i_loss = F.cross_entropy(final_logits, labels)
        
        # 이미지→텍스트 방향 손실
        i2t_loss = F.cross_entropy(final_logits.t(), labels)
        
        # 양방향 손실 결합
        loss = (t2i_loss + i2t_loss) / 2.0
        
        return loss
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch["img_id"]
        
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        loss = self.compute_multicaption_loss(image_embeds, text_embeds, img_ids)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch["img_id"]
        
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        val_loss = self.compute_multicaption_loss(image_embeds, text_embeds, img_ids)
        
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
        # 평균 검증 손실 기록
        val_losses = torch.stack([o["val_loss"] for o in self._val_outputs])
        avg_val_loss = val_losses.mean()
        self.log("val_loss", avg_val_loss, prog_bar=True)
        
        # 이미지 ID별 대표 임베딩 생성 (각 이미지당 하나의 임베딩만 사용)
        all_image_embeds = torch.cat([o["image_embeds"] for o in self._val_outputs], dim=0)
        all_text_embeds = torch.cat([o["text_embeds"] for o in self._val_outputs], dim=0)
        all_img_ids = torch.cat([o["img_ids"] for o in self._val_outputs], dim=0)
        
        # 유니크한 이미지 ID 추출
        unique_img_ids = torch.unique(all_img_ids)
        
        # 유니크한 이미지 ID별로 하나의 임베딩만 사용하여 평가 진행
        unique_image_embeds = []
        unique_text_embeds = []
        
        for img_id in unique_img_ids:
            # 현재 이미지 ID의 마스크
            mask = (all_img_ids == img_id)
            
            # 해당 이미지의 모든 임베딩 추출
            img_embeds = all_image_embeds[mask]
            txt_embeds = all_text_embeds[mask]
            
            # 대표 임베딩으로 첫 번째 임베딩 사용
            unique_image_embeds.append(img_embeds[0:1])
            unique_text_embeds.append(txt_embeds[0:1])
        
        # 대표 임베딩 텐서 생성
        unique_image_embeds = torch.cat(unique_image_embeds, dim=0)
        unique_text_embeds = torch.cat(unique_text_embeds, dim=0)
        
        # 유니크한 이미지 임베딩으로 유사도 행렬 계산
        similarity_matrix = torch.matmul(unique_text_embeds, unique_image_embeds.t())
        
        # 유니크한 이미지 ID로 리콜 계산
        recall_at_k = self.compute_recall(similarity_matrix, ks=[1, 5, 10])
        for k, v in recall_at_k.items():
            self.log(f"val_recall@{k}", v, prog_bar=True)
        
        self._val_outputs.clear()

    def test_step(self, batch, batch_idx):
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch["img_id"]
        
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        return {"image_embeds": image_embeds, "text_embeds": text_embeds, "img_ids": img_ids}

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_image_embeds.append(outputs["image_embeds"])
        self.test_text_embeds.append(outputs["text_embeds"])
        self._val_outputs.append(outputs)

    def on_test_epoch_end(self):
        # 이미지 ID별 대표 임베딩 생성 (각 이미지당 하나의 임베딩만 사용)
        all_image_embeds = torch.cat([o["image_embeds"] for o in self._val_outputs], dim=0)
        all_text_embeds = torch.cat([o["text_embeds"] for o in self._val_outputs], dim=0)
        all_img_ids = torch.cat([o["img_ids"] for o in self._val_outputs], dim=0)
        
        # 유니크한 이미지 ID 추출
        unique_img_ids = torch.unique(all_img_ids)
        
        # 유니크한 이미지 ID별로 하나의 임베딩만 사용
        unique_image_embeds = []
        unique_text_embeds = []
        
        for img_id in unique_img_ids:
            # 현재 이미지 ID의 마스크
            mask = (all_img_ids == img_id)
            
            # 해당 이미지의 모든 임베딩 추출
            img_embeds = all_image_embeds[mask]
            txt_embeds = all_text_embeds[mask]
            
            # 대표 임베딩으로 첫 번째 임베딩 사용
            unique_image_embeds.append(img_embeds[0:1])
            unique_text_embeds.append(txt_embeds[0:1])
        
        # 대표 임베딩 텐서 생성
        unique_image_embeds = torch.cat(unique_image_embeds, dim=0)
        unique_text_embeds = torch.cat(unique_text_embeds, dim=0)
        
        # 유니크한 이미지 임베딩으로 유사도 행렬 계산
        similarity_matrix = torch.matmul(unique_text_embeds, unique_image_embeds.t())
        
        # 유니크한 이미지 ID로 리콜 계산
        recall_at_k = self.compute_recall(similarity_matrix, ks=[1, 5, 10])
        for k, v in recall_at_k.items():
            self.log(f"test_recall@{k}", v, prog_bar=True)
        
        print(f"[테스트 결과] 테스트 Recall: {recall_at_k}")
        self.test_image_embeds.clear()
        self.test_text_embeds.clear()
        self._val_outputs.clear()

    def compute_recall(self, similarity_matrix, ks=[1, 5, 10]):
        # 각 쿼리(행)에 대해 ground-truth 인덱스가 top-k 검색 인덱스에 있는지 확인
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
        # 프로젝션 레이어만 학습
        optimizer = torch.optim.AdamW([
            {"params": self.image_proj.parameters()},
            {"params": self.text_proj.parameters()}
        ], lr=self.learning_rate, weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        
        return [optimizer], [scheduler]

# -----------------------------------------------------------------------------
# 5. Load Flickr30k Dataset and Prepare DataModule
# -----------------------------------------------------------------------------
dataset = load_dataset("nlphuji/flickr30k")["test"]
train_dataset = dataset.filter(lambda x: x["split"] == "train")
valid_dataset = dataset.filter(lambda x: x["split"] == "val")
test_dataset = dataset.filter(lambda x: x["split"] == "test")

data_module = Flickr30KDataModule(
    train_dataset_hf=train_dataset,
    valid_dataset_hf=valid_dataset,
    test_dataset_hf=test_dataset,
    batch_size=32,
    num_workers=4,
    max_length=96
)
data_module.setup("fit")

# -----------------------------------------------------------------------------
# 6. Trainer, Logger, and Callback Setup
# -----------------------------------------------------------------------------
logger = TensorBoardLogger(
    save_dir="ImageRetrieving_improved",
    name="Flickr30k_CLIP_improved"
)

early_stopping_callback = EarlyStopping(
    monitor="val_recall@1",
    patience=10,
    mode="max",
    min_delta=0.001
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_recall@1",
    mode="max",
    dirpath="checkpoints/improved",
    filename="flickr30k-clip-improved-{epoch:02d}-{val_recall@1:.4f}",
    save_top_k=5,
    save_weights_only=True
)

lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    logger=logger,
    callbacks=[early_stopping_callback, checkpoint_callback, lr_monitor],
    gradient_clip_val=1.0,
    accumulate_grad_batches=2,
    log_every_n_steps=50,
    deterministic=True
)

# -----------------------------------------------------------------------------
# 7. Model Initialization and Training
# -----------------------------------------------------------------------------
model = ImprovedImageTextModel(
    image_encoder_name="microsoft/swin-base-patch4-window7-224",
    text_encoder_name="roberta-large",
    embed_dim=256,
    temperature=0.07,
    learning_rate=1e-4,
    use_hard_negatives=True,
    hard_negative_weight=10.0,
    margin=0.2
)

# 훈련 실행
trainer.fit(model, data_module)

# 테스트 실행 및 결과 저장
test_results = trainer.test(model, datamodule=data_module)

with open("test_recall_improved.txt", "w") as f:
    f.write("테스트 리콜 지표 (개선된 모델):\n")
    for result in test_results:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")

print("테스트 평가 결과가 'test_recall_improved.txt' 파일에 저장되었습니다.") 