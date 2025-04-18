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
# 2. Flickr30K Multi-Caption Dataset for SupCon
# -----------------------------------------------------------------------------
class Flickr30KMultiCaptionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_transform, max_length=64, num_views=2):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.num_views = num_views
        
        # 이미지 ID를 기준으로 그룹화된 데이터 구성
        self.image_to_captions = {}
        for rec_idx, record in enumerate(self.hf_dataset):
            captions = record["caption"]
            self.image_to_captions[rec_idx] = captions
        
        # 유니크 이미지 ID 리스트
        self.image_ids = list(self.image_to_captions.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        record = self.hf_dataset[image_id]
        pil_image = record["image"]
        captions = record["caption"]
        
        # 이미지 여러 뷰 생성 (Augmentation 적용)
        image_views = []
        try:
            for _ in range(self.num_views):
                augmented_image = self.image_transform(pil_image)
                image_views.append(augmented_image)
        except Exception as e:
            print(f"이미지 변환 중 오류 발생 (ID: {image_id}): {e}")
            # 오류 발생 시 빈 텐서 대신 검은색 이미지 생성
            black_image = torch.zeros(3, 224, 224)
            image_views = [black_image] * self.num_views
        
        # 토큰화된 캡션 구성
        tokenized_captions = []
        
        # 캡션이 비어있는 경우 처리
        if not captions:
            # 더미 캡션 추가
            dummy_caption = "no caption available"
            tokenized = self.tokenizer(
                dummy_caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            
            tokenized_captions.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "caption": dummy_caption
            })
        else:
            # 정상적인 캡션 처리
            for caption in captions:
                try:
                    tokenized = self.tokenizer(
                        caption,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    
                    input_ids = tokenized["input_ids"].squeeze(0)
                    attention_mask = tokenized["attention_mask"].squeeze(0)
                    
                    tokenized_captions.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "caption": caption
                    })
                except Exception as e:
                    print(f"캡션 토큰화 중 오류 발생 (ID: {image_id}): {e}")
                    # 오류 발생 시 빈 캡션 대신 처리
                    continue
        
        # 캡션 처리 중 모든 오류가 발생한 경우 처리
        if not tokenized_captions:
            # 더미 캡션 추가
            dummy_caption = "error processing caption"
            tokenized = self.tokenizer(
                dummy_caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            
            tokenized_captions.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "caption": dummy_caption
            })
        
        return {
            "image_views": image_views,
            "tokenized_captions": tokenized_captions,
            "img_id": image_id,
            "num_captions": len(tokenized_captions)
        }

# 커스텀 콜레이트 함수 (배치 구성을 위한)
def supcon_collate_fn(batch):
    image_views_list = []
    input_ids_list = []
    attention_mask_list = []
    img_ids = []
    captions = []
    caption_ids = []  # 어떤 이미지에 속하는 캡션인지 추적
    
    # 각 배치 항목 처리
    for i, item in enumerate(batch):
        img_id = item["img_id"]
        
        # 이미지 뷰 수집 - 뷰 수에 맞게 처리
        for view in item["image_views"]:
            image_views_list.append(view)
        
        # 이미지별 모든 캡션 수집
        for j, cap_item in enumerate(item["tokenized_captions"]):
            input_ids_list.append(cap_item["input_ids"])
            attention_mask_list.append(cap_item["attention_mask"])
            captions.append(cap_item["caption"])
            img_ids.append(img_id)
            # 배치 내 이미지 인덱스 저장 (각 이미지마다 고유한 ID)
            caption_ids.append(i)
    
    # 디버깅용 정보 출력 (랜덤하게 10% 확률로만 출력)
    if random.random() < 0.1:
        print(f"\n[Collate] 이미지 뷰 리스트 길이: {len(image_views_list)}")
        print(f"[Collate] 캡션 ID 리스트 길이: {len(caption_ids)}")
        if image_views_list:
            print(f"[Collate] 이미지 뷰 모양: {image_views_list[0].shape}")
        print(f"[Collate] 배치 크기: {len(batch)}\n")
    
    # 빈 배치 처리
    if not image_views_list:
        # 빈 텐서 반환
        return {
            "image_views": torch.empty((0, 3, 224, 224)),
            "input_ids": torch.empty((0, 96), dtype=torch.long),
            "attention_mask": torch.empty((0, 96), dtype=torch.long),
            "img_ids": torch.empty(0, dtype=torch.long),
            "caption_ids": torch.empty(0, dtype=torch.long),
            "captions": []
        }
    
    # 이미지 뷰와 텍스트 데이터 일치 확인
    num_views_per_image = len(batch[0]["image_views"]) if batch else 0
    total_expected_views = len(batch) * num_views_per_image
    
    if len(image_views_list) != total_expected_views:
        print(f"경고: 이미지 뷰 수 불일치. 예상: {total_expected_views}, 실제: {len(image_views_list)}")
    
    # 텐서로 변환하고 반환
    return {
        "image_views": torch.stack(image_views_list) if image_views_list else torch.empty((0, 3, 224, 224)),
        "input_ids": torch.stack(input_ids_list) if input_ids_list else torch.empty((0, 96), dtype=torch.long),
        "attention_mask": torch.stack(attention_mask_list) if attention_mask_list else torch.empty((0, 96), dtype=torch.long),
        "img_ids": torch.tensor(img_ids, dtype=torch.long) if img_ids else torch.empty(0, dtype=torch.long),
        "caption_ids": torch.tensor(caption_ids, dtype=torch.long) if caption_ids else torch.empty(0, dtype=torch.long),
        "captions": captions
    }

# -----------------------------------------------------------------------------
# 3. DataModule
# -----------------------------------------------------------------------------
class Flickr30KDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dataset_hf,
                 valid_dataset_hf,
                 test_dataset_hf,
                 batch_size=16,
                 num_workers=4,
                 max_length=96,
                 num_views=2):
        super().__init__()
        self.train_dataset_hf = train_dataset_hf
        self.valid_dataset_hf = valid_dataset_hf
        self.test_dataset_hf = test_dataset_hf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.num_views = num_views

        # RoBERTa 토크나이저
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        # 학습용 이미지 변환 (강화된 Augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
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
                max_length=self.max_length,
                num_views=self.num_views
            )
            
            self.valid_dataset = Flickr30KMultiCaptionDataset(
                self.valid_dataset_hf,
                tokenizer=self.tokenizer,
                image_transform=self.eval_transform,
                max_length=self.max_length,
                num_views=1  # 검증에서는 1개 뷰 사용
            )
            
            print(f"학습 데이터셋 크기: {len(self.train_dataset)}")
            print(f"검증 데이터셋 크기: {len(self.valid_dataset)}")
            
        if stage == "test" or stage is None:
            self.test_dataset = Flickr30KMultiCaptionDataset(
                self.test_dataset_hf,
                tokenizer=self.tokenizer,
                image_transform=self.eval_transform,
                max_length=self.max_length,
                num_views=1  # 테스트에서는 1개 뷰 사용
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
            collate_fn=supcon_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=supcon_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=supcon_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

# -----------------------------------------------------------------------------
# 4. Supervised Contrastive Learning Image-Text Module
# -----------------------------------------------------------------------------
class SupConImageTextModel(pl.LightningModule):
    def __init__(self,
                 image_encoder_name="microsoft/swin-base-patch4-window7-224",
                 text_encoder_name="roberta-large",
                 embed_dim=256,
                 temperature=0.1,  # 낮은 온도 설정 (SupCon 논문 권장)
                 learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

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
        
        # 향상된 프로젝션 헤드 (MLP)
        self.image_proj = nn.Sequential(
            nn.Linear(image_hidden_size, image_hidden_size),
            nn.ReLU(),
            nn.Linear(image_hidden_size, embed_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size),
            nn.ReLU(),
            nn.Linear(text_hidden_size, embed_dim)
        )

        # 온도 파라미터
        self.temperature = temperature
        self.learning_rate = learning_rate

        # 검증/테스트 임베딩 로깅
        self._val_outputs = []
        self.test_image_embeds = []
        self.test_text_embeds = []

    def forward(self, images, input_ids, attention_mask):
        batch_size = images.size(0)
        
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

    def compute_supcon_loss(self, features, labels):
        """
        Supervised Contrastive Loss 구현
        features: 임베딩 특징 벡터
        labels: 각 특징 벡터의 레이블 (같은 이미지/캡션에 대한 것들은 같은 레이블)
        """
        device = features.device
        batch_size = features.size(0)
        
        # 유사도 행렬 계산 (코사인 유사도)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 자기 자신과의 유사도 마스킹
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        similarity_matrix.masked_fill_(mask_self, -float('inf'))
        
        # 라벨 마스크 생성: 같은 레이블(동일 이미지/캡션에서 온 것)이면 1, 아니면 0
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        
        # 자기 자신 마스킹 후 positive 마스크
        # 크기 불일치 문제 해결을 위해 직접 대각선 요소를 0으로 설정
        mask_pos.fill_diagonal_(0)
        
        # 각 인스턴스에 대한 positive 쌍 수 계산
        num_positives_per_row = mask_pos.sum(1)
        
        # log_prob 계산: 분자 - log(분모)
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Positive 쌍이 없는 경우 처리
        mask_valid = (num_positives_per_row > 0).float()
        
        # Positive 쌍이 있는 경우만 계산
        pos_log_prob = (mask_pos * log_prob).sum(1) / (num_positives_per_row + 1e-8)
        
        # 최종 손실 계산
        loss = -1 * (mask_valid * pos_log_prob).sum() / (mask_valid.sum() + 1e-8)
        
        return loss

    def training_step(self, batch, batch_idx):
        images = batch["image_views"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        caption_ids = batch["caption_ids"]  # 이미지 ID (같은 이미지에서 온 캡션들은 같은 ID)
        
        # 디버깅 정보 출력 (첫 몇 배치에만)
        if batch_idx < 2:
            print(f"[학습] 이미지 크기: {images.size()}, 캡션 ID 크기: {caption_ids.size()}")
        
        # 이미지/텍스트 임베딩 계산
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        
        # 모든 임베딩을 하나로 결합
        all_embeds = torch.cat([image_embeds, text_embeds], dim=0)
        
        # 이미지와 텍스트에 대한 라벨 결합 (같은 이미지/캡션 쌍은 같은 레이블)
        # 크기 일치를 확인하여 라벨 생성
        num_images = image_embeds.size(0)
        num_texts = text_embeds.size(0)
        total_embeds = num_images + num_texts
        
        # 라벨 생성 방식 변경: 이미지와 텍스트 임베딩 크기에 맞게 새로운 라벨 생성
        all_labels = torch.zeros(total_embeds, dtype=torch.long, device=self.device)
        
        # 이미지 당 뷰 수 계산
        views_per_img = num_images // len(caption_ids) if len(caption_ids) > 0 else 1
        
        # 각 이미지와 해당 텍스트에 동일한 라벨 부여
        for i in range(len(caption_ids)):
            # 이미지 임베딩에 라벨 부여
            for v in range(views_per_img):
                img_idx = i * views_per_img + v
                if img_idx < num_images:
                    all_labels[img_idx] = i
            
            # 해당 텍스트 임베딩에 라벨 부여
            if i < num_texts:
                all_labels[num_images + i] = i
        
        # 크기 일치 확인
        assert all_embeds.size(0) == all_labels.size(0), f"임베딩({all_embeds.size(0)})과 라벨({all_labels.size(0)}) 크기 불일치"
        
        # SupCon 손실 계산
        loss = self.compute_supcon_loss(all_embeds, all_labels)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image_views"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch["img_ids"]
        caption_ids = batch["caption_ids"]
        
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        
        # 디버깅 정보 출력
        print(f"[검증] 이미지 임베딩 크기: {image_embeds.size()}, 텍스트 임베딩 크기: {text_embeds.size()}")
        print(f"[검증] caption_ids 크기: {caption_ids.size()}")
        
        # 검증에서도 SupCon 손실 계산
        all_embeds = torch.cat([image_embeds, text_embeds], dim=0)
        
        # 크기 일치를 확인하여 라벨 생성
        num_images = image_embeds.size(0)
        num_texts = text_embeds.size(0)
        total_embeds = num_images + num_texts
        
        # 라벨 생성 방식 변경: 이미지와 텍스트 임베딩 크기에 맞게 새로운 라벨 생성
        # 같은 이미지-텍스트 쌍에 동일한 ID 부여
        all_labels = torch.zeros(total_embeds, dtype=torch.long, device=self.device)
        
        # 이미지 당 뷰 수 계산 (검증에서는 보통 1)
        views_per_img = num_images // len(caption_ids) if len(caption_ids) > 0 else 1
        
        # 각 이미지와 해당 텍스트에 동일한 라벨 부여
        for i in range(len(caption_ids)):
            # 이미지 임베딩에 라벨 부여
            for v in range(views_per_img):
                img_idx = i * views_per_img + v
                if img_idx < num_images:
                    all_labels[img_idx] = i
            
            # 해당 텍스트 임베딩에 라벨 부여
            if i < num_texts:
                all_labels[num_images + i] = i
        
        # 크기 일치 확인
        assert all_embeds.size(0) == all_labels.size(0), f"임베딩({all_embeds.size(0)})과 라벨({all_labels.size(0)}) 크기 불일치"
        
        val_loss = self.compute_supcon_loss(all_embeds, all_labels)
        
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
        if not self._val_outputs:
            print("경고: 검증 출력이 비어 있습니다.")
            return
        
        val_losses = torch.stack([o["val_loss"] for o in self._val_outputs])
        avg_val_loss = val_losses.mean()
        self.log("val_loss", avg_val_loss, prog_bar=True)
        
        # 임베딩을 연결하여 리콜 메트릭 계산
        try:
            all_image_embeds = torch.cat([o["image_embeds"] for o in self._val_outputs], dim=0)
            all_text_embeds = torch.cat([o["text_embeds"] for o in self._val_outputs], dim=0)
            
            # 이미지와 텍스트 임베딩을 그룹화하여 유니크한 이미지 ID별로 평균 계산
            # 이 부분은 데이터셋 구조에 따라 조정이 필요할 수 있음
            
            # 임베딩이 비어있는지 확인
            if all_image_embeds.size(0) == 0 or all_text_embeds.size(0) == 0:
                print("경고: 연결된 검증 임베딩이 비어 있습니다.")
                return
            
            # 리콜 계산을 위한 유사도 행렬 생성
            similarity_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
            
            recall_at_k = self.compute_recall(similarity_matrix, ks=[1, 5, 10])
            for k, v in recall_at_k.items():
                self.log(f"val_recall@{k}", v, prog_bar=True)
        except Exception as e:
            print(f"검증 리콜 계산 중 오류 발생: {e}")
        
        self._val_outputs.clear()

    def test_step(self, batch, batch_idx):
        images = batch["image_views"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch.get("img_ids", None)
        
        # 빈 배치 체크
        if images.size(0) == 0 or input_ids.size(0) == 0:
            # 빈 임베딩 반환
            device = self.device
            empty_embeds = torch.empty((0, self.hparams.embed_dim), device=device)
            return {"image_embeds": empty_embeds, "text_embeds": empty_embeds}
        
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        return {"image_embeds": image_embeds, "text_embeds": text_embeds}

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_image_embeds.append(outputs["image_embeds"])
        self.test_text_embeds.append(outputs["text_embeds"])

    def on_test_epoch_end(self):
        # 비어있는 리스트 체크
        if not self.test_image_embeds or not self.test_text_embeds:
            print("경고: 테스트 임베딩이 비어 있습니다.")
            return
        
        all_image_embeds = torch.cat(self.test_image_embeds, dim=0)
        all_text_embeds = torch.cat(self.test_text_embeds, dim=0)
        
        # 임베딩이 비어있는지 확인
        if all_image_embeds.size(0) == 0 or all_text_embeds.size(0) == 0:
            print("경고: 연결된 테스트 임베딩이 비어 있습니다.")
            return
        
        similarity_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
        
        recall_at_k = self.compute_recall(similarity_matrix, ks=[1, 5, 10])
        for k, v in recall_at_k.items():
            self.log(f"test_recall@{k}", v, prog_bar=True)
        
        print(f"[테스트 결과] 테스트 Recall: {recall_at_k}")
        self.test_image_embeds.clear()
        self.test_text_embeds.clear()

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

print(f"[데이터셋 정보] 원본 크기: {len(dataset)}")
print(f"[데이터셋 정보] 학습 데이터셋: {len(train_dataset)} 샘플")
print(f"[데이터셋 정보] 검증 데이터셋: {len(valid_dataset)} 샘플")
print(f"[데이터셋 정보] 테스트 데이터셋: {len(test_dataset)} 샘플")

# 배치 크기 감소 (더 다양한 negative 샘플)
data_module = Flickr30KDataModule(
    train_dataset_hf=train_dataset,
    valid_dataset_hf=valid_dataset,
    test_dataset_hf=test_dataset,
    batch_size=16,  # 배치 크기 감소
    num_workers=4,
    max_length=96,
    num_views=2  # 각 이미지당 뷰 수
)
data_module.setup("fit")

# -----------------------------------------------------------------------------
# 6. Trainer, Logger, and Callback Setup
# -----------------------------------------------------------------------------
logger = TensorBoardLogger(
    save_dir="ImageRetrieving_supcon",
    name="Flickr30k_CLIP_supcon"
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
    dirpath="checkpoints/supcon",
    filename="flickr30k-clip-supcon-{epoch:02d}-{val_recall@1:.4f}",
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
model = SupConImageTextModel(
    image_encoder_name="microsoft/swin-base-patch4-window7-224",
    text_encoder_name="roberta-large",
    embed_dim=256,
    temperature=0.1,  # SupCon용으로 낮은 온도 사용
    learning_rate=1e-4
)

# 훈련 실행
trainer.fit(model, data_module)

# 테스트 실행 및 결과 저장
test_results = trainer.test(model, datamodule=data_module)

with open("test_recall_supcon.txt", "w") as f:
    f.write("테스트 리콜 지표 (SupCon 모델):\n")
    for result in test_results:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")

print("테스트 평가 결과가 'test_recall_supcon.txt' 파일에 저장되었습니다.")