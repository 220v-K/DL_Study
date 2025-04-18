import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
from collections import defaultdict

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
# 2. Flickr30K 데이터셋 및 이미지별 캡션 그룹화
# -----------------------------------------------------------------------------
class Flickr30kDatasetWithCaptionPooling(Dataset):
    """
    Flickr30k 데이터셋을 로드하고 이미지별 여러 캡션을 그룹화하는 클래스
    각 이미지마다 여러 캡션이 있으며, 이들을 평균 풀링하여 사용
    """
    def __init__(self, hf_dataset, split='train', image_size=224, max_length=77, max_captions=5):
        self.split = split
        self.hf_dataset = hf_dataset
        self.max_captions = max_captions  # 이미지당 최대 캡션 수 (기본값: 5)
        
        # RoBERTa 토크나이저
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.max_length = max_length
        
        # 이미지 전처리 변환
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # 이미지별 캡션 그룹화
        self.image_captions = self._group_captions_by_image()
        
        # 데이터셋 인덱스 구성
        self.image_ids = list(self.image_captions.keys())
        
        print(f"총 {len(self.image_ids)}개의 유니크한 이미지와 평균 {len(self.hf_dataset)} / {len(self.image_ids):.1f}개의 캡션/이미지가 있습니다.")
        
    def _group_captions_by_image(self):
        """이미지 ID별로 캡션 그룹화"""
        image_captions = defaultdict(list)
        
        # 모든 데이터를 순회하며 이미지별로 캡션 그룹화
        for idx, item in enumerate(self.hf_dataset):
            img_id = str(idx)  # 원래 인덱스를 ID로 사용
            
            # 이미지별 최대 캡션 수 제한 (이미 최대치에 도달한 이미지는 건너뜀)
            captions = item["caption"]
            img_id = item.get("image_id", idx)  # 이미지 ID가 있으면 사용, 없으면 인덱스 사용
            
            # 캡션이 리스트가 아니면 리스트로 변환
            if not isinstance(captions, list):
                captions = [captions]
                
            # 최대 캡션 수 제한
            captions = captions[:self.max_captions]
            
            image_captions[str(img_id)] = {
                'captions': captions,
                'image': item['image']
            }
        
        # 캡션 수가 다른 이미지들에 대한 통계
        caption_counts = {}
        for img_id, data in image_captions.items():
            count = len(data['captions'])
            caption_counts[count] = caption_counts.get(count, 0) + 1
        
        print(f"이미지당 캡션 수 분포: {caption_counts}")
        
        return image_captions
            
    def __len__(self):
        """데이터셋 크기 = 유니크한 이미지 수"""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        하나의 이미지와 그 이미지에 해당하는 모든 캡션을 반환
        각 캡션은 토큰화된 형태로 반환
        """
        img_id = self.image_ids[idx]
        data = self.image_captions[img_id]
        captions = data['captions']
        
        # 이미지 로드 및 전처리
        try:
            image = data['image']
            image = self.transform(image)
        except Exception as e:
            print(f"이미지 로드 오류: {img_id} - {e}")
            # 오류 발생시 검은색 이미지로 대체
            image = torch.zeros(3, 224, 224)
        
        # 모든 캡션 토큰화 (최대 max_captions 개수만 사용)
        all_captions = []
        all_input_ids = []
        all_attention_masks = []
        
        # 최대 캡션 수 제한
        num_captions = min(len(captions), self.max_captions)
        
        for i in range(num_captions):
            caption = captions[i]
            all_captions.append(caption)
            
            tokenized = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            all_input_ids.append(tokenized['input_ids'].squeeze(0))
            all_attention_masks.append(tokenized['attention_mask'].squeeze(0))
        
        # 텐서로 변환
        input_ids_tensor = torch.stack(all_input_ids)
        attention_mask_tensor = torch.stack(all_attention_masks)
        
        return {
            "image": image,
            "input_ids": input_ids_tensor,  # [num_captions, max_length]
            "attention_mask": attention_mask_tensor,  # [num_captions, max_length]
            "img_id": img_id,
            "captions": all_captions,
            "num_captions": num_captions
        }

# 커스텀 collate 함수 - 다양한 캡션 수 처리
def caption_pooling_collate_fn(batch):
    # 기본 배치 처리
    batch_dict = {
        "image": [],
        "input_ids": [],
        "attention_mask": [],
        "img_id": [],
        "captions": [],
        "num_captions": []
    }
    
    for item in batch:
        batch_dict["image"].append(item["image"])
        batch_dict["input_ids"].append(item["input_ids"])
        batch_dict["attention_mask"].append(item["attention_mask"])
        batch_dict["img_id"].append(item["img_id"])
        batch_dict["captions"].append(item["captions"])
        batch_dict["num_captions"].append(item["num_captions"])
    
    # 이미지, 이미지 ID, 캡션 수 처리
    batch_dict["image"] = torch.stack(batch_dict["image"])
    batch_dict["img_id"] = batch_dict["img_id"]  # 이미 리스트 형식
    batch_dict["num_captions"] = batch_dict["num_captions"]  # 이미 리스트 형식
    
    # 기존 코드 유지 (모든 이미지가 동일한 수의 캡션을 가짐)
    batch_dict["input_ids"] = torch.stack(batch_dict["input_ids"])
    batch_dict["attention_mask"] = torch.stack(batch_dict["attention_mask"])
    
    return batch_dict

# -----------------------------------------------------------------------------
# 3. DataModule
# -----------------------------------------------------------------------------
class Flickr30kDataModuleWithCaptionPooling(pl.LightningDataModule):
    def __init__(self,
                train_dataset_hf,
                valid_dataset_hf,
                test_dataset_hf,
                batch_size=128,  # 배치 사이즈 증가
                num_workers=4,
                image_size=224,
                max_length=77,
                max_captions=5):
        super().__init__()
        self.train_dataset_hf = train_dataset_hf
        self.valid_dataset_hf = valid_dataset_hf
        self.test_dataset_hf = test_dataset_hf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_length = max_length
        self.max_captions = max_captions

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # 훈련 데이터셋 로드
            self.train_dataset = Flickr30kDatasetWithCaptionPooling(
                hf_dataset=self.train_dataset_hf,
                split='train',
                image_size=self.image_size,
                max_length=self.max_length,
                max_captions=self.max_captions
            )
            
            # 검증 데이터셋 로드
            self.valid_dataset = Flickr30kDatasetWithCaptionPooling(
                hf_dataset=self.valid_dataset_hf,
                split='val',
                image_size=self.image_size,
                max_length=self.max_length,
                max_captions=self.max_captions
            )
            
            print(f"학습 데이터셋 크기: {len(self.train_dataset)}")
            print(f"검증 데이터셋 크기: {len(self.valid_dataset)}")
            
        if stage == "test" or stage is None:
            # 테스트 데이터셋 로드
            self.test_dataset = Flickr30kDatasetWithCaptionPooling(
                hf_dataset=self.test_dataset_hf,
                split='test',
                image_size=self.image_size,
                max_length=self.max_length,
                max_captions=self.max_captions
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
            collate_fn=caption_pooling_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=caption_pooling_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=caption_pooling_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

# -----------------------------------------------------------------------------
# 4. 캡션 평균 풀링을 사용한 이미지-텍스트 모델
# -----------------------------------------------------------------------------
class EnhancedImageTextModel(pl.LightningModule):
    def __init__(self,
                 image_encoder_name="microsoft/swin-base-patch4-window7-224",
                 text_encoder_name="roberta-large",
                 embed_dim=256,
                 temperature=0.07,
                 learning_rate=1e-4,
                 should_train_image_encoder=True,
                 should_train_text_encoder=True,
                 hard_mining_factor=0.25,  # 하드 네거티브 비율
                 use_dual_softmax=True):  # 듀얼 소프트맥스 손실 사용 여부
        super().__init__()
        self.save_hyperparameters()

        # 하이퍼파라미터 저장
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.should_train_image_encoder = should_train_image_encoder
        self.should_train_text_encoder = should_train_text_encoder
        self.hard_mining_factor = hard_mining_factor
        self.use_dual_softmax = use_dual_softmax

        # 인코더 초기화
        self.image_encoder = SwinModel.from_pretrained(
            image_encoder_name,
            cache_dir="./model_cache"
        )
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_name)
        
        # 인코더 학습 여부 설정 - 처음부터 지정된 레이어들만 학습
        # 먼저 전체 인코더 동결
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # should_train_image_encoder가 True인 경우, 이미지 인코더의 특정 레이어 학습 활성화
        if should_train_image_encoder:
            # Swin 트랜스포머의 마지막 6개 레이어 학습 활성화
            image_layers_to_train = [
                "layers.3.blocks.1", "layers.3.blocks.0", 
                "layers.2.blocks.1", "layers.2.blocks.0",
                "layers.1.blocks.1", "layers.1.blocks.0"
            ]
            
            for name, param in self.image_encoder.named_parameters():
                if any(layer in name for layer in image_layers_to_train):
                    param.requires_grad = True
        
        # should_train_text_encoder가 True인 경우, 텍스트 인코더의 특정 레이어 학습 활성화
        if should_train_text_encoder:
            # RoBERTa의 마지막 6개 레이어 학습 활성화
            text_layers_to_train = [
                "encoder.layer.23", "encoder.layer.22", 
                "encoder.layer.21", "encoder.layer.20",
                "encoder.layer.19", "encoder.layer.18"
            ]
            
            for name, param in self.text_encoder.named_parameters():
                if any(layer in name for layer in text_layers_to_train):
                    param.requires_grad = True

        # 프로젝션 레이어 설정
        image_hidden_size = self.image_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        
        # 향상된 프로젝션 레이어 (MLP)
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

        # 검증/테스트 임베딩 로깅
        self._val_outputs = []
        self.test_image_embeds = []
        self.test_text_embeds = []
        
        # 학습 대상 파라미터 수 출력
        self._log_trainable_params()
        
    def _log_trainable_params(self):
        """학습 가능한 파라미터 수 로깅"""
        image_encoder_params = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)
        text_encoder_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        image_proj_params = sum(p.numel() for p in self.image_proj.parameters())
        text_proj_params = sum(p.numel() for p in self.text_proj.parameters())
        
        total_params = image_encoder_params + text_encoder_params + image_proj_params + text_proj_params
        
        print(f"학습 가능한 파라미터: ")
        print(f"  이미지 인코더: {image_encoder_params:,} ({image_encoder_params / total_params:.1%})")
        print(f"  텍스트 인코더: {text_encoder_params:,} ({text_encoder_params / total_params:.1%})")
        print(f"  이미지 프로젝션: {image_proj_params:,} ({image_proj_params / total_params:.1%})")
        print(f"  텍스트 프로젝션: {text_proj_params:,} ({text_proj_params / total_params:.1%})")
        print(f"  총 파라미터: {total_params:,}")
        
    def encode_image(self, images):
        """이미지 인코딩 및 임베딩 추출"""
        image_outputs = self.image_encoder(pixel_values=images)
        image_features = image_outputs.last_hidden_state
        
        # 이미지 임베딩 - 평균 풀링
        image_embeds = image_features.mean(dim=1)  # [batch_size, hidden_size]
        image_embeds = self.image_proj(image_embeds)  # [batch_size, embed_dim]
        
        # L2 정규화
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        
        return image_embeds
    
    def encode_text_batch(self, input_ids_batch, attention_mask_batch):
        """
        텍스트 인코딩 및 임베딩 추출 (배치 단위)
        input_ids_batch: [batch_size, num_captions, max_length]
        attention_mask_batch: [batch_size, num_captions, max_length]
        """
        batch_size, num_captions, max_length = input_ids_batch.shape
        
        # 모든 캡션을 개별적으로 인코딩하기 위해 차원 변경
        input_ids_flat = input_ids_batch.view(batch_size * num_captions, max_length)
        attention_mask_flat = attention_mask_batch.view(batch_size * num_captions, max_length)
        
        # 텍스트 인코딩
        text_outputs = self.text_encoder(
            input_ids=input_ids_flat, 
            attention_mask=attention_mask_flat
        )
        
        # CLS 토큰의 임베딩 추출
        text_features = text_outputs.last_hidden_state[:, 0]  # [batch_size * num_captions, hidden_size]
        
        # 프로젝션 및 정규화
        text_embeds = self.text_proj(text_features)  # [batch_size * num_captions, embed_dim]
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        # 원래 배치 구조로 복원 및 평균 풀링
        text_embeds = text_embeds.view(batch_size, num_captions, -1)  # [batch_size, num_captions, embed_dim]
        text_embeds = text_embeds.mean(dim=1)  # [batch_size, embed_dim]
        
        # 다시 정규화 (평균 풀링 후 정규화 필요)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        return text_embeds
        
    def forward(self, batch):
        """
        이미지와 텍스트 인코딩 및 임베딩 추출
        batch에는 이미지와 해당 이미지의 여러 캡션이 포함됨
        """
        images = batch["image"]  # [batch_size, 3, H, W]
        input_ids = batch["input_ids"]  # [batch_size, num_captions, max_length]
        attention_mask = batch["attention_mask"]  # [batch_size, num_captions, max_length]
        
        # 이미지 인코딩
        image_embeds = self.encode_image(images)  # [batch_size, embed_dim]
        
        # 텍스트 인코딩 (여러 캡션 평균 풀링)
        text_embeds = self.encode_text_batch(input_ids, attention_mask)  # [batch_size, embed_dim]
        
        return image_embeds, text_embeds

    def compute_contrastive_loss_with_hard_negatives(self, image_embeds, text_embeds):
        """
        하드 네거티브 마이닝을 적용한 대조 손실 계산
        """
        device = image_embeds.device
        batch_size = image_embeds.size(0)
        
        # 전체 유사도 행렬 계산
        sim_matrix = torch.matmul(text_embeds, image_embeds.t()) / self.temperature
        
        # 타깃: 대각선 요소 (자신의 쌍이 양성)
        targets = torch.arange(batch_size, device=device)
        
        # 하드 네거티브 마이닝 - 텍스트→이미지 방향
        neg_sim_t2i = sim_matrix.clone()
        # 대각선(양성 쌍)을 큰 음수값으로 설정
        neg_sim_t2i[torch.arange(batch_size), torch.arange(batch_size)] = -10000.0
        
        # 각 앵커에 대해 가장 어려운 네거티브 샘플 선택
        num_hard_negatives = max(int(batch_size * self.hard_mining_factor), 1)
        hard_indices_t2i = torch.topk(neg_sim_t2i, k=num_hard_negatives, dim=1)[1]
        
        # 모든 샘플을 포함하는 새로운 로직으로 수정
        # 먼저 각 텍스트에 대한 전체 이미지 유사도 (양성 + 하드 네거티브 + 일반 네거티브)
        all_indices_t2i = torch.argsort(sim_matrix, dim=1, descending=True)
        
        # 양성 샘플(자기 자신)을 첫 번째로 이동
        for i in range(batch_size):
            pos_idx = torch.where(all_indices_t2i[i] == i)[0]
            if pos_idx.shape[0] > 0:  # 안전 검사
                # 양성 샘플과 0번째 위치 교환
                all_indices_t2i[i, 0], all_indices_t2i[i, pos_idx] = all_indices_t2i[i, pos_idx].clone(), all_indices_t2i[i, 0].clone()
        
        # 선택된 샘플들(양성 1개 + 하드 네거티브 num_hard_negatives개)
        selected_indices_t2i = all_indices_t2i[:, :num_hard_negatives+1]
        
        # 선택된 인덱스를 사용해 축소된 유사도 행렬 구성
        reduced_sim_t2i = torch.gather(sim_matrix, 1, selected_indices_t2i)
        
        # 타깃은 항상 0 (첫 번째 요소가 양성 쌍)
        t2i_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 손실 계산
        t2i_loss = F.cross_entropy(reduced_sim_t2i, t2i_targets)
        
        # 이미지→텍스트 방향도 동일한 방식으로 처리
        # 모든 샘플을 포함하는 새로운 로직으로 수정
        sim_matrix_t = sim_matrix.t()  # 전치 행렬
        all_indices_i2t = torch.argsort(sim_matrix_t, dim=1, descending=True)
        
        # 양성 샘플(자기 자신)을 첫 번째로 이동
        for i in range(batch_size):
            pos_idx = torch.where(all_indices_i2t[i] == i)[0]
            if pos_idx.shape[0] > 0:  # 안전 검사
                # 양성 샘플과 0번째 위치 교환
                all_indices_i2t[i, 0], all_indices_i2t[i, pos_idx] = all_indices_i2t[i, pos_idx].clone(), all_indices_i2t[i, 0].clone()
        
        # 선택된 샘플들(양성 1개 + 하드 네거티브 num_hard_negatives개)
        selected_indices_i2t = all_indices_i2t[:, :num_hard_negatives+1]
        
        # 선택된 인덱스를 사용해 축소된 유사도 행렬 구성
        reduced_sim_i2t = torch.gather(sim_matrix_t, 1, selected_indices_i2t)
        
        # 타깃은 항상 0 (첫 번째 요소가 양성 쌍)
        i2t_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 손실 계산
        i2t_loss = F.cross_entropy(reduced_sim_i2t, i2t_targets)
        
        # 최종 손실은 양방향 평균
        loss = (t2i_loss + i2t_loss) / 2
        
        return loss

    def compute_dual_softmax_loss(self, image_embeds, text_embeds):
        """
        CLIP과 유사한 듀얼 소프트맥스 손실
        """
        device = image_embeds.device
        batch_size = image_embeds.size(0)
        
        # 유사도 행렬 계산
        logits = torch.matmul(text_embeds, image_embeds.t()) / self.temperature
        
        # 라벨은 모든 대각선 요소
        labels = torch.arange(batch_size, device=device)
        
        # 양방향 손실 계산 및 평균
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        # 소프트맥스 스케일링 적용 (학습 안정화)
        sim_i2t = F.softmax(logits, dim=1)
        sim_t2i = F.softmax(logits, dim=0)
        
        # 양성 쌍 모으기 및 평균
        pos_i2t = torch.sum(-torch.log(sim_i2t[labels, labels] + 1e-8)) / batch_size
        pos_t2i = torch.sum(-torch.log(sim_t2i[labels, labels] + 1e-8)) / batch_size
        
        # 모든 손실 항 결합
        loss = (loss_i2t + loss_t2i + pos_i2t + pos_t2i) / 4
        
        return loss

    def configure_optimizers(self):
        # 옵티마이저 설정 - 다양한 학습률 적용
        param_groups = []
        
        # 프로젝션 레이어 (가장 높은 학습률)
        param_groups.append({"params": self.image_proj.parameters(), "lr": self.learning_rate})
        param_groups.append({"params": self.text_proj.parameters(), "lr": self.learning_rate})
        
        # 이미지 인코더 레이어별 다른 학습률 적용
        if self.should_train_image_encoder:
            image_layer_lrs = {
                "layers.3.blocks.1": self.learning_rate * 0.01,    # 가장 마지막 레이어
                "layers.3.blocks.0": self.learning_rate * 0.008,
                "layers.2.blocks.1": self.learning_rate * 0.006,
                "layers.2.blocks.0": self.learning_rate * 0.004,
                "layers.1.blocks.1": self.learning_rate * 0.002,
                "layers.1.blocks.0": self.learning_rate * 0.001,
            }
            
            # 각 레이어 그룹에 대해 파라미터 수집 및 학습률 설정
            for layer_name, lr in image_layer_lrs.items():
                layer_params = []
                for name, param in self.image_encoder.named_parameters():
                    if layer_name in name and param.requires_grad:
                        layer_params.append(param)
                
                if layer_params:
                    param_groups.append({
                        "params": layer_params,
                        "lr": lr
                    })
        
        # 텍스트 인코더 레이어별 다른 학습률 적용
        if self.should_train_text_encoder:
            text_layer_lrs = {
                "encoder.layer.23": self.learning_rate * 0.01,   # 가장 마지막 레이어
                "encoder.layer.22": self.learning_rate * 0.008,
                "encoder.layer.21": self.learning_rate * 0.006,
                "encoder.layer.20": self.learning_rate * 0.004,
                "encoder.layer.19": self.learning_rate * 0.002,
                "encoder.layer.18": self.learning_rate * 0.001,  # 6번째 마지막 레이어
            }
            
            # 각 레이어 그룹에 대해 파라미터 수집 및 학습률 설정
            for layer_name, lr in text_layer_lrs.items():
                layer_params = []
                for name, param in self.text_encoder.named_parameters():
                    if layer_name in name and param.requires_grad:
                        layer_params.append(param)
                
                if layer_params:
                    param_groups.append({
                        "params": layer_params,
                        "lr": lr
                    })
        
        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        
        return optimizer
    
    def training_step(self, batch, batch_idx):
        image_embeds, text_embeds = self(batch)
        
        # 손실 함수 선택
        if self.use_dual_softmax:
            loss = self.compute_dual_softmax_loss(image_embeds, text_embeds)
        else:
            loss = self.compute_contrastive_loss_with_hard_negatives(image_embeds, text_embeds)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image_embeds, text_embeds = self(batch)
        
        # 검증 단계에서는 듀얼 소프트맥스 손실 사용 (평가용)
        val_loss = self.compute_dual_softmax_loss(image_embeds, text_embeds)
        
        self.log("val_loss_step", val_loss, prog_bar=False)
        return {
            "val_loss": val_loss,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "img_ids": batch["img_id"]
        }

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._val_outputs.append(outputs)

    def on_validation_epoch_end(self):
        # 평균 검증 손실 기록
        val_losses = torch.stack([o["val_loss"] for o in self._val_outputs])
        avg_val_loss = val_losses.mean()
        self.log("val_loss", avg_val_loss, prog_bar=True)
        
        # 모든 임베딩 수집
        all_image_embeds = torch.cat([o["image_embeds"] for o in self._val_outputs], dim=0)
        all_text_embeds = torch.cat([o["text_embeds"] for o in self._val_outputs], dim=0)
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
        
        # Recall@K 계산
        recall_at_k = self.compute_recall(similarity_matrix, ks=[1, 5, 10])
        
        for k, v in recall_at_k.items():
            self.log(f"val_recall@{k}", v, prog_bar=True)
        
        self._val_outputs.clear()

    def test_step(self, batch, batch_idx):
        image_embeds, text_embeds = self(batch)
        return {"image_embeds": image_embeds, "text_embeds": text_embeds}

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_image_embeds.append(outputs["image_embeds"])
        self.test_text_embeds.append(outputs["text_embeds"])

    def on_test_epoch_end(self):
        # 모든 임베딩 수집
        all_image_embeds = torch.cat(self.test_image_embeds, dim=0)
        all_text_embeds = torch.cat(self.test_text_embeds, dim=0)
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
        
        # Recall@K 계산
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

# -----------------------------------------------------------------------------
# 6. 메인 실행 코드
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Flickr30k 데이터셋 로드
    print("Flickr30k 데이터셋 로드 중...")
    flickr_dataset = load_dataset("nlphuji/flickr30k")["test"]
    train_dataset = flickr_dataset.filter(lambda x: x["split"] == "train")
    valid_dataset = flickr_dataset.filter(lambda x: x["split"] == "val")
    test_dataset = flickr_dataset.filter(lambda x: x["split"] == "test")
    print(f"로드된 데이터 크기: train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")
    
    # 데이터 모듈 초기화
    data_module = Flickr30kDataModuleWithCaptionPooling(
        train_dataset_hf=train_dataset,
        valid_dataset_hf=valid_dataset,
        test_dataset_hf=test_dataset,
        batch_size=64,
        num_workers=4,
        image_size=224,
        max_length=77,
        max_captions=5  # 이미지당 최대 5개의 캡션만 사용
    )
    
    # 모델 초기화 (인코더 학습 여부 설정 가능)
    model = EnhancedImageTextModel(
        image_encoder_name="microsoft/swin-base-patch4-window7-224",
        text_encoder_name="roberta-large",
        embed_dim=256,
        temperature=0.07,
        learning_rate=1e-4,
        should_train_image_encoder=True,  # 이미지 인코더 학습 활성화
        should_train_text_encoder=True,   # 텍스트 인코더 학습 활성화
        hard_mining_factor=0.25,          # 하드 네거티브 비율
        use_dual_softmax=False             # 듀얼 소프트맥스 손실 사용
    )
    
    # 체크포인트 저장 경로
    checkpoint_dir = "./checkpoints/flickr30k_hard_mining"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 콜백 정의
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model-{epoch:02d}-{val_recall@1:.4f}",  # recall@1로 저장
        save_top_k=3,
        monitor="val_recall@1",  # 모니터링 지표: recall@1
        mode="max"  # 더 높은 recall이 좋음
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_recall@1",
        patience=5,
        mode="max"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # 로거 설정
    logger = TensorBoardLogger("./logs", name="flickr30k_hard_mining")
    
    # 훈련 시작
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # 모델 훈련
    trainer.fit(model, data_module)
    
    # 테스트 수행
    trainer.test(model, data_module)