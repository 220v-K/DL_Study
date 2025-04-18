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
# 2. COCO 데이터셋 및 이미지별 캡션 그룹화
# -----------------------------------------------------------------------------
class COCODatasetWithCaptionPooling(Dataset):
    """
    MS-COCO 데이터셋을 로드하고 이미지별 여러 캡션을 그룹화하는 클래스
    각 이미지마다 여러 캡션이 있으며, 이들을 평균 풀링하여 사용
    """
    def __init__(self, split='train', root_dir='./ImageRetrieving_coco', image_size=224, max_length=77, max_captions=5):
        self.split = split
        self.root_dir = root_dir
        self.max_captions = max_captions  # 이미지당 최대 캡션 수 (기본값: 5)
        
        # 데이터 경로 설정
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'images/train2014')
            self.ann_file = os.path.join(root_dir, 'annotations/captions_train2014.json')
        elif split in ['val', 'validation', 'test']:  # validation과 test는 같은 데이터셋 사용
            self.img_dir = os.path.join(root_dir, 'images/val2017')
            self.ann_file = os.path.join(root_dir, 'annotations/captions_val2017.json')
        else:
            raise ValueError(f"지원하지 않는 split: {split}")
        
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
        
        # 어노테이션 데이터 로드
        self.raw_annotations = self._load_annotations()
        
        # 이미지별 캡션 그룹화
        self.image_captions = self._group_captions_by_image()
        
        # 데이터셋 인덱스 구성
        self.image_ids = list(self.image_captions.keys())
        
        print(f"총 {len(self.image_ids)}개의 유니크한 이미지와 평균 {len(self.raw_annotations) / len(self.image_ids):.1f}개의 캡션/이미지가 있습니다.")
        
    def _load_annotations(self):
        """어노테이션 파일 로드"""
        print(f"어노테이션 파일 로드 중: {self.ann_file}")
        with open(self.ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 이미지 정보 추출 (id -> 파일 이름 매핑)
        img_id_to_filename = {}
        for img in data['images']:
            img_id_to_filename[img['id']] = img['file_name']
        
        # 어노테이션 처리
        annotations = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption']
            file_name = img_id_to_filename.get(img_id)
            
            if file_name:
                annotations.append({
                    'image_id': img_id,
                    'caption': caption,
                    'file_name': file_name
                })
        
        print(f"{len(annotations)}개의 캡션을 로드했습니다.")
        return annotations
    
    def _group_captions_by_image(self):
        """이미지 ID별로 캡션 그룹화"""
        image_captions = defaultdict(list)
        
        # 모든 주석을 순회하며 이미지별로 캡션 그룹화
        for ann in self.raw_annotations:
            img_id = str(ann['image_id'])
            
            # 이미지별 최대 캡션 수 제한 (이미 최대치에 도달한 이미지는 건너뜀)
            if len(image_captions[img_id]) < self.max_captions:
                image_captions[img_id].append({
                    'caption': ann['caption'],
                    'file_name': ann['file_name']
                })
        
        # 캡션 수가 다른 이미지들에 대한 통계
        caption_counts = {}
        for img_id, captions in image_captions.items():
            count = len(captions)
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
        captions_data = self.image_captions[img_id]
        
        # 모든 캡션 데이터에서 동일한 파일 이름 사용
        file_name = captions_data[0]['file_name']
        
        # 이미지 로드 및 전처리
        image_path = os.path.join(self.img_dir, file_name)
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"이미지 로드 오류: {image_path} - {e}")
            # 오류 발생시 검은색 이미지로 대체
            image = torch.zeros(3, 224, 224)
        
        # 모든 캡션 토큰화 (최대 max_captions 개수만 사용)
        all_captions = []
        all_input_ids = []
        all_attention_masks = []
        
        # 최대 캡션 수 제한 (이미 _group_captions_by_image에서 처리했지만 안전을 위해 다시 확인)
        num_captions = min(len(captions_data), self.max_captions)
        
        for i in range(num_captions):
            caption_data = captions_data[i]
            caption = caption_data['caption']
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
class COCODataModuleWithCaptionPooling(pl.LightningDataModule):
    def __init__(self,
                batch_size=128,  # 배치 사이즈 증가
                num_workers=4,
                image_size=224,
                max_length=77,
                max_captions=5,
                root_dir='./ImageRetrieving_coco'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_length = max_length
        self.max_captions = max_captions
        self.root_dir = root_dir

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # 훈련 데이터셋 로드
            self.train_dataset = COCODatasetWithCaptionPooling(
                split='train',
                root_dir=self.root_dir,
                image_size=self.image_size,
                max_length=self.max_length,
                max_captions=self.max_captions
            )
            
            # 검증 데이터셋 로드
            self.valid_dataset = COCODatasetWithCaptionPooling(
                split='val',
                root_dir=self.root_dir,
                image_size=self.image_size,
                max_length=self.max_length,
                max_captions=self.max_captions
            )
            
            print(f"학습 데이터셋 크기: {len(self.train_dataset)}")
            print(f"검증 데이터셋 크기: {len(self.valid_dataset)}")
            
        if stage == "test" or stage is None:
            # 테스트 데이터셋 로드 (검증 데이터셋과 동일)
            self.test_dataset = COCODatasetWithCaptionPooling(
                split='test',
                root_dir=self.root_dir,
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
# 4. 어텐션 풀링 모듈 추가
# -----------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """
    어텐션 메커니즘을 사용한 캡션 풀링 모듈
    각 캡션의 중요도를 학습하여 가중 평균을 계산
    """
    def __init__(self, embed_dim):
        super().__init__()
        # 어텐션 스코어 계산을 위한 레이어
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, embeddings, mask=None):
        """
        어텐션 기반 풀링 적용
        embeddings: [batch_size, num_captions, embed_dim] - 캡션 임베딩
        mask: [batch_size, num_captions] - 유효한 캡션을 나타내는 마스크 (선택 사항)
        """
        # 어텐션 점수 계산
        attention_scores = self.attention(embeddings).squeeze(-1)  # [batch_size, num_captions]
        
        # 마스크 적용 (패딩된 캡션이나 유효하지 않은 캡션은 무시)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e4)  # 마스킹된 위치에 매우 작은 값 할당
        
        # 소프트맥스로 어텐션 가중치 계산
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_captions]
        
        # 어텐션 가중치를 적용하여 가중 평균 계산
        pooled_embeddings = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, num_captions]
            embeddings  # [batch_size, num_captions, embed_dim]
        ).squeeze(1)  # [batch_size, embed_dim]
        
        return pooled_embeddings, attention_weights

# -----------------------------------------------------------------------------
# 5. 크로스 모달 어텐션 모듈
# -----------------------------------------------------------------------------
class CrossModalAttention(nn.Module):
    """
    이미지와 텍스트 임베딩 간의 크로스 모달 어텐션
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x, y):
        """
        x가 y에 어텐션하는 크로스 모달 어텐션 수행
        x: [batch_size, embed_dim] - 소스 임베딩
        y: [batch_size, embed_dim] - 타겟 임베딩
        """
        # 차원 확장 (Transformer 요구사항)
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, embed_dim]
        y_expanded = y.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # 크로스 어텐션 적용 (x가 y를 참조)
        attn_output, _ = self.attention(x_expanded, y_expanded, y_expanded)
        attn_output = attn_output.squeeze(1)  # [batch_size, embed_dim]
        
        # 잔차 연결 및 레이어 정규화
        x_out = self.norm1(x + attn_output)
        
        # 피드포워드 네트워크
        ffn_output = self.ffn(x_out)
        
        # 잔차 연결 및 레이어 정규화
        output = self.norm2(x_out + ffn_output)
        
        return output

# -----------------------------------------------------------------------------
# 6. 어텐션 기반 캡션 풀링을 사용한 이미지-텍스트 모델
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
                 use_dual_softmax=True,    # 듀얼 소프트맥스 손실 사용 여부
                 use_cross_modal_attention=True,  # 크로스 모달 어텐션 사용 여부
                 checkpoint_path=None):  # 체크포인트 경로 추가
        super().__init__()
        self.save_hyperparameters()

        # 하이퍼파라미터 저장
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.should_train_image_encoder = should_train_image_encoder
        self.should_train_text_encoder = should_train_text_encoder
        self.hard_mining_factor = hard_mining_factor
        self.use_dual_softmax = use_dual_softmax
        self.use_cross_modal_attention = use_cross_modal_attention
        self.checkpoint_path = checkpoint_path

        # 인코더 초기화
        self.image_encoder = SwinModel.from_pretrained(
            image_encoder_name,
            cache_dir="./model_cache"
        )
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_name)
        
        # 처음에는 모든 인코더 파라미터 동결
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # should_train_image_encoder가 True이고 체크포인트가 없는 경우에만 이미지 인코더의 특정 레이어 학습 활성화
        if should_train_image_encoder and checkpoint_path is None:
            # Swin 트랜스포머의 마지막 6개 레이어 학습 활성화
            image_layers_to_train = [
                "layers.3.blocks.1", "layers.3.blocks.0", 
                "layers.2.blocks.1", "layers.2.blocks.0",
                "layers.1.blocks.1", "layers.1.blocks.0"
            ]
            
            for name, param in self.image_encoder.named_parameters():
                if any(layer in name for layer in image_layers_to_train):
                    param.requires_grad = True
        
        # should_train_text_encoder가 True이고 체크포인트가 없는 경우에만 텍스트 인코더의 특정 레이어 학습 활성화
        if should_train_text_encoder and checkpoint_path is None:
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
        
        # 어텐션 기반 캡션 풀링 추가
        self.attention_pooling = AttentionPooling(embed_dim)
        
        # 크로스 모달 어텐션 추가
        if use_cross_modal_attention:
            self.img2txt_attention = CrossModalAttention(embed_dim, num_heads=8)
            self.txt2img_attention = CrossModalAttention(embed_dim, num_heads=8)

        # 검증/테스트 임베딩 로깅
        self._val_outputs = []
        self.test_image_embeds = []
        self.test_text_embeds = []
        
        # 체크포인트에서 모델 파라미터 로드 및 동결
        if checkpoint_path is not None:
            self._load_from_checkpoint_and_freeze(checkpoint_path)
        
        # 학습 대상 파라미터 수 출력
        self._log_trainable_params()
    
    def _load_from_checkpoint_and_freeze(self, checkpoint_path):
        """체크포인트에서 모델 파라미터 로드 및 어텐션 모듈 외 파라미터 동결"""
        print(f"체크포인트에서 파라미터 로드 중: {checkpoint_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 모델 상태 사전 추출
        state_dict = checkpoint["state_dict"]
        
        # 현재 모델에 맞는 키만 필터링
        model_keys = self.state_dict().keys()
        
        # 어텐션 풀링 및 크로스 모달 어텐션 관련 키 제외한 파라미터만 로드
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # attention_pooling, img2txt_attention, txt2img_attention 관련 키는 제외
            if "attention_pooling" not in key and "img2txt_attention" not in key and "txt2img_attention" not in key:
                if key in model_keys:
                    filtered_state_dict[key] = value
        
        # 필터링된 상태 사전 로드
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        print(f"로드된 파라미터: {len(filtered_state_dict)} 개")
        print(f"누락된 키: {missing_keys}")
        print(f"예상치 못한 키: {unexpected_keys}")
        
        # 어텐션 관련 모듈 외 모든 파라미터 동결
        for name, param in self.named_parameters():
            if "attention_pooling" not in name and "img2txt_attention" not in name and "txt2img_attention" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
    def _log_trainable_params(self):
        """학습 가능한 파라미터 수 로깅"""
        image_encoder_params = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)
        text_encoder_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        image_proj_params = sum(p.numel() for p in self.image_proj.parameters() if p.requires_grad)
        text_proj_params = sum(p.numel() for p in self.text_proj.parameters() if p.requires_grad)
        attention_pooling_params = sum(p.numel() for p in self.attention_pooling.parameters() if p.requires_grad)
        
        cross_modal_params = 0
        if self.use_cross_modal_attention:
            cross_modal_params = sum(p.numel() for p in self.img2txt_attention.parameters() if p.requires_grad) + \
                                sum(p.numel() for p in self.txt2img_attention.parameters() if p.requires_grad)
        
        total_params = image_encoder_params + text_encoder_params + image_proj_params + \
                      text_proj_params + attention_pooling_params + cross_modal_params
        
        total_all_params = sum(p.numel() for p in self.parameters())
        
        print(f"학습 가능한 파라미터: ")
        print(f"  이미지 인코더: {image_encoder_params:,} ({image_encoder_params / total_params:.1%})")
        print(f"  텍스트 인코더: {text_encoder_params:,} ({text_encoder_params / total_params:.1%})")
        print(f"  이미지 프로젝션: {image_proj_params:,} ({image_proj_params / total_params:.1%})")
        print(f"  텍스트 프로젝션: {text_proj_params:,} ({text_proj_params / total_params:.1%})")
        print(f"  어텐션 풀링: {attention_pooling_params:,} ({attention_pooling_params / total_params:.1%})")
        if self.use_cross_modal_attention:
            print(f"  크로스 모달 어텐션: {cross_modal_params:,} ({cross_modal_params / total_params:.1%})")
        print(f"  학습 가능한 총 파라미터: {total_params:,} (전체 파라미터 대비 {total_params / total_all_params:.1%})")
        print(f"  전체 파라미터: {total_all_params:,}")

    def configure_optimizers(self):
        # 옵티마이저 설정 - 학습 가능한 파라미터만 포함
        # 체크포인트를 로드한 경우 어텐션 관련 모듈만 학습
        param_groups = []
        
        # 어텐션 풀링 파라미터
        param_groups.append({"params": self.attention_pooling.parameters(), "lr": self.learning_rate})
        
        # 크로스 모달 어텐션 파라미터 (사용하는 경우)
        if self.use_cross_modal_attention:
            param_groups.append({"params": self.img2txt_attention.parameters(), "lr": self.learning_rate})
            param_groups.append({"params": self.txt2img_attention.parameters(), "lr": self.learning_rate})
        
        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        
        # 학습률 스케줄러 (옵션)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=self.learning_rate * 0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

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
        
        # 원래 배치 구조로 복원
        text_embeds = text_embeds.view(batch_size, num_captions, -1)  # [batch_size, num_captions, embed_dim]
        
        # 캡션 마스크 생성 (실제 캡션수에 따라)
        # attention_mask의 합계를 사용하여 유효한 캡션 식별
        caption_mask = (attention_mask_batch.sum(dim=2) > 0).float()
        
        # 어텐션 기반 풀링 적용 (평균 풀링 대신)
        pooled_text_embeds, attention_weights = self.attention_pooling(text_embeds, caption_mask)
        
        # 정규화
        pooled_text_embeds = F.normalize(pooled_text_embeds, p=2, dim=-1)
        
        return pooled_text_embeds
        
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
        
        # 텍스트 인코딩 (여러 캡션 어텐션 풀링)
        text_embeds = self.encode_text_batch(input_ids, attention_mask)  # [batch_size, embed_dim]
        
        # 크로스 모달 어텐션 적용 (학습 시 및 옵션 활성화된 경우)
        if self.training and self.use_cross_modal_attention:
            image_embeds_enhanced = self.img2txt_attention(image_embeds, text_embeds)
            text_embeds_enhanced = self.txt2img_attention(text_embeds, image_embeds)
            
            # 다시 정규화
            image_embeds = F.normalize(image_embeds_enhanced, p=2, dim=-1)
            text_embeds = F.normalize(text_embeds_enhanced, p=2, dim=-1)
        
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
        
        # 하드 네거티브를 위한 마스크 생성
        hard_mask_t2i = torch.zeros_like(sim_matrix, dtype=torch.bool)
        for i in range(batch_size):
            hard_mask_t2i[i, hard_indices_t2i[i]] = True
        
        # 대각선(양성 쌍)을 마스크에 추가
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        mask_t2i = diag_mask | hard_mask_t2i
        
        # 양성 쌍과 하드 네거티브만 포함하는 축소된 유사도 행렬
        reduced_sim_t2i = sim_matrix[mask_t2i].reshape(batch_size, -1)
        
        # 축소된 유사도 행렬의 타깃 생성 (첫 번째 열이 양성 쌍)
        reduced_targets_t2i = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 손실 계산
        t2i_loss = F.cross_entropy(reduced_sim_t2i, reduced_targets_t2i)
        
        # 이미지→텍스트 방향 반복
        neg_sim_i2t = sim_matrix.clone().t()
        neg_sim_i2t[torch.arange(batch_size), torch.arange(batch_size)] = -10000.0
        
        hard_indices_i2t = torch.topk(neg_sim_i2t, k=num_hard_negatives, dim=1)[1]
        
        hard_mask_i2t = torch.zeros_like(sim_matrix.t(), dtype=torch.bool)
        for i in range(batch_size):
            hard_mask_i2t[i, hard_indices_i2t[i]] = True
        
        mask_i2t = diag_mask | hard_mask_i2t
        reduced_sim_i2t = sim_matrix.t()[mask_i2t].reshape(batch_size, -1)
        reduced_targets_i2t = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        i2t_loss = F.cross_entropy(reduced_sim_i2t, reduced_targets_i2t)
        
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
    # 데이터 모듈 초기화
    data_module = COCODataModuleWithCaptionPooling(
        batch_size=64,  # 배치 사이즈 설정
        num_workers=4,
        image_size=224,
        max_length=77,
        max_captions=5,  # 이미지당 최대 5개의 캡션만 사용
        root_dir='/home/gpu_04/jw2020/ImageRetrieving_coco'
    )
    
    # 체크포인트 경로 설정
    checkpoint_path = "/home/gpu_04/jw2020/ImageRetrieving_coco/checkpoints/enhanced_coco_retrieval/model-epoch=19-val_recall@1=0.5860.ckpt"
    
    # 모델 초기화 (체크포인트에서 파라미터 로드하고 어텐션 모듈만 학습)
    model = EnhancedImageTextModel(
        image_encoder_name="microsoft/swin-base-patch4-window7-224",
        text_encoder_name="roberta-large",
        embed_dim=256,
        temperature=0.07,
        learning_rate=2e-4,  # 어텐션 모듈 학습을 위한 학습률
        should_train_image_encoder=True,  # 체크포인트 로드 시 무시됨
        should_train_text_encoder=True,   # 체크포인트 로드 시 무시됨
        hard_mining_factor=0.25,          # 하드 네거티브 비율
        use_dual_softmax=True,            # 듀얼 소프트맥스 손실 사용
        use_cross_modal_attention=False,   # 크로스 모달 어텐션 비활성화 (먼저 어텐션 풀링만 학습)
        checkpoint_path=checkpoint_path    # 체크포인트 경로 전달
    )
    
    # 체크포인트 저장 경로
    output_checkpoint_dir = "./checkpoints/attention_pooling_coco"
    os.makedirs(output_checkpoint_dir, exist_ok=True)
    
    # 콜백 정의
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_checkpoint_dir,
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
    logger = TensorBoardLogger("./logs", name="attention_pooling_coco")
    
    # 훈련 시작
    trainer = pl.Trainer(
        max_epochs=50,
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