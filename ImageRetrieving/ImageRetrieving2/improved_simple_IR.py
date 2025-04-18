import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import h5py
from PIL import Image

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
# 2. 로컬 Flickr30K 데이터셋 클래스 (전처리된 버전)
# -----------------------------------------------------------------------------
class PreprocessedFlickr30KDataset(Dataset):
    """미리 전처리된 Flickr30K 데이터셋을 로드하는 클래스"""
    def __init__(self, image_h5_path, caption_h5_path):
        self.image_h5_path = image_h5_path
        self.caption_h5_path = caption_h5_path
        
        # 이미지 데이터 로드
        self.image_h5 = h5py.File(image_h5_path, 'r')
        self.images = self.image_h5['images']
        self.image_ids = [id.decode('utf-8') for id in self.image_h5['image_ids']]
        
        # 캡션 데이터 로드
        self.caption_h5 = h5py.File(caption_h5_path, 'r')
        self.input_ids = self.caption_h5['input_ids']
        self.attention_masks = self.caption_h5['attention_masks']
        self.caption_image_ids = [id.decode('utf-8') for id in self.caption_h5['image_ids']]
        self.captions = [cap.decode('utf-8') for cap in self.caption_h5['captions']]
        
        # 이미지 ID -> 인덱스 매핑
        self.image_id_to_idx = {id: i for i, id in enumerate(self.image_ids)}
        
        # 매핑 데이터 로드
        self.img_to_cap_mapping = {}
        for img_id in self.image_id_to_idx.keys():
            if img_id in self.caption_h5['img_to_cap_mapping']:
                self.img_to_cap_mapping[img_id] = self.caption_h5['img_to_cap_mapping'][img_id][:]
        
        # 유니크한 이미지 ID 목록 (배치 샘플링용)
        self.unique_image_ids = list(self.image_id_to_idx.keys())
        
        # 인덱스 매핑 구성
        self.index_map = []
        for i, img_id in enumerate(self.caption_image_ids):
            self.index_map.append((img_id, i))
            
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        img_id, caption_idx = self.index_map[idx]
        
        # 이미지 로드
        img_idx = self.image_id_to_idx[img_id]
        image = torch.from_numpy(self.images[img_idx])
        
        # 텍스트 로드
        input_ids = torch.from_numpy(self.input_ids[caption_idx])
        attention_mask = torch.from_numpy(self.attention_masks[caption_idx])
        caption = self.captions[caption_idx]
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "img_id": img_id,
            "caption_idx": caption_idx,
            "caption": caption
        }
    
    def get_unique_image_sample(self, img_id):
        """유니크한 이미지 샘플링을 위한 함수"""
        # 이미지 로드
        img_idx = self.image_id_to_idx[img_id]
        image = torch.from_numpy(self.images[img_idx])
        
        # 해당 이미지의 캡션 중 하나를 무작위로 선택
        if img_id in self.img_to_cap_mapping and len(self.img_to_cap_mapping[img_id]) > 0:
            caption_indices = self.img_to_cap_mapping[img_id]
            caption_idx = random.choice(caption_indices)
            
            # 텍스트 로드
            input_ids = torch.from_numpy(self.input_ids[caption_idx])
            attention_mask = torch.from_numpy(self.attention_masks[caption_idx])
            caption = self.captions[caption_idx]
            
            return {
                "image": image,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "img_id": img_id,
                "caption_idx": caption_idx,
                "caption": caption
            }
        
        # 캡션 매핑이 없는 경우 (일반적으로 발생하지 않음)
        # 이 이미지 ID에 해당하는 첫 번째 캡션 찾기
        for i, (map_img_id, map_cap_idx) in enumerate(self.index_map):
            if map_img_id == img_id:
                caption_idx = map_cap_idx
                input_ids = torch.from_numpy(self.input_ids[caption_idx])
                attention_mask = torch.from_numpy(self.attention_masks[caption_idx])
                caption = self.captions[caption_idx]
                return {
                    "image": image,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "img_id": img_id,
                    "caption_idx": caption_idx,
                    "caption": caption
                }
    
    def close(self):
        """H5 파일 닫기"""
        if hasattr(self, 'image_h5') and self.image_h5 is not None:
            self.image_h5.close()
        if hasattr(self, 'caption_h5') and self.caption_h5 is not None:
            self.caption_h5.close()
    
    def __del__(self):
        self.close()

# -----------------------------------------------------------------------------
# 3. 데이터 전처리 함수
# -----------------------------------------------------------------------------
def load_preprocessed_flickr30k(base_dir, split='train'):
    """전처리된 Flickr30k 데이터 로드"""
    image_h5_path = os.path.join(base_dir, f'flickr30k_{split}_images.h5')
    caption_h5_path = os.path.join(base_dir, f'flickr30k_{split}_captions.h5')
    
    return image_h5_path, caption_h5_path

# -----------------------------------------------------------------------------
# 4. DataModule
# -----------------------------------------------------------------------------
class PreprocessedFlickr30KDataModule(pl.LightningDataModule):
    def __init__(self,
                 base_dir,
                 batch_size=32,
                 num_workers=4,
                 use_unique_images_in_batch=True):
        super().__init__()
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_unique_images_in_batch = use_unique_images_in_batch

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # 훈련 데이터셋 로드
            train_img_h5, train_cap_h5 = load_preprocessed_flickr30k(self.base_dir, 'train')
            self.train_dataset = PreprocessedFlickr30KDataset(
                image_h5_path=train_img_h5,
                caption_h5_path=train_cap_h5
            )
            
            # 검증 데이터셋 로드
            val_img_h5, val_cap_h5 = load_preprocessed_flickr30k(self.base_dir, 'val')
            self.valid_dataset = PreprocessedFlickr30KDataset(
                image_h5_path=val_img_h5,
                caption_h5_path=val_cap_h5
            )
            
            print(f"학습 데이터셋 크기: {len(self.train_dataset)}")
            print(f"검증 데이터셋 크기: {len(self.valid_dataset)}")
            print(f"유니크한 이미지 수 (학습): {len(self.train_dataset.unique_image_ids)}")
            print(f"유니크한 이미지 수 (검증): {len(self.valid_dataset.unique_image_ids)}")
            
        if stage == "test" or stage is None:
            # 테스트 데이터셋 로드
            test_img_h5, test_cap_h5 = load_preprocessed_flickr30k(self.base_dir, 'test')
            self.test_dataset = PreprocessedFlickr30KDataset(
                image_h5_path=test_img_h5,
                caption_h5_path=test_cap_h5
            )
            print(f"테스트 데이터셋 크기: {len(self.test_dataset)}")
            print(f"유니크한 이미지 수 (테스트): {len(self.test_dataset.unique_image_ids)}")

    def train_dataloader(self):
        if self.use_unique_images_in_batch:
            # 유니크한 이미지 ID를 사용하는 커스텀 데이터셋
            class UniqueImageDataset(Dataset):
                def __init__(self, base_dataset):
                    self.base_dataset = base_dataset
                    self.unique_image_ids = base_dataset.unique_image_ids

                def __len__(self):
                    return len(self.unique_image_ids)

                def __getitem__(self, idx):
                    # 이미지 ID 가져오기
                    img_id = self.unique_image_ids[idx]
                    return self.base_dataset.get_unique_image_sample(img_id)

            # 유니크한 이미지만 처리하는 데이터셋 생성
            unique_dataset = UniqueImageDataset(self.train_dataset)
            
            return DataLoader(
                unique_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True if self.num_workers > 0 else False
            )
        else:
            # 기존 방식 (이미지 중복 가능)
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
    
    def teardown(self, stage=None):
        # 데이터셋 H5 파일 닫기
        if stage == "fit" or stage is None:
            if hasattr(self, 'train_dataset'):
                self.train_dataset.close()
            if hasattr(self, 'valid_dataset'):
                self.valid_dataset.close()
        
        if stage == "test" or stage is None:
            if hasattr(self, 'test_dataset'):
                self.test_dataset.close()

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
        같은 이미지의 캡션은 모두 양성으로 취급
        
        Args:
            image_embeds: 이미지 임베딩 [batch_size, embed_dim]
            text_embeds: 텍스트 임베딩 [batch_size, embed_dim]
            img_ids: 이미지 ID 리스트 [batch_size]
        """
        device = image_embeds.device
        batch_size = image_embeds.size(0)
        
        # 이미지 ID 그룹화 (같은 ID를 가진 샘플들의 인덱스 그룹화)
        unique_ids = list(set(img_ids))
        id_to_indices = {id: [] for id in unique_ids}
        for i, id in enumerate(img_ids):
            id_to_indices[id].append(i)
        
        # 유사도 행렬 계산
        logits = torch.matmul(text_embeds, image_embeds.t()) / self.temperature
        
        # 양성 샘플 마스크 생성 (같은 이미지 ID를 갖는 샘플들은 모두 양성)
        pos_mask = torch.zeros_like(logits, dtype=torch.float)
        
        for idx, id in enumerate(img_ids):
            # 현재 캡션과 동일한 ID를 가진 모든 이미지를 양성으로 표시
            for pos_idx in id_to_indices[id]:
                pos_mask[idx, pos_idx] = 1.0
        
        # 하드 네거티브 마이닝 및 가중치 적용
        if self.use_hard_negatives:
            neg_mask = 1.0 - pos_mask
            
            # 마스킹된 유사도 행렬 (양성은 매우 낮은 값으로 설정)
            masked_logits = logits - pos_mask * 1e9
            
            # 각 캡션에 대해 가장 어려운 네거티브 찾기
            hardest_negatives_scores, _ = masked_logits.max(dim=1, keepdim=True)
            
            # 하드 네거티브에 가중치 적용
            hard_negative_mask = (masked_logits == hardest_negatives_scores)
            weighted_logits = logits + hard_negative_mask.float() * self.hard_negative_weight
        else:
            weighted_logits = logits
        
        # 텍스트→이미지 방향 손실 (다중 양성 고려)
        t2i_loss = 0.0
        for i in range(batch_size):
            # 현재 캡션에 대응하는 양성 이미지들 찾기
            pos_indices = torch.where(pos_mask[i] > 0)[0]
            
            # 양성 로짓값 평균 (마진 적용)
            pos_logits = weighted_logits[i, pos_indices] - self.margin
            pos_logits = torch.clamp(pos_logits, min=0.0).mean()
            
            # 모든 로짓값의 로그섬 계산
            all_logits = weighted_logits[i]
            exp_logits = torch.exp(all_logits)
            log_sum_exp = torch.log(exp_logits.sum())
            
            # 현재 캡션에 대한 손실 계산
            curr_loss = log_sum_exp - pos_logits
            t2i_loss += curr_loss
        
        t2i_loss = t2i_loss / batch_size
        
        # 이미지→텍스트 방향 손실
        i2t_loss = 0.0
        for i in range(batch_size):
            # 현재 이미지에 대응하는 양성 캡션들 찾기
            pos_indices = torch.where(pos_mask[:, i] > 0)[0]
            
            # 양성 로짓값 평균 (마진 적용)
            pos_logits = weighted_logits[pos_indices, i] - self.margin
            pos_logits = torch.clamp(pos_logits, min=0.0).mean()
            
            # 모든 로짓값의 로그섬 계산
            all_logits = weighted_logits[:, i]
            exp_logits = torch.exp(all_logits)
            log_sum_exp = torch.log(exp_logits.sum())
            
            # 현재 이미지에 대한 손실 계산
            curr_loss = log_sum_exp - pos_logits
            i2t_loss += curr_loss
        
        i2t_loss = i2t_loss / batch_size
        
        # 최종 손실 결합
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
        
        # 모든 임베딩 및 이미지 ID 수집
        all_image_embeds = torch.cat([o["image_embeds"] for o in self._val_outputs], dim=0)
        all_text_embeds = torch.cat([o["text_embeds"] for o in self._val_outputs], dim=0)
        all_img_ids = [img_id for output in self._val_outputs for img_id in output["img_ids"]]
        
        # 유니크한 이미지 ID 및 인덱스 매핑
        unique_img_ids = list(set(all_img_ids))
        img_id_to_indices = {img_id: [] for img_id in unique_img_ids}
        for i, img_id in enumerate(all_img_ids):
            img_id_to_indices[img_id].append(i)
        
        # 이미지 ID별 이미지 임베딩
        unique_image_embeds = []
        for img_id in unique_img_ids:
            indices = img_id_to_indices[img_id]
            # 각 이미지 ID에 해당하는 첫 번째 이미지 임베딩만 사용
            # (같은 이미지는 동일한 이미지 임베딩을 가짐)
            img_embed = all_image_embeds[indices[0]]
            unique_image_embeds.append(img_embed.unsqueeze(0))
        
        # 이미지 임베딩 텐서 생성
        unique_image_embeds = torch.cat(unique_image_embeds, dim=0)  # [num_unique_images, embed_dim]
        
        # 다중 캡션 평가를 위한 R@K 계산
        recall_at_k = self.compute_recall_multi_caption(
            all_text_embeds, unique_image_embeds, all_img_ids, unique_img_ids, ks=[1, 5, 10]
        )
        
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
        # 모든 임베딩 및 이미지 ID 수집
        all_image_embeds = torch.cat([o["image_embeds"] for o in self._val_outputs], dim=0)
        all_text_embeds = torch.cat([o["text_embeds"] for o in self._val_outputs], dim=0)
        all_img_ids = [img_id for output in self._val_outputs for img_id in output["img_ids"]]
        
        # 유니크한 이미지 ID 및 인덱스 매핑
        unique_img_ids = list(set(all_img_ids))
        img_id_to_indices = {img_id: [] for img_id in unique_img_ids}
        for i, img_id in enumerate(all_img_ids):
            img_id_to_indices[img_id].append(i)
        
        # 이미지 ID별 이미지 임베딩
        unique_image_embeds = []
        for img_id in unique_img_ids:
            indices = img_id_to_indices[img_id]
            # 각 이미지 ID에 해당하는 첫 번째 이미지 임베딩만 사용
            img_embed = all_image_embeds[indices[0]]
            unique_image_embeds.append(img_embed.unsqueeze(0))
        
        # 이미지 임베딩 텐서 생성
        unique_image_embeds = torch.cat(unique_image_embeds, dim=0)  # [num_unique_images, embed_dim]
        
        # 다중 캡션 평가를 위한 R@K 계산
        recall_at_k = self.compute_recall_multi_caption(
            all_text_embeds, unique_image_embeds, all_img_ids, unique_img_ids, ks=[1, 5, 10]
        )
        
        for k, v in recall_at_k.items():
            self.log(f"test_recall@{k}", v, prog_bar=True)
        
        print(f"[테스트 결과] 테스트 Recall: {recall_at_k}")
        self.test_image_embeds.clear()
        self.test_text_embeds.clear()
        self._val_outputs.clear()
    
    def compute_recall_multi_caption(self, text_embeds, image_embeds, text_img_ids, unique_img_ids, ks=[1, 5, 10]):
        """
        다중 캡션을 고려한 리콜 계산
        
        Arguments:
            text_embeds: 모든 텍스트 임베딩 [num_texts, embed_dim]
            image_embeds: 유니크한 이미지 임베딩 [num_unique_images, embed_dim]
            text_img_ids: 각 텍스트 임베딩의 이미지 ID 리스트 [num_texts]
            unique_img_ids: 유니크한 이미지 ID 리스트 [num_unique_images]
            ks: 계산할 리콜@K 값 리스트
            
        Returns:
            recall_scores: 리콜@K 딕셔너리 {k: score}
        """
        device = text_embeds.device
        
        # 텍스트-이미지 유사도 행렬 계산
        similarity_matrix = torch.matmul(text_embeds, image_embeds.t())  # [num_texts, num_unique_images]
        
        # 텍스트 임베딩의 그라운드 트루스 인덱스 생성
        # 각 텍스트(캡션)에 대해, 해당 텍스트가 속한 이미지의 인덱스가 그라운드 트루스
        img_id_to_idx = {img_id: i for i, img_id in enumerate(unique_img_ids)}
        ground_truth = torch.tensor([img_id_to_idx[img_id] for img_id in text_img_ids], device=device)
        
        # 각 텍스트 쿼리에 대해 정렬된 인덱스 계산
        sorted_indices = similarity_matrix.argsort(dim=1, descending=True)
        
        # 각 k에 대한 리콜 계산
        recall_scores = {}
        for k in ks:
            top_k = sorted_indices[:, :k]
            match = (top_k == ground_truth.unsqueeze(1)).any(dim=1)
            recall_scores[k] = match.float().mean().item()
        
        return recall_scores
        
    def compute_recall(self, similarity_matrix, ks=[1, 5, 10]):
        # 각 쿼리(행)에 대해 ground-truth 인덱스가 top-k 검색 인덱스에 있는지 확인
        # 주의: 이 함수는 각 행과 각 열이 1:1 대응되는 경우를 가정합니다.
        # 다중 캡션을 처리하려면 compute_recall_multi_caption 함수를 사용하세요.
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
        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        # )
        
        # return [optimizer], [scheduler]
        return optimizer

# -----------------------------------------------------------------------------
# 6. 메인 실행 코드
# -----------------------------------------------------------------------------
# 전처리된 데이터셋 경로 설정
PREPROCESSED_DIR = "./flickr30k_preprocessed"  # 전처리된 데이터가 저장된 경로

# 로컬 데이터셋 로드
data_module = PreprocessedFlickr30KDataModule(
    base_dir=PREPROCESSED_DIR,
    batch_size=32,
    num_workers=4,
    use_unique_images_in_batch=True  # 배치에 유니크한 이미지만 포함
)
data_module.setup("fit")

# -----------------------------------------------------------------------------
# 7. Trainer, Logger, and Callback Setup
# -----------------------------------------------------------------------------
logger = TensorBoardLogger(
    save_dir="ImageRetrieving_multi",
    name="Flickr30k_CLIP_multi"
)

early_stopping_callback = EarlyStopping(
    monitor="val_recall@1",
    patience=5,
    mode="max",
    min_delta=0.001
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_recall@1",
    mode="max",
    dirpath="checkpoints/multi",
    filename="flickr30k-clip-multi-{epoch:02d}-{val_recall@1:.4f}",
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
# 8. Model Initialization and Training
# -----------------------------------------------------------------------------
model = ImprovedImageTextModel(
    image_encoder_name="microsoft/swin-base-patch4-window7-224",
    text_encoder_name="roberta-large",
    embed_dim=256,
    temperature=0.07,
    learning_rate=5e-5,
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