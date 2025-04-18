import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import ViTModel, SwinModel, RobertaModel, RobertaTokenizer
from datasets import load_dataset
import numpy as np
import math
import os

# 재현성을 위한 랜덤 시드 설정
pl.seed_everything(42)

# GPU 성능 최적화를 위한 설정
torch.set_float32_matmul_precision('high')


# =============================================================================
# 1. Flickr30K 다중 캡션 데이터셋
# =============================================================================
class Flickr30KMultiCaptionDataset(Dataset):
    """
    Flickr30K 데이터셋에서 각 이미지의 5개 캡션을 모두 사용하는 데이터셋 클래스.
    각 이미지-캡션 쌍은 img_id를 통해 원본 이미지를 추적합니다.
    """
    def __init__(self, hf_dataset, tokenizer, image_transform, max_length=64):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        
        # 데이터셋 인덱스에서 (record_idx, caption_idx)로의 매핑 생성
        # 이미지당 5개의 캡션이 있으므로 필요함
        self.index_map = []
        for rec_idx, record in enumerate(self.hf_dataset):
            captions = record["caption"]
            for cap_idx in range(len(captions)):
                self.index_map.append((rec_idx, cap_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 레코드와 캡션 인덱스 가져오기
        record_idx, caption_idx = self.index_map[idx]
        record = self.hf_dataset[record_idx]
        
        # 이미지와 캡션 추출
        pil_image = record["image"]
        caption = record["caption"][caption_idx]

        # 이미지 전처리: (3, 224, 224) 크기의 텐서로 변환
        image = self.image_transform(pil_image)

        # 텍스트 토큰화 (패딩 및 자르기 포함)
        tokenized = self.tokenizer(
            caption,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "img_id": record_idx,  # 같은 이미지에 대한 다른 캡션을 추적하기 위한 이미지 ID
            "caption": caption     # 디버깅을 위한 원본 캡션
        }


# =============================================================================
# 2. Data Module
# =============================================================================
class Flickr30KDataModule(pl.LightningDataModule):
    """
    Flickr30K 데이터셋을 위한 PyTorch Lightning DataModule
    """
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

        # 이미지 전처리 파이프라인 (CLIP 논문에서 사용한 정규화 값)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        # RoBERTa 토크나이저 초기화
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    def setup(self, stage=None):
        """각 단계에 대한 데이터셋 준비 (fit, test)"""
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
        """학습 데이터로더 반환"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """검증 데이터로더 반환"""
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        """테스트 데이터로더 반환"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# =============================================================================
# 3. LLIP 모델
# =============================================================================
class LLIPModel(pl.LightningModule):
    """
    LLIP(Latent Language Image Pretraining) 모델 구현.
    다중 캡션을 활용하고 SigLIP 손실 함수 사용.
    """
    def __init__(self,
                 image_encoder_name="microsoft/swin-base-patch4-window7-224",
                 text_encoder_name="roberta-large",
                 embed_dim=256,
                 a_init=10.0,            # SigLIP 손실 함수의 스케일 초기값
                 b_init=-10.0,           # SigLIP 손실 함수의 바이어스 초기값
                 learning_rate=2e-5,
                 num_mixture_tokens=64,
                 attention_heads=8,
                 attention_temperature=5.0,
                 freeze_image_encoder=True,
                 freeze_text_encoder=True):
        """
        Args:
            image_encoder_name: 사전학습된 비전 모델 이름
            text_encoder_name: 사전학습된 텍스트 모델 이름
            embed_dim: 공동 임베딩 공간의 차원
            a_init: SigLIP 손실 함수의 스케일 초기값
            b_init: SigLIP 손실 함수의 바이어스 초기값
            learning_rate: 학습률
            num_mixture_tokens: 혼합 토큰의 수 (K)
            attention_heads: Cross-Attention의 헤드 수
            attention_temperature: Cross-Attention softmax의 온도
            freeze_image_encoder: 이미지 인코더 동결 여부
            freeze_text_encoder: 텍스트 인코더 동결 여부
        """
        super().__init__()
        self.save_hyperparameters()
        self._step_counter = 0  # global_step 대신 자체 카운터 사용

        # 1) 이미지 인코더 초기화 (Swin Transformer 또는 ViT)
        if "swin" in image_encoder_name:
            self.image_encoder = SwinModel.from_pretrained(image_encoder_name)
            self.is_swin = True
        else:
            self.image_encoder = ViTModel.from_pretrained(image_encoder_name)
            self.is_swin = False
        
        # 2) 텍스트 인코더 초기화 (RoBERTa)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_name)
        
        # 인코더 동결 설정
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
                
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # 히든 차원 크기 설정
        image_hidden_size = self.image_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        
        # 3) 학습 가능한 혼합 토큰 초기화 (Xavier 초기화 적용)
        self.num_mixture_tokens = num_mixture_tokens
        self.mixture_tokens = nn.Parameter(
            torch.zeros(self.num_mixture_tokens, image_hidden_size)
        )
        nn.init.xavier_uniform_(self.mixture_tokens)
        
        # 4) 다중 헤드 Cross-Attention 모듈
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=image_hidden_size,
            num_heads=attention_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 5) 프로젝션 레이어 (공동 임베딩 공간으로 매핑)
        self.image_proj = nn.Sequential(
            nn.LayerNorm(image_hidden_size),
            nn.Linear(image_hidden_size, embed_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_hidden_size),
            nn.Linear(text_hidden_size, embed_dim)
        )
        
        # 6) SigLIP 손실 함수 파라미터: 스케일 a와 바이어스 b
        self.a = nn.Parameter(torch.tensor(a_init))
        self.b = nn.Parameter(torch.tensor(b_init))
        
        # 기타 파라미터
        self.attention_temperature = attention_temperature
        self.learning_rate = learning_rate

        # 검증/테스트 출력 저장용
        self._val_outputs = []
        self.test_image_embeds = []
        self.test_text_embeds = []
        self.test_img_ids = []

    def encode_image(self, images):
        """
        이미지 인코딩 함수
        
        Args:
            images: 이미지 배치 (B, 3, 224, 224)
            
        Returns:
            image_feats: 이미지 특성 (B, hidden_size)
        """
        # 이미지 처리
        image_outputs = self.image_encoder(pixel_values=images)
        
        # 모델 유형에 따라 적절한 특성 사용
        if self.is_swin:
            # Swin의 경우 평균 풀링 사용
            image_feats = image_outputs.last_hidden_state.mean(dim=1)  # (B, hidden_size)
        else:
            # ViT의 경우 [CLS] 토큰 사용
            image_feats = image_outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
            
        return image_feats
    
    def encode_text(self, input_ids, attention_mask):
        """
        텍스트 인코딩 함수
        
        Args:
            input_ids: 토큰 ID (B, L)
            attention_mask: 어텐션 마스크 (B, L)
            
        Returns:
            text_feats: 텍스트 특성 (B, hidden_size)
        """
        # 텍스트 처리
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # [CLS] 토큰 특성 사용
        text_feats = text_outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
        
        return text_feats

    def forward(self, images, input_ids, attention_mask):
        """
        모델의 순방향 패스
        
        Args:
            images: 이미지 배치 (B, 3, 224, 224)
            input_ids: 토큰 ID (B, L)
            attention_mask: 어텐션 마스크 (B, L)
            
        Returns:
            image_embeds: 정규화된 이미지 임베딩 (B, embed_dim)
            text_embeds: 정규화된 텍스트 임베딩 (B, embed_dim)
        """
        batch_size = images.size(0)
        
        # 이미지 인코딩
        image_feats = self.encode_image(images)
        
        # 텍스트 인코딩
        text_feats = self.encode_text(input_ids, attention_mask)
        
        # 배치 크기에 맞게 혼합 토큰 확장
        # (num_mixture_tokens, hidden_size) -> (B, num_mixture_tokens, hidden_size)
        mixture_tokens_batch = self.mixture_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 텍스트 특성을 쿼리로 사용하여 Cross-Attention 적용
        # text_feats: (B, hidden_size) -> (B, 1, hidden_size)
        text_query = text_feats.unsqueeze(1)
        
        # Cross-Attention: 텍스트 특성이 혼합 토큰에 주의를 기울임
        attn_output, attn_weights = self.cross_attention(
            query=text_query,                # (B, 1, hidden_size)
            key=mixture_tokens_batch,        # (B, num_mixture_tokens, hidden_size)
            value=mixture_tokens_batch,      # (B, num_mixture_tokens, hidden_size)
            need_weights=True                # 어텐션 가중치 반환
        )
        
        # 시퀀스 차원 제거: (B, 1, hidden_size) -> (B, hidden_size)
        contextual_image_feats = attn_output.squeeze(1)
        
        # 공동 임베딩 공간으로 프로젝션
        # 컨텍스트화된 이미지 특성을 임베딩 공간으로 프로젝션
        image_embeds = self.image_proj(contextual_image_feats)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        
        # 텍스트 특성을 임베딩 공간으로 프로젝션
        text_embeds = self.text_proj(text_feats)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        return image_embeds, text_embeds

    def compute_multi_positive_siglip_loss(self, image_embeds, text_embeds, img_ids):
        """
        다중 양성 SigLIP 대조 손실 계산 - 수치적 안정성 개선
        
        Args:
            image_embeds: 정규화된 이미지 임베딩 (B, embed_dim)
            text_embeds: 정규화된 텍스트 임베딩 (B, embed_dim)
            img_ids: 각 샘플의 이미지 ID (B)
            
        Returns:
            loss: SigLIP 대조 손실 값
        """
        # 모든 이미지-텍스트 쌍 간의 유사도 행렬 계산
        sim_matrix = torch.matmul(image_embeds, text_embeds.t())
        batch_size = sim_matrix.size(0)
        
        # 양성 마스크 생성 (동일한 이미지 ID를 가진 모든 쌍이 양성)
        img_ids_expanded_row = img_ids.unsqueeze(1).expand(-1, batch_size)
        img_ids_expanded_col = img_ids.unsqueeze(0).expand(batch_size, -1)
        positive_mask = (img_ids_expanded_row == img_ids_expanded_col).float()
        
        # 양성 샘플이 없는 행 또는 열이 없도록 대각선 요소를 항상 양성으로 설정
        positive_mask = positive_mask.fill_diagonal_(1.0)
        
        # 음성 마스크 (양성이 아닌 모든 것)
        negative_mask = 1.0 - positive_mask
        
        # 수치적 안정성을 위해 F.softplus 사용
        # 양성 손실
        pos_sim = sim_matrix * positive_mask
        pos_counts = positive_mask.sum(dim=1).clamp(min=1)
        pos_term = -self.a * pos_sim + self.b
        # 오버플로우 방지를 위한 클리핑
        pos_term = torch.clamp(pos_term, -50.0, 50.0)
        pos_loss = F.softplus(pos_term) * positive_mask
        pos_loss = (pos_loss.sum(dim=1) / pos_counts).mean()
        
        # 음성 손실
        neg_sim = sim_matrix * negative_mask
        neg_counts = negative_mask.sum(dim=1).clamp(min=1)
        neg_term = self.a * neg_sim - self.b
        # 오버플로우 방지를 위한 클리핑
        neg_term = torch.clamp(neg_term, -50.0, 50.0)
        neg_loss = F.softplus(neg_term) * negative_mask
        neg_loss = (neg_loss.sum(dim=1) / neg_counts).mean()
        
        # 총 손실: 양성 손실 + 음성 손실
        total_loss = pos_loss + neg_loss
        
        return total_loss

    def training_step(self, batch, batch_idx):
        """학습 배치 처리"""
        self._step_counter += 1  # 카운터 증가
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch["img_id"]
        
        # 순방향 패스
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        
        # 다중 양성 SigLIP 손실 계산
        loss = self.compute_multi_positive_siglip_loss(image_embeds, text_embeds, img_ids)
        
        # 학습 손실 로그
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # 학습 진행상황 모니터링을 위한 추가 로그 (100배치마다)
        if self._step_counter % 100 == 0:  # global_step 대신 자체 카운터 사용
            with torch.no_grad():
                # 대각선 요소에 대한 정확도 계산 (기본 양성 쌍)
                sim_matrix = torch.matmul(image_embeds, text_embeds.t())
                accuracy = (sim_matrix.argmax(dim=1) == torch.arange(len(sim_matrix), device=sim_matrix.device)).float().mean()
                self.log("train_accuracy", accuracy, prog_bar=True)
                
                # SigLIP 손실 파라미터 로그
                self.log("a_param", self.a.data, prog_bar=True)
                self.log("b_param", self.b.data, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """검증 배치 처리"""
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch["img_id"]
        
        # 순방향 패스
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        
        # 검증 손실 계산
        val_loss = self.compute_multi_positive_siglip_loss(image_embeds, text_embeds, img_ids)
        
        # 스텝 손실 로그
        self.log("val_loss_step", val_loss, prog_bar=False)
        
        # on_validation_epoch_end에서 처리할 값 반환
        return {
            "val_loss": val_loss,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "img_ids": img_ids
        }

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """검증 배치 종료 시 출력 저장"""
        self._val_outputs.append(outputs)

    def on_validation_epoch_end(self):
        """검증 에폭 종료 시 출력 처리 - 보다 강건한 버전"""
        if not self._val_outputs:
            return
        
        try:
            # 평균 검증 손실 계산
            val_losses = torch.stack([o["val_loss"] for o in self._val_outputs])
            avg_val_loss = val_losses.mean()
            self.log("val_loss", avg_val_loss, prog_bar=True)
            
            # 모든 검증 임베딩 및 이미지 ID 연결
            all_image_embeds = torch.cat([o["image_embeds"] for o in self._val_outputs], dim=0)
            all_text_embeds = torch.cat([o["text_embeds"] for o in self._val_outputs], dim=0)
            all_img_ids = torch.cat([o["img_ids"] for o in self._val_outputs], dim=0)
            
            # 텍스트->이미지 유사도 행렬 계산
            sim_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
            
            # 다중 양성 Recall@k 계산 (텍스트->이미지만 계산)
            t2i_recall_at_k = self.compute_simple_recall(sim_matrix, ks=[1, 5, 10])
            
            # Recall 메트릭 로그
            for k, v in t2i_recall_at_k.items():
                self.log(f"val_recall@{k}", v, prog_bar=True)
                
        except Exception as e:
            print(f"Error in validation epoch end: {e}")
        finally:
            # 항상 저장된 출력 정리
            self._val_outputs.clear()
            # GPU 메모리 정리
            torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        """테스트 배치 처리"""
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch["img_id"]
        
        # 순방향 패스
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        
        return {
            "image_embeds": image_embeds, 
            "text_embeds": text_embeds,
            "img_ids": img_ids
        }

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """테스트 배치 종료 시 출력 저장"""
        self.test_image_embeds.append(outputs["image_embeds"])
        self.test_text_embeds.append(outputs["text_embeds"])
        self.test_img_ids.append(outputs["img_ids"])

    def on_test_epoch_end(self):
        """테스트 에폭 종료 시 출력 처리"""
        if not self.test_image_embeds or not self.test_text_embeds:
            return
        
        try:    
            # 모든 배치의 임베딩 및 이미지 ID 연결
            all_image_embeds = torch.cat(self.test_image_embeds, dim=0)
            all_text_embeds = torch.cat(self.test_text_embeds, dim=0)
            all_img_ids = torch.cat(self.test_img_ids, dim=0)
            
            # 유사도 행렬 계산
            sim_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
            
            # 간단한 Recall@k 계산 (텍스트->이미지)
            t2i_recall_at_k = self.compute_simple_recall(sim_matrix, ks=[1, 5, 10])
            
            # Recall 메트릭 로그
            for k, v in t2i_recall_at_k.items():
                self.log(f"test_recall@{k}", v, prog_bar=True)
                
            # 결과 출력
            print("\n" + "="*50)
            print("Test Results:")
            print("-"*50)
            print("Text-to-Image Retrieval:")
            for k, v in t2i_recall_at_k.items():
                print(f"  Recall@{k}: {v:.4f}")
            print("="*50)
            
        except Exception as e:
            print(f"Error in test epoch end: {e}")
        finally:
            # 저장된 임베딩 정리
            self.test_image_embeds.clear()
            self.test_text_embeds.clear()
            self.test_img_ids.clear()
            # GPU 메모리 정리
            torch.cuda.empty_cache()

    def compute_simple_recall(self, similarity_matrix, ks=[1, 5, 10]):
        """
        기본 Recall@k 메트릭 계산 - 간단하고 안전한 버전
        
        Args:
            similarity_matrix: 쿼리와 항목 간의 유사도 행렬 (Q, I)
            ks: Recall 계산을 위한 k 값 목록
            
        Returns:
            recall_scores: k 값에 따른 Recall 점수를 매핑하는 딕셔너리
        """
        # 대각선 요소가 정답인 경우 사용
        num_queries = similarity_matrix.size(0)
        labels = torch.arange(num_queries, device=similarity_matrix.device)
        
        recall_scores = {}
        for k in ks:
            # 상위 k개 인덱스 구하기
            _, indices = similarity_matrix.topk(min(k, similarity_matrix.size(1)), dim=1)
            # 정답이 상위 k개에 있는지 확인
            correct = torch.any(indices == labels.unsqueeze(1), dim=1)
            # 평균 계산
            recall_scores[k] = correct.float().mean().item()
        
        return recall_scores

    def on_train_end(self):
        """학습 종료 시 메모리 정리"""
        # 메모리 정리
        self._val_outputs.clear()
        self.test_image_embeds.clear()
        self.test_text_embeds.clear()
        self.test_img_ids.clear()
        
        # 캐시 정리
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        """옵티마이저 및 학습률 스케줄러 구성"""
        # 최적화할 매개변수
        params_with_grad = []
        
        # 혼합 토큰 및 크로스 어텐션 매개변수
        params_with_grad.append({
            "params": [self.mixture_tokens],
            "lr": self.learning_rate * 2.0,
            "name": "mixture_tokens"
        })
        
        params_with_grad.append({
            "params": self.cross_attention.parameters(),
            "lr": self.learning_rate,
            "name": "cross_attention"
        })
        
        # 프로젝션 레이어 매개변수
        params_with_grad.append({
            "params": self.image_proj.parameters(),
            "lr": self.learning_rate,
            "name": "image_proj"
        })
        
        params_with_grad.append({
            "params": self.text_proj.parameters(),
            "lr": self.learning_rate,
            "name": "text_proj"
        })
        
        # SigLIP 손실 파라미터
        params_with_grad.append({
            "params": [self.a, self.b],
            "lr": self.learning_rate * 5.0,
            "name": "siglip_params"
        })
        
        # 이미지 인코더가 동결되지 않은 경우
        if not self.hparams.freeze_image_encoder:
            params_with_grad.append({
                "params": self.image_encoder.parameters(),
                "lr": self.learning_rate * 0.1,
                "name": "image_encoder"
            })
        
        # 텍스트 인코더가 동결되지 않은 경우
        if not self.hparams.freeze_text_encoder:
            params_with_grad.append({
                "params": self.text_encoder.parameters(),
                "lr": self.learning_rate * 0.1,
                "name": "text_encoder"
            })
        
        # AdamW 옵티마이저 생성
        optimizer = torch.optim.AdamW(
            params_with_grad,
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.98),
            eps=1e-6
        )
        
        # 웜업이 있는 코사인 학습률 스케줄러
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[group["lr"] for group in params_with_grad],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.05,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=10000.0
            ),
            "interval": "step",
            "frequency": 1
        }
        
        return [optimizer], [scheduler]


# =============================================================================
# 4. 메인 함수: 데이터셋 로딩, 모델 설정, 학습 및 평가
# =============================================================================
def main():
    """메인 함수: 데이터셋 로딩, 모델 설정, 학습 및 평가"""
    # Flickr30k 데이터셋을 HuggingFace에서 로드
    print("Loading Flickr30k dataset...")
    dataset = load_dataset("nlphuji/flickr30k")
    
    # 'split' 필드로 데이터셋 분할
    train_dataset = dataset.filter(lambda x: x["split"] == "train")
    valid_dataset = dataset.filter(lambda x: x["split"] == "val")
    test_dataset = dataset.filter(lambda x: x["split"] == "test")
    
    print(f"Train set: {len(train_dataset['test'])} samples")
    print(f"Valid set: {len(valid_dataset['test'])} samples")
    print(f"Test set: {len(test_dataset['test'])} samples")
    
    # 배치 크기를 64로 줄임 (메모리 문제 방지)
    data_module = Flickr30KDataModule(
        train_dataset_hf=train_dataset,
        valid_dataset_hf=valid_dataset,
        test_dataset_hf=test_dataset,
        batch_size=64,  # 128에서 64로 줄임
        num_workers=4,
        max_length=64
    )
    data_module.setup("fit")
    
    # 모델 생성
    model = LLIPModel(
        image_encoder_name="microsoft/swin-base-patch4-window7-224",
        text_encoder_name="roberta-large",
        embed_dim=256,
        a_init=10.0,
        b_init=-10.0,
        learning_rate=2e-5,
        num_mixture_tokens=64,
        attention_heads=8,
        attention_temperature=5.0,
        freeze_image_encoder=True,
        freeze_text_encoder=True
    )
    
    # 로거 설정
    logger = TensorBoardLogger(
        save_dir="logs",
        name="llip_multi_caption",
        default_hp_metric=False  # 하이퍼파라미터 메트릭 비활성화
    )
    
    # 조기 중지 콜백 설정
    early_stopping_callback = EarlyStopping(
        monitor="val_recall@1",  # 변경: val_t2i_recall@1 → val_recall@1
        patience=6,
        mode="max"
    )
    
    # 학습기 생성
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=5.0,  # 1.0에서 5.0으로 증가
        accumulate_grad_batches=2,  # 그래디언트 누적 배치 수 추가
        logger=logger,
        callbacks=[early_stopping_callback],
        log_every_n_steps=10,
        deterministic=True,
        enable_checkpointing=False,  # 체크포인트 비활성화
        enable_model_summary=True,
        num_sanity_val_steps=0  # 안정성 검증 단계 비활성화
    )
    
    # 모델 학습
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # 최종 모델 테스트
    print("Testing final model...")
    trainer.test(model=model, datamodule=data_module)
    
    print("Training and evaluation complete!")


if __name__ == "__main__":
    main()