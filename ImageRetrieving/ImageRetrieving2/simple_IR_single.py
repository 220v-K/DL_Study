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
# 2. 단일 캡션 Flickr30K 데이터셋 클래스 (전처리된 버전)
# -----------------------------------------------------------------------------
class PreprocessedFlickr30KSingleDataset(Dataset):
    """이미지당 하나의 캡션만 사용하는 전처리된 Flickr30K 데이터셋"""
    def __init__(self, image_h5_path, caption_h5_path, caption_selection='fixed', random_seed=42):
        """
        Args:
            image_h5_path: 전처리된 이미지가 저장된 H5 파일 경로
            caption_h5_path: 전처리된 캡션이 저장된 H5 파일 경로
            caption_selection: 캡션 선택 방식 - 'fixed' 또는 'random'
            random_seed: 랜덤 선택 시 사용할 시드
        """
        self.image_h5_path = image_h5_path
        self.caption_h5_path = caption_h5_path
        self.caption_selection = caption_selection
        
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
        
        # 이미지별 캡션 인덱스 구성
        self.img_to_caps = {}
        for i, img_id in enumerate(self.caption_image_ids):
            if img_id not in self.img_to_caps:
                self.img_to_caps[img_id] = []
            self.img_to_caps[img_id].append(i)
        
        # 각 이미지에 대해 단일 캡션 선택
        random.seed(random_seed)
        self.selected_caps = {}
        
        for img_id, caption_indices in self.img_to_caps.items():
            if self.caption_selection == 'fixed':
                # 항상 첫 번째 캡션 선택
                self.selected_caps[img_id] = caption_indices[0]
            elif self.caption_selection == 'random':
                # 랜덤하게 하나의 캡션 선택
                self.selected_caps[img_id] = random.choice(caption_indices)
            else:
                raise ValueError(f"지원되지 않는 캡션 선택 방식: {self.caption_selection}")
        
        # 최종 이미지 ID 및 캡션 인덱스 쌍 구성
        self.index_map = [(img_id, self.selected_caps[img_id]) for img_id in sorted(self.selected_caps.keys())]
            
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
def load_preprocessed_flickr30k_single(base_dir, split='train', caption_selection='fixed', random_seed=42):
    """단일 캡션 방식으로 전처리된 Flickr30k 데이터 로드"""
    image_h5_path = os.path.join(base_dir, f'flickr30k_{split}_images.h5')
    caption_h5_path = os.path.join(base_dir, f'flickr30k_{split}_captions.h5')
    
    return image_h5_path, caption_h5_path

# -----------------------------------------------------------------------------
# 4. DataModule
# -----------------------------------------------------------------------------
class PreprocessedFlickr30KSingleDataModule(pl.LightningDataModule):
    def __init__(self,
                 base_dir,
                 batch_size=32,
                 num_workers=4,
                 caption_selection='fixed',
                 random_seed=42):
        super().__init__()
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.caption_selection = caption_selection
        self.random_seed = random_seed

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # 훈련 데이터셋 로드
            train_img_h5, train_cap_h5 = load_preprocessed_flickr30k_single(self.base_dir, 'train', self.caption_selection, self.random_seed)
            self.train_dataset = PreprocessedFlickr30KSingleDataset(
                image_h5_path=train_img_h5,
                caption_h5_path=train_cap_h5,
                caption_selection=self.caption_selection,
                random_seed=self.random_seed
            )
            
            # 검증 데이터셋 로드
            val_img_h5, val_cap_h5 = load_preprocessed_flickr30k_single(self.base_dir, 'val', self.caption_selection, self.random_seed)
            self.valid_dataset = PreprocessedFlickr30KSingleDataset(
                image_h5_path=val_img_h5,
                caption_h5_path=val_cap_h5,
                caption_selection=self.caption_selection,
                random_seed=self.random_seed
            )
            
            print(f"[단일 캡션] 학습 데이터셋 크기: {len(self.train_dataset)}")
            print(f"[단일 캡션] 검증 데이터셋 크기: {len(self.valid_dataset)}")
            
        if stage == "test" or stage is None:
            # 테스트 데이터셋 로드
            test_img_h5, test_cap_h5 = load_preprocessed_flickr30k_single(self.base_dir, 'test', self.caption_selection, self.random_seed)
            self.test_dataset = PreprocessedFlickr30KSingleDataset(
                image_h5_path=test_img_h5,
                caption_h5_path=test_cap_h5,
                caption_selection=self.caption_selection,
                random_seed=self.random_seed
            )
            print(f"[단일 캡션] 테스트 데이터셋 크기: {len(self.test_dataset)}")

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
# 5. 이미지 텍스트 Lightning 모델
# -----------------------------------------------------------------------------
class SimpleImageTextModel(pl.LightningModule):
    def __init__(self,
                 image_encoder_name="microsoft/swin-base-patch4-window7-224",
                 text_encoder_name="roberta-large",
                 embed_dim=256,
                 temperature=0.07,
                 learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 하이퍼파라미터 저장
        self.temperature = temperature
        self.learning_rate = learning_rate

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

    def compute_loss(self, image_embeds, text_embeds):
        """
        단일 캡션 모델을 위한 단순화된 대조 손실 계산
        """
        device = image_embeds.device
        batch_size = image_embeds.size(0)
        
        # 유사도 행렬 계산
        logits = torch.matmul(text_embeds, image_embeds.t()) / self.temperature
        
        # 대조 손실을 위한 라벨 (대각선)
        labels = torch.arange(batch_size, device=device)
        
        # 텍스트→이미지 방향 손실
        t2i_loss = F.cross_entropy(logits, labels)
        
        # 이미지→텍스트 방향 손실
        i2t_loss = F.cross_entropy(logits.t(), labels)
        
        # 양방향 손실 결합
        loss = (t2i_loss + i2t_loss) / 2.0
        
        return loss
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        loss = self.compute_loss(image_embeds, text_embeds)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch["img_id"]
        
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        val_loss = self.compute_loss(image_embeds, text_embeds)
        
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
        
        # 임베딩 및 ID 추출
        all_image_embeds = torch.cat([o["image_embeds"] for o in self._val_outputs], dim=0)
        all_text_embeds = torch.cat([o["text_embeds"] for o in self._val_outputs], dim=0)
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
        
        # 리콜 계산
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
        # 임베딩 및 ID 추출
        all_image_embeds = torch.cat([o["image_embeds"] for o in self._val_outputs], dim=0)
        all_text_embeds = torch.cat([o["text_embeds"] for o in self._val_outputs], dim=0)
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
        
        # 리콜 계산
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
        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        # )
        
        # return [optimizer], [scheduler]
        return optimizer

# -----------------------------------------------------------------------------
# 6. 메인 실행 코드
# -----------------------------------------------------------------------------
def train_single_caption_model(preprocessed_dir, caption_selection='fixed', random_seed=42):
    """단일 캡션 모델 학습"""
    print(f"단일 캡션 모델 학습 시작 (선택 방식: {caption_selection}, 시드: {random_seed})")
    
    # 데이터 모듈 생성
    data_module = PreprocessedFlickr30KSingleDataModule(
        base_dir=preprocessed_dir,
        batch_size=32,
        num_workers=4,
        caption_selection=caption_selection,
        random_seed=random_seed
    )
    data_module.setup("fit")

    # 로거 및 콜백 설정
    logger = TensorBoardLogger(
        save_dir="ImageRetrieving_single",
        name=f"Flickr30k_Single_{caption_selection}"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_recall@1",
        patience=5, #! 변경: early stopping patience
        mode="max",
        min_delta=0.001
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_recall@1",
        mode="max",
        dirpath=f"checkpoints/single_{caption_selection}",
        filename=f"flickr30k-single-{caption_selection}-{{epoch:02d}}-{{val_recall@1:.4f}}",
        save_top_k=5,
        save_weights_only=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 트레이너 생성
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

    # 모델 생성
    model = SimpleImageTextModel(
        image_encoder_name="microsoft/swin-base-patch4-window7-224",
        text_encoder_name="roberta-large",
        embed_dim=256,
        temperature=0.07,
        learning_rate=5e-5 # ! 변경: lr (1e-4)
    )

    # 훈련 실행
    trainer.fit(model, data_module)

    # 테스트 실행 및 결과 저장
    test_results = trainer.test(model, datamodule=data_module)

    # 결과 저장
    with open(f"test_recall_single_{caption_selection}.txt", "w") as f:
        f.write(f"단일 캡션 테스트 리콜 지표 (선택 방식: {caption_selection}):\n")
        for result in test_results:
            for key, value in result.items():
                f.write(f"{key}: {value}\n")

    print(f"테스트 평가 결과가 'test_recall_single_{caption_selection}.txt' 파일에 저장되었습니다.")
    
    return model, trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='단일 캡션 모델 학습')
    parser.add_argument('--preprocessed_dir', type=str, default='./flickr30k_preprocessed',
                        help='전처리된 데이터셋 경로')
    parser.add_argument('--caption_selection', type=str, choices=['fixed', 'random'], default='fixed',
                        help='캡션 선택 방식: fixed(첫 번째 캡션 고정) 또는 random(랜덤 선택)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='랜덤 캡션 선택 시 사용할 시드')
    
    args = parser.parse_args()
    
    train_single_caption_model(
        preprocessed_dir=args.preprocessed_dir,
        caption_selection=args.caption_selection,
        random_seed=args.random_seed
    ) 