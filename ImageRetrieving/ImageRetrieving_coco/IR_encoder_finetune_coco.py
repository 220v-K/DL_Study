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
# 2. COCO 데이터셋 클래스
# -----------------------------------------------------------------------------
class COCODataset(Dataset):
    """로컬 파일 시스템에서 MS-COCO 데이터셋을 로드하는 클래스"""
    def __init__(self, split='train', root_dir='./ImageRetrieving_coco', image_size=224, max_length=77):
        self.split = split
        self.root_dir = root_dir
        
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
        
        # 데이터 로드
        self.annotations = self._load_annotations()
        
        # 이미지 ID -> 인덱스 매핑 생성
        self.img_id_to_idx = {}
        self.index_map = []
        self.create_mapping()
        
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
        
    def create_mapping(self):
        """데이터셋의 이미지 ID와 캡션 인덱스 매핑 생성"""
        for i, ann in enumerate(self.annotations):
            img_id = str(ann['image_id'])
            
            # 이미지가 처음 등장하는 경우 매핑 추가
            if img_id not in self.img_id_to_idx:
                self.img_id_to_idx[img_id] = len(self.img_id_to_idx)
            
            # 캡션 인덱스 추가
            caption_idx = i
            self.index_map.append((img_id, caption_idx))
        
        print(f"총 {len(self.img_id_to_idx)}개의 유니크한 이미지와 {len(self.index_map)}개의 캡션이 있습니다.")
            
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        img_id, caption_idx = self.index_map[idx]
        annotation = self.annotations[caption_idx]
        
        # 이미지 로드 및 전처리
        image_path = os.path.join(self.img_dir, annotation['file_name'])
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"이미지 로드 오류: {image_path} - {e}")
            # 오류 발생시 검은색 이미지로 대체
            image = torch.zeros(3, 224, 224)
        
        # 캡션 토큰화
        caption = annotation['caption']
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "img_id": img_id,
            "caption_idx": caption_idx,
            "caption": caption
        }
    
    def get_unique_image_sample(self, img_id):
        """특정 이미지 ID에 대한 샘플 반환 (유니크한 이미지 배치 구성용)"""
        # 해당 이미지 ID의 모든 캡션 인덱스 찾기
        caption_indices = [i for i, (id_, _) in enumerate(self.index_map) if id_ == img_id]
        
        if not caption_indices:
            # 캡션이 없는 경우 (일반적으로 발생하지 않음)
            return None
        
        # 무작위로 하나의 캡션 선택
        caption_idx = random.choice(caption_indices)
        return self[caption_idx]

# -----------------------------------------------------------------------------
# 3. DataModule
# -----------------------------------------------------------------------------
class COCODataModule(pl.LightningDataModule):
    def __init__(self,
                batch_size=64,
                num_workers=4,
                image_size=224,
                max_length=77,
                root_dir='./ImageRetrieving_coco',
                use_unique_images_in_batch=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_length = max_length
        self.root_dir = root_dir
        self.use_unique_images_in_batch = use_unique_images_in_batch

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # 훈련 데이터셋 로드
            self.train_dataset = COCODataset(
                split='train',
                root_dir=self.root_dir,
                image_size=self.image_size,
                max_length=self.max_length
            )
            
            # 검증 데이터셋 로드
            self.valid_dataset = COCODataset(
                split='val',
                root_dir=self.root_dir,
                image_size=self.image_size,
                max_length=self.max_length
            )
            
            print(f"학습 데이터셋 크기: {len(self.train_dataset)}")
            print(f"검증 데이터셋 크기: {len(self.valid_dataset)}")
            
        if stage == "test" or stage is None:
            # 테스트 데이터셋 로드 (검증 데이터셋과 동일)
            self.test_dataset = COCODataset(
                split='test',
                root_dir=self.root_dir,
                image_size=self.image_size,
                max_length=self.max_length
            )
            print(f"테스트 데이터셋 크기: {len(self.test_dataset)}")

    def train_dataloader(self):
        if self.use_unique_images_in_batch:
            # 유니크한 이미지 ID를 사용하는 커스텀 데이터셋
            class UniqueImageDataset(Dataset):
                def __init__(self, base_dataset):
                    self.base_dataset = base_dataset
                    self.unique_image_ids = list(base_dataset.img_id_to_idx.keys())

                def __len__(self):
                    return len(self.unique_image_ids)

                def __getitem__(self, idx):
                    # 이미지 ID 가져오기
                    img_id = self.unique_image_ids[idx]
                    sample = self.base_dataset.get_unique_image_sample(img_id)
                    if sample is None:
                        # 유효하지 않은 샘플인 경우, 다른 샘플 반환
                        return self.__getitem__((idx + 1) % len(self))
                    return sample

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
                 lr_decay_factor=0.8):  # 레이어별 학습률 감소 계수 추가
        super().__init__()
        self.save_hyperparameters()

        # 하이퍼파라미터 저장
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.lr_decay_factor = lr_decay_factor

        # 인코더 초기화
        self.image_encoder = SwinModel.from_pretrained(
            image_encoder_name,
            cache_dir="./model_cache"
        )
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_name)
        
        # 기본적으로 모든 파라미터 동결
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # Vision Encoder (Swin Transformer)의 마지막 6개 레이어 미세 조정 설정
        # Swin Transformer는 stage별로 레이어가 구성되어 있으므로 마지막 스테이지의 레이어 해제
        # 일반적으로 마지막 스테이지가 가장 고수준 특성을 담당
        trainable_swin_layers = []
        if hasattr(self.image_encoder, 'encoder') and hasattr(self.image_encoder.encoder, 'layers'):
            # 마지막 스테이지의 블록들 훈련 활성화
            swin_stages = self.image_encoder.encoder.layers
            if len(swin_stages) >= 1:  # 적어도 하나의 스테이지가 있는지 확인
                last_stage = swin_stages[-1]
                if hasattr(last_stage, 'blocks'):
                    num_blocks = len(last_stage.blocks)
                    # 마지막 6개 블록 또는 가능한 모든 블록 활성화
                    num_trainable = min(6, num_blocks)
                    for i in range(num_blocks - num_trainable, num_blocks):
                        block = last_stage.blocks[i]
                        for param in block.parameters():
                            param.requires_grad = True
                        trainable_swin_layers.append((f"encoder.layers.{len(swin_stages)-1}.blocks.{i}", block))
                        
        # Text Encoder (RoBERTa)의 마지막 6개 레이어 미세 조정 설정
        trainable_roberta_layers = []
        if hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'layer'):
            roberta_layers = self.text_encoder.encoder.layer
            num_layers = len(roberta_layers)
            # 마지막 6개 레이어 활성화
            num_trainable = min(6, num_layers)
            for i in range(num_layers - num_trainable, num_layers):
                layer = roberta_layers[i]
                for param in layer.parameters():
                    param.requires_grad = True
                trainable_roberta_layers.append((f"encoder.layer.{i}", layer))

        # 미세 조정할 레이어 저장 (나중에 옵티마이저 설정에 사용)
        self.trainable_swin_layers = trainable_swin_layers
        self.trainable_roberta_layers = trainable_roberta_layers
        
        # 로깅을 위해 미세 조정 가능한 레이어 수 출력
        print(f"미세 조정 가능한 Swin Transformer 레이어: {len(trainable_swin_layers)}")
        print(f"미세 조정 가능한 RoBERTa 레이어: {len(trainable_roberta_layers)}")

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

    def compute_simple_infonce_loss(self, image_embeds, text_embeds):
        """
        단순한 InfoNCE 손실 계산 - 각 이미지-텍스트 쌍을 독립적으로 처리
        
        Args:
            image_embeds: 이미지 임베딩 [batch_size, embed_dim]
            text_embeds: 텍스트 임베딩 [batch_size, embed_dim]
        """
        device = image_embeds.device
        batch_size = image_embeds.size(0)
        
        # 유사도 행렬 계산
        logits = torch.matmul(text_embeds, image_embeds.t()) / self.temperature
        
        # 타겟: 대각선 요소 (자신의 쌍이 양성)
        targets = torch.arange(batch_size, device=device)
        
        # 텍스트→이미지 방향 손실 (각 텍스트가 자신의 이미지를 찾도록)
        t2i_loss = F.cross_entropy(logits, targets)
        
        # 이미지→텍스트 방향 손실 (각 이미지가 자신의 텍스트를 찾도록)
        i2t_loss = F.cross_entropy(logits.t(), targets)
        
        # 최종 손실 - 양방향 평균
        loss = (t2i_loss + i2t_loss) / 2
        
        return loss

    def configure_optimizers(self):
        # 레이어별 학습률 설정을 위한 파라미터 그룹 생성
        param_groups = []
        
        # 프로젝션 레이어 (기본 학습률)
        param_groups.append({"params": self.image_proj.parameters(), "lr": self.learning_rate})
        param_groups.append({"params": self.text_proj.parameters(), "lr": self.learning_rate})
        
        # Vision Encoder 레이어 (깊이에 따라 학습률 증가)
        for i, (name, layer) in enumerate(reversed(self.trainable_swin_layers)):
            # 더 깊은 레이어(인덱스가 작은)에 더 낮은 학습률 적용
            layer_lr = self.learning_rate * (self.lr_decay_factor ** (len(self.trainable_swin_layers) - i - 1))
            param_groups.append({
                "params": layer.parameters(),
                "lr": layer_lr,
                "name": f"swin_{name}"
            })
            print(f"Swin 레이어 {name}의 학습률: {layer_lr:.6f}")
        
        # Text Encoder 레이어 (깊이에 따라 학습률 증가)
        for i, (name, layer) in enumerate(reversed(self.trainable_roberta_layers)):
            # 더 깊은 레이어(인덱스가 작은)에 더 낮은 학습률 적용
            layer_lr = self.learning_rate * (self.lr_decay_factor ** (len(self.trainable_roberta_layers) - i - 1))
            param_groups.append({
                "params": layer.parameters(),
                "lr": layer_lr,
                "name": f"roberta_{name}"
            })
            print(f"RoBERTa 레이어 {name}의 학습률: {layer_lr:.6f}")
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        
        # 학습률 스케줄러 (옵션)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        # )
        # return [optimizer], [scheduler]
        
        return optimizer
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch["img_id"]
        
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        loss = self.compute_simple_infonce_loss(image_embeds, text_embeds)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        img_ids = batch["img_id"]
        
        image_embeds, text_embeds = self(images, input_ids, attention_mask)
        val_loss = self.compute_simple_infonce_loss(image_embeds, text_embeds)
        
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
        
        # 모든 임베딩 수집
        all_image_embeds = torch.cat([o["image_embeds"] for o in self._val_outputs], dim=0)
        all_text_embeds = torch.cat([o["text_embeds"] for o in self._val_outputs], dim=0)
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
        
        # 단순 Recall@K 계산 (각 텍스트가 자신의 이미지를 찾는 능력)
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
        # 모든 임베딩 수집
        all_image_embeds = torch.cat(self.test_image_embeds, dim=0)
        all_text_embeds = torch.cat(self.test_text_embeds, dim=0)
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(all_text_embeds, all_image_embeds.t())
        
        # 단순 Recall@K 계산
        recall_at_k = self.compute_recall(similarity_matrix, ks=[1, 5, 10])
        
        for k, v in recall_at_k.items():
            self.log(f"test_recall@{k}", v, prog_bar=True)
        
        print(f"[테스트 결과] 테스트 Recall: {recall_at_k}")
        self.test_image_embeds.clear()
        self.test_text_embeds.clear()

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

# -----------------------------------------------------------------------------
# 6. 메인 실행 코드
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 데이터 모듈 초기화
    data_module = COCODataModule(
        batch_size=64,  # 32에서 64로 증가
        num_workers=4,
        image_size=224,
        max_length=77,
        root_dir='/home/gpu_04/jw2020/ImageRetrieving_coco',
        use_unique_images_in_batch=False  # 중복 이미지 허용 (각 캡션을 별도 샘플로 처리)
    )
    
    # 모델 초기화
    model = ImprovedImageTextModel(
        image_encoder_name="microsoft/swin-base-patch4-window7-224",
        text_encoder_name="roberta-large",
        embed_dim=256,
        temperature=0.07,
        learning_rate=1e-4,
        lr_decay_factor=0.9  # 레이어별 학습률 감소 계수
    )
    
    # 체크포인트 저장 경로
    checkpoint_dir = "./checkpoints/coco_finetune"  # 새 체크포인트 디렉토리
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 콜백 정의
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # 로거 설정
    logger = TensorBoardLogger("./logs", name="coco_finetune")  # 새 로그 이름
    
    # 훈련 시작
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
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