from datasets import load_dataset
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import RobertaTokenizer, RobertaModel, SwinModel
import pytorch_lightning as pl
from PIL import Image
import numpy as np
from tqdm import tqdm

# 체크포인트 경로
CHECKPOINT_PATH = "/home/gpu_04/jw2020/ImageRetrieving_coco/checkpoints/enhanced_coco_retrieval_hard_mining/model-epoch=27-val_recall@1=0.6026.ckpt"

# Flickr30k 데이터셋 로드
dataset = load_dataset("nlphuji/flickr30k")["test"]
test_dataset = dataset.filter(lambda x: x["split"] == "test")

# 모델 클래스 정의 (체크포인트의 모델 구조와 동일하게 구성)
class EnhancedImageTextModel(pl.LightningModule):
    def __init__(self,
                 image_encoder_name="microsoft/swin-base-patch4-window7-224",
                 text_encoder_name="roberta-large",
                 embed_dim=256,
                 temperature=0.07):
        super().__init__()
        self.save_hyperparameters()

        # 하이퍼파라미터 저장
        self.temperature = temperature

        # 인코더 초기화
        self.image_encoder = SwinModel.from_pretrained(
            image_encoder_name,
            cache_dir="./model_cache"
        )
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_name)
        
        # 프로젝션 레이어 설정
        image_hidden_size = self.image_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        
        # 향상된 프로젝션 레이어 (MLP)
        self.image_proj = torch.nn.Sequential(
            torch.nn.Linear(image_hidden_size, image_hidden_size),
            torch.nn.LayerNorm(image_hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(image_hidden_size, embed_dim)
        )
        
        self.text_proj = torch.nn.Sequential(
            torch.nn.Linear(text_hidden_size, text_hidden_size),
            torch.nn.LayerNorm(text_hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(text_hidden_size, embed_dim)
        )
        
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
    
    def encode_text(self, input_ids, attention_mask):
        """텍스트 인코딩 및 임베딩 추출"""
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # CLS 토큰의 임베딩 추출
        text_features = text_outputs.last_hidden_state[:, 0]
        
        # 프로젝션 및 정규화
        text_embeds = self.text_proj(text_features)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        return text_embeds
        
    def forward(self, images, input_ids, attention_mask):
        """이미지와 텍스트 인코딩 및 임베딩 추출"""
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        
        return image_embeds, text_embeds

# Flickr30k 평가 데이터셋 클래스
class Flickr30kEvalDataset(Dataset):
    def __init__(self, dataset, image_size=224, max_length=77):
        self.dataset = dataset
        self.max_length = max_length
        
        # 토크나이저 초기화
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # 데이터 전처리
        self.images = []
        self.captions = []
        self.image_ids = []
        
        for idx, item in enumerate(dataset):
            # 'image_path' 대신 'image' 키 사용 (PIL 이미지 객체)
            image = item["image"]
            captions = item["caption"]
            
            for caption in captions:
                self.images.append(image)
                self.captions.append(caption)
                self.image_ids.append(idx)
        
        print(f"총 {len(self.images)}개의 이미지-텍스트 쌍이 로드되었습니다.")
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # 이미 PIL 이미지 객체임
        caption = self.captions[idx]
        image_id = self.image_ids[idx]
        
        try:
            # 이미 PIL 이미지이므로 변환만 적용
            image = self.transform(image)
        except Exception as e:
            print(f"이미지 변환 오류: {e}")
            image = torch.zeros(3, 224, 224)
        
        # 텍스트 토큰화
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            "image": image,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "caption": caption,
            "image_id": image_id
        }

# 평가 함수
def evaluate_zero_shot(model, dataloader, device):
    model.eval()
    model.to(device)
    
    # 전체 이미지와 텍스트 임베딩 저장
    all_image_embeds = []
    all_text_embeds = []
    all_image_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="임베딩 추출 중"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_ids = batch["image_id"]
            
            # 임베딩 추출
            image_embeds = model.encode_image(images)
            text_embeds = model.encode_text(input_ids, attention_mask)
            
            all_image_embeds.append(image_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())
            all_image_ids.extend(image_ids.tolist())
    
    # 텐서로 변환
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    all_image_ids = torch.tensor(all_image_ids)
    
    # Flickr30k는 한 이미지당 여러 캡션이 있으므로 이미지 ID별로 텍스트 임베딩 평균
    unique_image_ids = torch.unique(all_image_ids)
    unique_image_embeds = []
    
    for img_id in unique_image_ids:
        indices = (all_image_ids == img_id).nonzero().squeeze()
        if indices.dim() == 0:  # 차원이 0인 경우 (단일 인덱스)
            unique_image_embeds.append(all_image_embeds[indices.item()].unsqueeze(0))
        else:
            # 동일한 이미지 ID의 이미지 임베딩 중 첫 번째만 사용
            unique_image_embeds.append(all_image_embeds[indices[0]].unsqueeze(0))
    
    unique_image_embeds = torch.cat(unique_image_embeds, dim=0)
    
    # 중복 없는 이미지 임베딩에 대한 이미지 ID
    unique_image_ids_set = unique_image_ids.tolist()
    
    # 텍스트->이미지 검색 및 이미지->텍스트 검색 계산
    i2t_similarities = calculate_similarities(unique_image_embeds, all_text_embeds, all_image_ids, unique_image_ids_set)
    t2i_similarities = calculate_similarities(all_text_embeds, unique_image_embeds, all_image_ids, unique_image_ids_set, text_to_image=True)
    
    # 결과 출력
    print("\n제로샷 평가 결과:")
    print(f"텍스트->이미지 검색: R@1: {i2t_similarities['R@1']:.4f}, R@5: {i2t_similarities['R@5']:.4f}, R@10: {i2t_similarities['R@10']:.4f}")
    print(f"이미지->텍스트 검색: R@1: {t2i_similarities['R@1']:.4f}, R@5: {t2i_similarities['R@5']:.4f}, R@10: {t2i_similarities['R@10']:.4f}")
    
    # 평균 결과
    avg_r1 = (i2t_similarities['R@1'] + t2i_similarities['R@1']) / 2
    avg_r5 = (i2t_similarities['R@5'] + t2i_similarities['R@5']) / 2
    avg_r10 = (i2t_similarities['R@10'] + t2i_similarities['R@10']) / 2
    
    print(f"평균 결과: R@1: {avg_r1:.4f}, R@5: {avg_r5:.4f}, R@10: {avg_r10:.4f}")
    
    return {
        'i2t': i2t_similarities,
        't2i': t2i_similarities,
        'avg': {
            'R@1': avg_r1,
            'R@5': avg_r5,
            'R@10': avg_r10
        }
    }

def calculate_similarities(query_embeds, gallery_embeds, image_ids, unique_ids, text_to_image=False):
    """유사도 계산 및 Recall@K 평가"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 배치 처리로 유사도 행렬 계산 (OOM 방지)
    batch_size = 256
    num_batches = (query_embeds.size(0) + batch_size - 1) // batch_size
    
    all_recalls = {'R@1': 0, 'R@5': 0, 'R@10': 0}
    query_total = 0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, query_embeds.size(0))
        query_batch = query_embeds[start_idx:end_idx].to(device)
        gallery_device = gallery_embeds.to(device)
        
        similarities = torch.matmul(query_batch, gallery_device.t())
        
        for j, query_idx in enumerate(range(start_idx, end_idx)):
            actual_idx = j + start_idx
            
            if text_to_image:
                # 텍스트 쿼리의 경우 대응하는 이미지 ID 찾기
                if actual_idx >= len(image_ids):
                    continue
                query_image_id = image_ids[actual_idx].item()
                target_gallery_idx = unique_ids.index(query_image_id)
            else:
                # 이미지 쿼리의 경우 대응하는 텍스트의 모든 이미지 ID 인덱스 찾기
                query_image_id = unique_ids[actual_idx]
                target_gallery_indices = (image_ids == query_image_id).nonzero().squeeze(-1)
                
                if target_gallery_indices.dim() == 0:
                    target_gallery_indices = torch.tensor([target_gallery_indices.item()])
            
            sim_row = similarities[j]
            
            # 정렬된 인덱스
            sorted_indices = torch.argsort(sim_row, descending=True)
            
            if text_to_image:
                # 정답 이미지가 상위 K개 내에 있는지 확인
                r1 = target_gallery_idx in sorted_indices[:1].tolist()
                r5 = target_gallery_idx in sorted_indices[:5].tolist()
                r10 = target_gallery_idx in sorted_indices[:10].tolist()
                
                all_recalls['R@1'] += int(r1)
                all_recalls['R@5'] += int(r5)
                all_recalls['R@10'] += int(r10)
                query_total += 1
            else:
                # 상위 K개 내에서 정답 텍스트가 하나라도 있는지 확인
                sorted_indices_list = sorted_indices.cpu().tolist()
                
                has_match_in_top1 = False
                has_match_in_top5 = False
                has_match_in_top10 = False
                
                for idx in target_gallery_indices:
                    idx_val = idx.item()
                    if idx_val in sorted_indices_list[:1]:
                        has_match_in_top1 = True
                    if idx_val in sorted_indices_list[:5]:
                        has_match_in_top5 = True
                    if idx_val in sorted_indices_list[:10]:
                        has_match_in_top10 = True
                
                all_recalls['R@1'] += int(has_match_in_top1)
                all_recalls['R@5'] += int(has_match_in_top5)
                all_recalls['R@10'] += int(has_match_in_top10)
                query_total += 1
    
    # 평균 계산
    for k in all_recalls:
        all_recalls[k] = all_recalls[k] / query_total if query_total > 0 else 0
    
    return all_recalls

# 메인 실행 코드
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
    # 데이터셋 준비
    eval_dataset = Flickr30kEvalDataset(test_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4
    )
    
    # 모델 로드
    model = EnhancedImageTextModel()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"체크포인트 {CHECKPOINT_PATH}가 성공적으로 로드되었습니다.")
    
    # 제로샷 평가 실행
    results = evaluate_zero_shot(model, eval_dataloader, device)
    
    return results

if __name__ == "__main__":
    main()

