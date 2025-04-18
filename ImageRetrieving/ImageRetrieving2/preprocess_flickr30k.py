import os
import sys
import json
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import argparse
from transformers import RobertaTokenizer
from torchvision import transforms
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import time

# 이미지 변환 함수 정의
def create_image_transforms(is_train=False):
    """이미지 전처리를 위한 변환 함수 생성"""
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))
        ])

def preprocess_images(image_dir, image_ids, output_file, is_train=False, batch_size=100, num_workers=4):
    """이미지 전처리 및 HDF5 파일로 저장"""
    transform = create_image_transforms(is_train)
    total_images = len(image_ids)
    
    # HDF5 파일 생성
    with h5py.File(output_file, 'w') as h5_file:
        # 데이터셋 생성 (이미지 텐서와 이미지 ID를 저장)
        # 이미지 크기는 [3, 224, 224]로 고정
        images_dataset = h5_file.create_dataset('images', 
                                               shape=(total_images, 3, 224, 224), 
                                               dtype=np.float32,
                                               chunks=(1, 3, 224, 224),
                                               compression="gzip", 
                                               compression_opts=9)
        
        # 이미지 ID 저장을 위한 데이터셋
        str_dt = h5py.special_dtype(vlen=str)
        id_dataset = h5_file.create_dataset('image_ids', 
                                           shape=(total_images,), 
                                           dtype=str_dt)
        
        # 이미지 처리 함수
        def process_image(idx, img_id):
            try:
                img_path = os.path.join(image_dir, img_id)
                pil_image = Image.open(img_path).convert("RGB")
                tensor_image = transform(pil_image)
                return idx, img_id, tensor_image.numpy()
            except Exception as e:
                print(f"이미지 처리 오류 (이미지 ID: {img_id}): {str(e)}")
                return idx, img_id, None
        
        # 멀티스레드로 이미지 처리
        processed_count = 0
        with tqdm(total=total_images, desc="이미지 전처리 중") as pbar:
            for i in range(0, total_images, batch_size):
                batch_ids = image_ids[i:i+batch_size]
                batch_indices = list(range(i, min(i+batch_size, total_images)))
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(process_image, idx, img_id) 
                              for idx, img_id in zip(batch_indices, batch_ids)]
                    
                    for future in as_completed(futures):
                        idx, img_id, tensor = future.result()
                        if tensor is not None:
                            images_dataset[idx] = tensor
                            id_dataset[idx] = img_id
                            processed_count += 1
                        pbar.update(1)
        
        print(f"성공적으로 처리된 이미지: {processed_count}/{total_images}")
        return processed_count

def preprocess_captions(caption_file, image_ids, output_file, tokenizer, max_length=96):
    """캡션 전처리 및 HDF5 파일로 저장"""
    # 캡션 데이터 로드
    with open(caption_file, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
    
    # 변환할 이미지 ID 필터링 (이미지 전처리에서 처리된 이미지만)
    image_id_set = set(image_ids)
    
    # 각 이미지별 캡션 리스트 생성
    all_captions = []
    all_image_ids = []
    image_to_caption_indices = {}
    
    for img_id in image_ids:
        if img_id in captions_data:
            # 각 이미지ID의 시작 인덱스 저장
            start_idx = len(all_captions)
            # 단일 이미지의 모든 캡션 추가
            for caption in captions_data[img_id]:
                all_captions.append(caption)
                all_image_ids.append(img_id)
            # 각 이미지에 대한 캡션 인덱스 범위 저장
            end_idx = len(all_captions)
            image_to_caption_indices[img_id] = list(range(start_idx, end_idx))
    
    # 모든 캡션 토큰화
    total_captions = len(all_captions)
    
    # 토큰화 처리
    all_input_ids = []
    all_attention_masks = []
    
    with tqdm(total=total_captions, desc="캡션 토큰화 중") as pbar:
        for i in range(0, total_captions, 1000):
            batch_captions = all_captions[i:i+1000]
            
            # 배치 토큰화
            tokenized = tokenizer(
                batch_captions,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="np"
            )
            
            all_input_ids.extend(tokenized["input_ids"])
            all_attention_masks.extend(tokenized["attention_mask"])
            pbar.update(len(batch_captions))
    
    # NumPy 배열로 변환
    all_input_ids = np.array(all_input_ids)
    all_attention_masks = np.array(all_attention_masks)
    
    # HDF5 파일 생성 및 데이터 저장
    with h5py.File(output_file, 'w') as h5_file:
        h5_file.create_dataset('input_ids', 
                              data=all_input_ids,
                              chunks=(1, max_length),
                              compression="gzip", 
                              compression_opts=9)
        
        h5_file.create_dataset('attention_masks', 
                              data=all_attention_masks,
                              chunks=(1, max_length),
                              compression="gzip", 
                              compression_opts=9)
        
        # 이미지 ID 저장
        str_dt = h5py.special_dtype(vlen=str)
        h5_file.create_dataset('image_ids', 
                              data=np.array(all_image_ids, dtype=object),
                              dtype=str_dt)
        
        # 원본 캡션 텍스트 저장
        h5_file.create_dataset('captions', 
                              data=np.array(all_captions, dtype=object),
                              dtype=str_dt)
        
        # 이미지-캡션 인덱스 매핑 저장
        img_to_cap_group = h5_file.create_group('img_to_cap_mapping')
        for img_id, caption_indices in image_to_caption_indices.items():
            img_to_cap_group.create_dataset(img_id, data=np.array(caption_indices))
    
    return total_captions, len(image_to_caption_indices)

def preprocess_flickr30k(base_dir, output_dir, splits=('train', 'val', 'test'), max_length=96, num_workers=4):
    """Flickr30k 데이터셋 전처리 메인 함수"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # RoBERTa 토크나이저 초기화
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    
    # 각 분할에 대한 전처리 수행
    for split in splits:
        print(f"\n--- {split.upper()} 데이터셋 전처리 시작 ---")
        
        # 캡션 파일 및 이미지 디렉토리 경로
        caption_file = os.path.join(base_dir, f'captions_{split}.json')
        img_dir = os.path.join(base_dir, 'images')
        
        # 캡션 파일 로드 및 이미지 ID 추출
        with open(caption_file, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
        
        image_ids = list(captions_data.keys())
        is_train = (split == 'train')
        
        # 이미지 전처리 파일 경로
        image_output_file = os.path.join(output_dir, f'flickr30k_{split}_images.h5')
        
        # 이미지 전처리 수행
        print(f"{split} 이미지 전처리 시작 (총 {len(image_ids)} 이미지)")
        processed_image_count = preprocess_images(
            image_dir=img_dir,
            image_ids=image_ids,
            output_file=image_output_file,
            is_train=is_train,
            num_workers=num_workers
        )
        
        # 성공적으로 처리된 이미지 ID만 가져옴 (HDF5 파일에서)
        with h5py.File(image_output_file, 'r') as h5_file:
            processed_image_ids = [id.decode('utf-8') for id in h5_file['image_ids']]
        
        # 캡션 전처리 파일 경로
        caption_output_file = os.path.join(output_dir, f'flickr30k_{split}_captions.h5')
        
        # 캡션 전처리 수행
        print(f"{split} 캡션 전처리 시작")
        processed_caption_count, processed_image_count_from_captions = preprocess_captions(
            caption_file=caption_file,
            image_ids=processed_image_ids,
            output_file=caption_output_file,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        print(f"{split} 데이터셋 전처리 완료:")
        print(f"  - 처리된 이미지: {processed_image_count}")
        print(f"  - 처리된 캡션: {processed_caption_count}")
        print(f"  - 캡션이 있는 이미지: {processed_image_count_from_captions}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flickr30k 데이터셋 전처리')
    parser.add_argument('--base_dir', type=str, default='./flickr30k_local',
                        help='로컬 Flickr30k 데이터셋 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default='./flickr30k_preprocessed',
                        help='전처리된 데이터를 저장할 경로')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='처리할 데이터셋 분할 (기본값: train val test)')
    parser.add_argument('--max_length', type=int, default=96,
                        help='텍스트 토큰화 최대 길이')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='이미지 처리에 사용할 워커 수')
    
    args = parser.parse_args()
    
    start_time = time.time()
    preprocess_flickr30k(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    end_time = time.time()
    
    print(f"\n전체 처리 시간: {end_time - start_time:.2f}초") 