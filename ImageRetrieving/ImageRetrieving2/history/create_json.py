import os
import json
import random
import pandas as pd
from tqdm import tqdm

# 경로 설정
DATASET_DIR = "./flickr30k"
IMAGES_DIR = os.path.join(DATASET_DIR, "flickr30k_images")
RESULTS_FILE = os.path.join(DATASET_DIR, "results.csv")
OUTPUT_DIR = "./flickr30k_local"

def create_json_files():
    """Flickr30k 데이터셋을 JSON 파일로 변환"""
    print("Flickr30k 데이터셋을 JSON 파일로 변환 중...")
    
    # CSV 파일 로드 (여러 구분자 시도)
    try:
        # 먼저 기본 구분자로 시도
        df = pd.read_csv(RESULTS_FILE)
    except Exception as e:
        print(f"기본 구분자로 로드 실패: {e}")
        try:
            # 파이프(|) 구분자로 시도
            df = pd.read_csv(RESULTS_FILE, delimiter='|')
            print("파이프(|) 구분자로 로드 성공")
        except Exception as e:
            print(f"파이프 구분자로 로드 실패: {e}")
            try:
                # 탭 구분자로 시도
                df = pd.read_csv(RESULTS_FILE, delimiter='\t')
                print("탭 구분자로 로드 성공")
            except Exception as e:
                print(f"탭 구분자로 로드 실패: {e}")
                print("CSV 파일을 로드할 수 없습니다. 파일 형식을 확인해주세요.")
                return
    
    print(f"CSV 파일 로드 성공! 파일 크기: {df.shape}")
    print(f"열 이름: {df.columns.tolist()}")
    
    # 열 이름 추측
    image_col = None
    caption_col = None
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['image', 'file', 'img', 'name']):
            image_col = col
        elif any(keyword in col.lower() for keyword in ['caption', 'text', 'description', 'comment']):
            caption_col = col
    
    if image_col is None or caption_col is None:
        print("이미지 또는 캡션 열을 자동으로 식별할 수 없습니다.")
        return
    
    print(f"식별된 열: 이미지 열 = {image_col}, 캡션 열 = {caption_col}")
    
    # 이미지 파일명에서 경로 제거
    df['image_filename'] = df[image_col].apply(
        lambda x: os.path.basename(str(x)) if pd.notnull(x) else None
    )
    
    # 이미지-캡션 매핑 생성
    image_to_captions = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="캡션 매핑 생성 중"):
        image_file = row['image_filename']
        caption = row[caption_col]
        
        if pd.isnull(image_file) or pd.isnull(caption):
            continue
            
        if image_file not in image_to_captions:
            image_to_captions[image_file] = []
            
        image_to_captions[image_file].append(str(caption))
    
    print(f"변환 가능한 이미지-캡션 매핑: {len(image_to_captions)} 이미지")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
    
    # 데이터셋을 train/val/test로 분할 (요구사항: test 1000개, train 28000개, 나머지는 valid)
    all_images = list(image_to_captions.keys())
    random.shuffle(all_images)
    
    n = len(all_images)
    test_size = 1000
    train_size = 28000
    
    # 테스트, 학습, 검증 세트 분할
    test_images = all_images[:test_size]
    train_images = all_images[test_size:test_size+train_size]
    val_images = all_images[test_size+train_size:]
    
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    # 각 분할에 대한 JSON 파일 생성
    for split, images in splits.items():
        split_data = {img: image_to_captions[img] for img in images}
        
        with open(os.path.join(OUTPUT_DIR, f'captions_{split}.json'), 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
            
        print(f"{split} 세트: {len(images)} 이미지, {sum(len(captions) for captions in split_data.values())} 캡션")
    
    print(f"\nJSON 파일이 {OUTPUT_DIR}에 생성되었습니다.")
    
    return OUTPUT_DIR

if __name__ == "__main__":
    create_json_files() 