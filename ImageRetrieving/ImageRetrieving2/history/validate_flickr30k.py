import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import re
from collections import Counter

# 경로 설정
DATASET_DIR = "./flickr30k"
IMAGES_DIR = os.path.join(DATASET_DIR, "flickr30k_images")
RESULTS_FILE = os.path.join(DATASET_DIR, "results.csv")

def validate_flickr30k_dataset():
    """Flickr30k 데이터셋의 유효성을 검증하는 함수"""
    
    print("="*80)
    print("Flickr30k 데이터셋 검증 시작")
    print("="*80)
    
    # 1. CSV 파일 존재 확인
    print(f"\n1. 결과 파일 확인 중: {RESULTS_FILE}")
    if not os.path.exists(RESULTS_FILE):
        print(f"오류: {RESULTS_FILE} 파일이 존재하지 않습니다.")
        return
    
    # 2. 이미지 디렉토리 확인
    print(f"\n2. 이미지 디렉토리 확인 중: {IMAGES_DIR}")
    if not os.path.exists(IMAGES_DIR):
        print(f"오류: {IMAGES_DIR} 디렉토리가 존재하지 않습니다.")
        return
    
    # 3. CSV 파일 로드 및 구조 확인
    print("\n3. CSV 파일 로드 및 구조 확인 중...")
    try:
        df = pd.read_csv(RESULTS_FILE, delimiter='|')
        print(f"CSV 파일 로드 성공! 파일 크기: {df.shape}")
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {e}")
        print("다른 구분자로 시도해 보겠습니다.")
        
        try:
            df = pd.read_csv(RESULTS_FILE)
            print(f"CSV 파일 로드 성공! 파일 크기: {df.shape}")
        except Exception as e:
            print(f"CSV 파일 로드 중 오류 발생: {e}")
            return
    
    # 열 이름 확인 및 조정
    print("\n4. CSV 파일 열 이름 확인:")
    print(df.columns.tolist())
    
    # 열 이름 재조정 (필요한 경우)
    # 일반적인 Flickr30k 결과 파일의 경우 이미지 이름과 캡션 열이 있어야 함
    if len(df.columns) == 1:
        print("CSV 파일이 제대로 파싱되지 않았습니다. 다른 구분자로 시도합니다.")
        content = open(RESULTS_FILE, 'r').read()
        if '\t' in content:
            df = pd.read_csv(RESULTS_FILE, delimiter='\t')
        elif ',' in content:
            df = pd.read_csv(RESULTS_FILE, delimiter=',')
    
    # 열 이름 추측
    image_col = None
    caption_col = None
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['image', 'file', 'img', 'name']):
            image_col = col
        elif any(keyword in col.lower() for keyword in ['caption', 'text', 'description']):
            caption_col = col
    
    # 열을 찾지 못한 경우 추측
    if image_col is None or caption_col is None:
        print("이미지 또는 캡션 열을 자동으로 식별할 수 없습니다.")
        print("열 내용을 분석하여 추측하겠습니다...")
        
        for col in df.columns:
            sample_values = df[col].dropna().astype(str).tolist()[:5]
            
            # 이미지 파일명 패턴 확인
            if any('.jpg' in str(val).lower() for val in sample_values):
                image_col = col
                print(f"이미지 열로 추측됨: {col}")
            
            # 캡션 패턴 확인 (긴 텍스트)
            elif any(len(str(val).split()) > 5 for val in sample_values):
                caption_col = col
                print(f"캡션 열로 추측됨: {col}")
    
    # 여전히 열을 찾지 못한 경우
    if image_col is None or caption_col is None:
        print("CSV 파일에서 이미지 및 캡션 열을 식별할 수 없습니다.")
        print("파일 형식을 확인하고 다시 시도해주세요.")
        
        # 샘플 데이터 출력
        print("\n처음 5개 행:")
        print(df.head())
        return
    
    print(f"\n식별된 열: 이미지 열 = {image_col}, 캡션 열 = {caption_col}")
    
    # 5. 데이터 결측치 확인
    print("\n5. 결측치 확인 중...")
    missing_images = df[image_col].isna().sum()
    missing_captions = df[caption_col].isna().sum()
    
    print(f"이미지 파일명 결측치: {missing_images} ({missing_images/len(df)*100:.2f}%)")
    print(f"캡션 결측치: {missing_captions} ({missing_captions/len(df)*100:.2f}%)")
    
    # 6. 이미지 파일 존재 여부 확인
    print("\n6. 이미지 파일 존재 여부 확인 중...")
    
    # 이미지 파일명에서 경로 제거
    df['image_filename'] = df[image_col].apply(
        lambda x: os.path.basename(str(x)) if pd.notnull(x) else None
    )
    
    # 랜덤 샘플링 (전체 확인은 시간이 오래 걸림)
    sample_size = min(500, len(df))
    df_sample = df.sample(sample_size, random_state=42)
    
    missing_files = 0
    for idx, row in df_sample.iterrows():
        if pd.isnull(row['image_filename']):
            missing_files += 1
            continue
            
        img_path = os.path.join(IMAGES_DIR, row['image_filename'])
        if not os.path.exists(img_path):
            missing_files += 1
    
    print(f"샘플에서 누락된 이미지 파일: {missing_files}/{sample_size} ({missing_files/sample_size*100:.2f}%)")
    
    # 7. 캡션 품질 확인
    print("\n7. 캡션 품질 확인 중...")
    
    # 캡션 길이 분포
    df['caption_length'] = df[caption_col].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    caption_lengths = df['caption_length'].value_counts().sort_index()
    
    print("캡션 단어 수 분포:")
    for length, count in caption_lengths.items():
        if count > 10:  # 빈도가 10 이상인 경우만 출력
            print(f"  {length} 단어: {count}개 ({count/len(df)*100:.2f}%)")
    
    # 짧은 캡션 확인
    short_captions = df[df['caption_length'] < 3]
    if not short_captions.empty:
        print(f"\n짧은 캡션 (3단어 미만): {len(short_captions)}개")
        print("샘플:")
        for _, row in short_captions.head(5).iterrows():
            print(f"  {row[image_col]}: {row[caption_col]}")
    
    # 8. 이미지-캡션 매핑 시각화
    print("\n8. 이미지-캡션 매핑 시각화:")
    num_samples = 3
    random_indices = random.sample(range(len(df)), num_samples)
    
    for i, idx in enumerate(random_indices):
        row = df.iloc[idx]
        img_file = os.path.join(IMAGES_DIR, row['image_filename'])
        caption = row[caption_col]
        
        if os.path.exists(img_file):
            print(f"\n샘플 {i+1}:")
            print(f"  이미지: {row['image_filename']}")
            print(f"  캡션: {caption}")
            print(f"  이미지 경로: {img_file}")
        else:
            print(f"\n샘플 {i+1} - 이미지 파일을 찾을 수 없음: {row['image_filename']}")
    
    # 9. 요약 통계
    print("\n9. 데이터셋 요약 통계:")
    unique_images = df['image_filename'].nunique()
    total_captions = len(df)
    captions_per_image = total_captions / unique_images if unique_images > 0 else 0
    
    print(f"총 이미지 수: {unique_images}")
    print(f"총 캡션 수: {total_captions}")
    print(f"이미지 당 평균 캡션 수: {captions_per_image:.2f}")
    
    # 10. JSON 형식으로 변환 가능한지 확인
    print("\n10. 로컬 데이터셋 형식으로 변환 가능성 확인:")
    image_to_captions = {}
    
    for idx, row in df.iterrows():
        image_file = row['image_filename']
        caption = row[caption_col]
        
        if pd.isnull(image_file) or pd.isnull(caption):
            continue
            
        if image_file not in image_to_captions:
            image_to_captions[image_file] = []
            
        image_to_captions[image_file].append(str(caption))
    
    print(f"변환 가능한 이미지-캡션 매핑: {len(image_to_captions)} 이미지")
    
    # 첫 번째 예시 이미지의 캡션들 출력
    if image_to_captions:
        first_image = list(image_to_captions.keys())[0]
        captions = image_to_captions[first_image]
        
        print(f"\n이미지 '{first_image}'의 캡션 {len(captions)}개:")
        for i, cap in enumerate(captions, 1):
            print(f"  캡션 {i}: {cap}")
    
    # 11. 결론
    print("\n="*80)
    print("Flickr30k 데이터셋 검증 결과:")
    
    if missing_images > 0 or missing_captions > 0:
        print("⚠️ 결측치가 발견되었습니다.")
    else:
        print("✅ 결측치가 없습니다.")
        
    if missing_files / sample_size > 0.1:  # 10% 이상의 파일이 누락된 경우
        print("⚠️ 많은 이미지 파일이 누락되었습니다.")
    else:
        print("✅ 이미지 파일이 대부분 존재합니다.")
        
    if len(short_captions) / len(df) > 0.05:  # 5% 이상의 캡션이 짧은 경우
        print("⚠️ 짧은 캡션이 많습니다.")
    else:
        print("✅ 캡션 품질이 적절합니다.")
        
    print("="*80)
    
    return image_to_captions

def create_json_files(image_to_captions, output_dir="./flickr30k_local"):
    """검증된 데이터로 JSON 파일 생성"""
    import json
    
    print(f"\n로컬 데이터셋용 JSON 파일 생성 중...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
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
        
        with open(os.path.join(output_dir, f'captions_{split}.json'), 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
            
        print(f"{split} 세트: {len(images)} 이미지, {sum(len(captions) for captions in split_data.values())} 캡션")
    
    # 이미지 파일 심볼릭 링크 생성 안내
    print("\n이미지 파일을 다음 명령으로 연결할 수 있습니다:")
    src_path = os.path.abspath(IMAGES_DIR)
    dst_path = os.path.abspath(os.path.join(output_dir, 'images'))
    
    if os.name == 'nt':  # Windows
        print(f"Windows 명령:")
        print(f"mklink /D {dst_path} {src_path}")
    else:  # Unix/Linux/Mac
        print(f"Linux/Mac 명령:")
        print(f"ln -s {src_path}/* {dst_path}/")
    
    print(f"\n또는 다음 Python 코드로 이미지를 복사할 수 있습니다:")
    print("```python")
    print("import shutil")
    print(f"for img in image_to_captions.keys():")
    print(f"    src = os.path.join('{IMAGES_DIR}', img)")
    print(f"    dst = os.path.join('{output_dir}/images', img)")
    print(f"    shutil.copy2(src, dst)")
    print("```")
    
    return output_dir

if __name__ == "__main__":
    # 데이터셋 검증
    image_to_captions = validate_flickr30k_dataset()
    
    if image_to_captions:
        # 사용자 입력 없이 바로 JSON 파일 생성
        output_dir = create_json_files(image_to_captions)
        print(f"\nJSON 파일이 {output_dir}에 생성되었습니다.")
        
        # improved_simple_IR.py 실행 안내
        print("\n생성된 로컬 데이터셋으로 모델을 학습하려면 다음과 같이 코드를 수정하세요:")
        print("```python")
        print(f"# improved_simple_IR.py에서")
        print(f"LOCAL_DATASET_DIR = '{output_dir}'")
        print("```")
    else:
        print("\n데이터셋 검증 중 오류가 발생했습니다. 문제를 해결 후 다시 시도해주세요.") 