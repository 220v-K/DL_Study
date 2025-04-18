import os
import json
import shutil
from tqdm import tqdm

def create_symlinks():
    """이미지 파일에 대한 심볼릭 링크 생성"""
    # 경로 설정
    src_dir = os.path.join(os.getcwd(), 'flickr30k/flickr30k_images')
    dst_dir = os.path.join(os.getcwd(), 'flickr30k_local/images')
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(dst_dir, exist_ok=True)
    
    # 모든 JSON 파일에서 이미지 ID 추출
    all_images = set()
    for split in ['train', 'val', 'test']:
        json_file = os.path.join(os.getcwd(), f'flickr30k_local/captions_{split}.json')
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_images.update(data.keys())
    
    print(f"연결할 이미지 파일 수: {len(all_images)}")
    
    # 심볼릭 링크 생성
    success = 0
    errors = []
    
    for img_id in tqdm(all_images, desc="이미지 링크 생성 중"):
        src_path = os.path.join(src_dir, img_id)
        dst_path = os.path.join(dst_dir, img_id)
        
        if os.path.exists(src_path):
            try:
                # 기존 링크가 있으면 제거
                if os.path.exists(dst_path):
                    if os.path.islink(dst_path):
                        os.unlink(dst_path)
                    else:
                        os.remove(dst_path)
                
                # 심볼릭 링크 생성
                os.symlink(src_path, dst_path)
                success += 1
            except Exception as e:
                errors.append((img_id, str(e)))
    
    print(f"성공적으로 연결된 이미지: {success}/{len(all_images)}")
    if errors:
        print(f"오류 발생: {len(errors)} 이미지")
        for i, (img_id, error) in enumerate(errors[:10], 1):
            print(f"  {i}. {img_id}: {error}")
        if len(errors) > 10:
            print(f"  ... 외 {len(errors) - 10}개 오류")

def update_improved_ir_code():
    """improved_simple_IR.py 코드를 업데이트"""
    local_dataset_dir = os.path.join(os.getcwd(), 'flickr30k_local')
    
    # 파일 경로
    file_path = os.path.join(os.getcwd(), 'improved_simple_IR.py')
    
    # 파일 내용 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # LOCAL_DATASET_DIR 업데이트
    if 'LOCAL_DATASET_DIR = ' in content:
        updated_content = content.replace(
            'LOCAL_DATASET_DIR = "./flickr30k_local"',
            f'LOCAL_DATASET_DIR = "{local_dataset_dir}"'
        )
        
        # 업데이트된 내용 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"\nimproved_simple_IR.py 업데이트 완료: LOCAL_DATASET_DIR = '{local_dataset_dir}'")
    else:
        print("\nimproved_simple_IR.py에서 LOCAL_DATASET_DIR를 찾을 수 없습니다. 수동으로 업데이트해주세요.")

if __name__ == "__main__":
    # 심볼릭 링크 생성
    create_symlinks()
    
    # improved_simple_IR.py 업데이트
    update_improved_ir_code() 