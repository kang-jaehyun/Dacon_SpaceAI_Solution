import os
from PIL import Image
from tqdm import tqdm
import cv2

def split_image_pil(image_path, output_folder, filename):
    # 이미지 로드
    image = Image.open(image_path)

    # 이미지를 16개의 256x256 이미지로 분할
    count = 0
    for i in range(0, 1024, 256):
        for j in range(0, 1024, 256):
            # 이미지 슬라이스
            image_slice = image.crop((j, i, j+256, i+256))

            # 이미지 저장
            output_path = os.path.join(output_folder, f"{filename}_{count}.png")
            image_slice.save(output_path)

            count += 1


def split_image_cv(image_path, output_folder, filename):
    # 이미지 로드
    image = cv2.imread(image_path)

    # 이미지를 16개의 256x256 이미지로 분할
    count = 0
    for i in range(0, 1024, 256):
        for j in range(0, 1024, 256):
            # 이미지 슬라이스
            image_slice = image[i:i+256, j:j+256]
            
            assert image_slice.shape==(256,256,3), f"image_slice.shape={image_slice.shape}"
            # 이미지 저장
            output_path = os.path.join(output_folder, f"{filename}_{count}.png")
            cv2.imwrite(output_path, image_slice)

            count += 1
            

def call(input_folder, output_folder):

    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 입력 폴더 내의 모든 이미지에 대해 분할 수행
    for filename in tqdm(sorted(os.listdir(input_folder))):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            
            image_path = os.path.join(input_folder, filename)
            if 'ann' in input_folder:
                split_image_pil(image_path, output_folder, filename.split('.')[0])
            else:
                split_image_cv(image_path, output_folder, filename.split('.')[0])


if __name__ == "__main__":
    
    for fold in range(5):
        # Crop 결과 출력 폴더 생성
        os.makedirs(f"./Satellite/img_dir/val_slice_{fold}", exist_ok=True)
        os.makedirs(f"./Satellite/ann_dir/val_slice_{fold}", exist_ok=True)

        # 입력 폴더 내의 모든 이미지에 대해 분할 수행
        call(f"./Satellite/img_dir/val_{fold}", f"./Satellite/img_dir/val_slice_{fold}")
        call(f"./Satellite/ann_dir/val_{fold}", f"./Satellite/ann_dir/val_slice_{fold}")