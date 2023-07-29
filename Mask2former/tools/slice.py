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
            if 'gt' in image_path:
                image_slice = image_slice.convert("L")  # grayscale로 변환
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
            if 'gt' in image_path:
                cv2.imwrite(output_path, image_slice[:,:,0])
            else:
                cv2.imwrite(output_path, image_slice)

            count += 1
            

def call(input_folder, output_folder):

    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 입력 폴더 내의 모든 이미지에 대해 분할 수행
    for filename in tqdm(sorted(os.listdir(input_folder))):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # print(filename)
            image_path = os.path.join(input_folder, filename)
            if 'gt' in input_folder:
                split_image_pil(image_path, output_folder, filename.split('.')[0])
            else:
                split_image_cv(image_path, output_folder, filename.split('.')[0])

if __name__ == "__main__":
    call("/workspace/datasets/dacon/train_gt", "/workspace/datasets/dacon/train_slice_gt")
    call("/workspace/datasets/dacon/train_img", "/workspace/datasets/dacon/train_slice_img")
    call("/workspace/datasets/dacon/val_img", "/workspace/datasets/dacon/val_slice_img")
    call("/workspace/datasets/dacon/val_gt", "/workspace/datasets/dacon/val_slice_gt")
