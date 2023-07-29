import torch
import torch.multiprocessing as mp
import sys
sys.path.append("/workspace/Mask2former-MP")
import os
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
# import Mask2Former project
from mask2former import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config
import pandas as pd
from tqdm import tqdm
import torchvision.transforms.functional as TF
import warnings
from torch.multiprocessing import Manager, Process
import torchvision.transforms as T

warnings.filterwarnings("ignore")

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

# 회전된 이미지를 생성하는 함수
def rotate_image(image, angle):
    # 이미지의 중심을 기준으로 회전하기 위해 중심 좌표 계산
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # 회전 변환 행렬 계산
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 이미지 회전
    rotated_image = cv2.warpAffine(image, matrix, (width, height))

    return rotated_image

def rotate_mask(mask, angle):
    # 마스크를 이미지로 변환
    mask_max = torch.max(mask)
    
    normed_mask = mask / mask_max
    reverse_angle = 360 - angle

    mask_image = TF.to_pil_image(normed_mask)

    rotated_mask_image = TF.rotate(mask_image, reverse_angle)
    normed_rotated_mask = TF.to_tensor(rotated_mask_image)

    mask_max = mask_max.to(normed_rotated_mask.device)
    rotated_mask = normed_rotated_mask * mask_max

    return rotated_mask

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


EXP_NAME="mfs_refineslice_base448_sl_augver1"
EPOCHS='final'
TTA = 1
SIZE = 3000
test_data_folder = "/workspace/Mask2former-MP/datasets/dacon/test_img"
# 학습된 모델 파일 경로
if EPOCHS == 'final':
    model_weights_path = f"/workspace/Mask2former-MP/checkpoints/{EXP_NAME}/model_final.pth"
else:
    model_weights_path = f"/workspace/Mask2former-MP/checkpoints/{EXP_NAME}/model_{EPOCHS.zfill(7)}.pth"
config_path = f'/workspace/Mask2former-MP/configs/dacon/semantic-segmentation/{EXP_NAME}.yaml'

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)

cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = model_weights_path

val_img_dir = '/workspace/datasets/dacon/val_fold_slice_img'
val_mask_dir = '/workspace/datasets/dacon/val_fold_slice_gt'

lst = sorted(os.listdir(val_img_dir))
if SIZE > 0:
    lst = lst[:SIZE]
else:
    pass

def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + 1.) / (np.sum(y_true) + np.sum(y_pred) + 1.)

def iou_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + 1.) / (union + 1.)

def compute_iou(mask, pred_mask):
    intersection = np.logical_and(mask, pred_mask)
    union = np.logical_or(mask, pred_mask)
    iou = np.sum(intersection) / (np.sum(union) + 1e-8)
    return iou

def compute_dice(mask, pred_mask):
    intersection = np.logical_and(mask, pred_mask)
    dice = 2 * np.sum(intersection) / ((np.sum(mask) + np.sum(pred_mask)) + 1e-8)
    return dice

def rotate_image(image, angle):

    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (width, height))

    return rotated_image

def rotate_mask(mask, angle):
    mask_max = torch.max(mask)
    
    normed_mask = mask / mask_max
    reverse_angle = 360 - angle

    mask_image = TF.to_pil_image(normed_mask)

    rotated_mask_image = TF.rotate(mask_image, reverse_angle)
    normed_rotated_mask = TF.to_tensor(rotated_mask_image)

    mask_max = mask_max.to(normed_rotated_mask.device)
    rotated_mask = normed_rotated_mask * mask_max

    return rotated_mask

def process_image(lst, start_idx, end_idx, ious, dices):
    predictor = DefaultPredictor(cfg)
    
    ciou = 0
    cdice = 0
    cnt = 0
    for i, filename in tqdm(enumerate(lst[start_idx:end_idx]), total=len(lst[start_idx:end_idx])):
        img_path = os.path.join(val_img_dir, filename)
        mask_path = os.path.join(val_mask_dir, filename)

        image = cv2.imread(img_path)
        
        # gt = torch.from_numpy(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if TTA:
            # image_tta = [(448,448), (672,672), (896,896), (896,448), (448, 896)]
            # image_tta = [(448,448), (672,672), (896,896)]
            image_tta = [(448,448)]
            # image_tta = [(896,896)]
            # rotation_angles = [0, 90, 180, 270]
            rotation_angles = [0]
            tta = torch.zeros((image.shape[0], image.shape[1], 2))
            for size in image_tta:
                target_image = cv2.resize(image, size)
                for angle in rotation_angles:
                    rotated_image = rotate_image(target_image, angle)
                    with torch.no_grad():
                        outputs = predictor(rotated_image)
                    rotated_mask = outputs["sem_seg"]

                    mask = rotate_mask(rotated_mask, angle)
                    mask = mask.permute(1,2,0).cpu().numpy()
                    resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    tta += resized_mask
            
            mask = tta.argmax(dim=2).byte().cpu().numpy()
        else:
            target_image = cv2.resize(image, (448, 448))
            with torch.no_grad():
                outputs = predictor(target_image)

            mask = outputs["sem_seg"].argmax(dim=0).byte().cpu().numpy()

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        assert mask.shape == (256,256)

        union = np.logical_or(gt, mask)

        if union.sum() == 0:
            continue
        
        iou = compute_iou(gt, mask)
        dice = compute_dice(gt, mask)

        ciou += iou
        cdice += dice
        cnt += 1.0

    ious.append(ciou/cnt)
    dices.append(cdice/cnt)


if __name__ == '__main__':
    # 프로세스 개수
    num_processes = 4

    # 데이터 분할
    chunk_size = len(lst) // num_processes
    chunks = [lst[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
    if len(lst) % num_processes != 0:
        chunks[-1] += lst[num_processes * chunk_size:]

    # 공유 변수
    mp.set_start_method('spawn')
    manager = Manager()
    iou_queue = manager.list()
    dice_queue = manager.list()
    processes = []

    # 프로세스 생성
    for i in range(num_processes):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_processes - 1 else len(lst)

        # 프로세스 생성 및 실행
        p = mp.Process(target=process_image, args=(lst, start_idx, end_idx, iou_queue, dice_queue))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    iou_list = list(iou_queue)
    dice_list = list(dice_queue)

    iou_tensor = np.array(iou_list)
    dice_tensor = np.array(dice_list)
    # print(iou_tensor)
    print(f"{EXP_NAME}_{EPOCHS}_{TTA}")
    print("mIoU: ", np.mean(iou_tensor).item())
    print("mDice: ", np.mean(dice_tensor).item())
