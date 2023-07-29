import sys
sys.path.append("/workspace/Mask2former-MP")
import os
import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os
from tqdm import tqdm
import warnings
from torch.multiprocessing import Manager, Process
import torch.multiprocessing as mp
# import Mask2Former project
from mask2former import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config
from copy import deepcopy

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

# 테스트 데이터 폴더 경로
test_data_folder = "/workspace/Mask2former-MP/datasets/dacon/test_img"
# 학습된 모델 파일 경로
# model_weights_path = "/workspace/Mask2former-MP/output/dacon/inst_base/model_0007499.pth"
# config_path = '/workspace/Mask2former-MP/output/dacon/inst_base/config.yaml'

exp_name="mf_slice_base224_baseline2"
epoch='17999'

test_data_folder = "/workspace/Mask2former-MP/datasets/dacon/test_img"
# 학습된 모델 파일 경로
model_weights_path = f"/workspace/Mask2former-MP/checkpoints/{exp_name}/model_{epoch.zfill(7)}.pth"
config_path = f'/workspace/Mask2former-MP/configs/dacon/instance-segmentation/swin/{exp_name}.yaml'

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)

cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = model_weights_path
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# from image_processing import process_image
warnings.filterwarnings("ignore")

val_img_dir = '/workspace/Mask2former-MP/datasets/dacon/val_slice_img'
val_mask_dir = '/workspace/Mask2former-MP/datasets/dacon/val_slice_gt'

lst = sorted(os.listdir(val_img_dir))
# lst = lst[:600]

def compute_iou(mask, pred_mask):
    intersection = torch.logical_and(mask, pred_mask)
    union = torch.logical_or(mask, pred_mask)
    iou = torch.sum(intersection) / (torch.sum(union) + 1e-8)
    return iou

def compute_dice(mask, pred_mask):
    intersection = torch.logical_and(mask, pred_mask)
    dice = 2 * torch.sum(intersection) / ((torch.sum(mask) + torch.sum(pred_mask)) + 1e-8)
    return dice

def process_image(lst, start_idx, end_idx, ious, dices):
    predictor = DefaultPredictor(cfg)
    
    ciou = 0
    cdice = 0
    cnt = 0
    for i, filename in tqdm(enumerate(lst[start_idx:end_idx]), total=len(lst[start_idx:end_idx])):
        img_path = os.path.join(val_img_dir, filename)
        mask_path = os.path.join(val_mask_dir, filename)

        image = cv2.imread(img_path)
        # image = cv2.resize(image, (800, 800))
        mask = torch.from_numpy(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
        with torch.no_grad():
            outputs = predictor(image)

        score = outputs['instances'].scores
        pred_masks = outputs['instances'].pred_masks

        indices = torch.where(score >= 0.9)[0]
        # _, indices = score.topk(5)

        pred_mask = torch.zeros((256, 256), dtype=torch.float32).to(pred_masks.device)
        mask = mask.to(pred_masks.device)
        for index in indices:
            pred_mask = torch.logical_or(pred_mask, pred_masks[index])
        
        union = torch.logical_or(mask, pred_mask)

        if union.sum() == 0:
            continue

        iou = compute_iou(mask, pred_mask)
        dice = compute_dice(mask, pred_mask)

        ciou += iou
        cdice += dice
        cnt += 1.0

    ious.append((ciou/cnt).cpu().numpy())
    dices.append((cdice/cnt).cpu().numpy())


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
    print("mIoU: ", np.mean(iou_tensor).item())
    print("mDice: ", np.mean(dice_tensor).item())
