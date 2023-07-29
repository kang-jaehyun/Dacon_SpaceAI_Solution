import torch
import torch.multiprocessing as mp
import sys
sys.path.append("/workspace/Mask2former-MP")
import os
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from mask2former.test_time_augmentation import SemanticSegmentorWithTTA
from detectron2.modeling import build_model
from mask2former import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config
from tqdm import tqdm
import detectron2.data.transforms as T
import warnings
from torch.multiprocessing import Manager
from detectron2.checkpoint import DetectionCheckpointer
warnings.filterwarnings("ignore")

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


EXP_NAME="mfs_refineslice_base448_dinatl_augver1"
EPOCHS='final'
TTA = 0
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
cfg.TEST.AUG.MIN_SIZES = [448, 560, 672]
val_img_dir = '/workspace/datasets/dacon/val_fold_slice_img'
val_mask_dir = '/workspace/datasets/dacon/val_fold_slice_gt'

lst = sorted(os.listdir(val_img_dir))
if SIZE > 0:
    lst = lst[:SIZE]
else:
    pass

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
    model = build_model(cfg)
    model = model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    if TTA:
        ttamodel = SemanticSegmentorWithTTA(cfg, model, batch_size=1)
    else:
        predictor = DefaultPredictor(cfg)
    ciou = 0
    cdice = 0
    cnt = 0
    aug = T.ResizeShortestEdge([448, 448], 448)

    for i, filename in tqdm(enumerate(lst[start_idx:end_idx]), total=len(lst[start_idx:end_idx])):
        img_path = os.path.join(val_img_dir, filename)
        mask_path = os.path.join(val_mask_dir, filename)

        original_image = cv2.imread(img_path)
        
        # gt = torch.from_numpy(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
        
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        target_image = aug.get_transform(original_image).apply_image(original_image)
        target_image = torch.as_tensor(target_image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": target_image, "height": height, "width": width}
        if TTA:
            output = ttamodel([inputs])
            
            mask = output[0]['sem_seg'].argmax(dim=0)
        else:
            with torch.no_grad():
                outputs = model([inputs])

            mask = outputs[0]["sem_seg"].argmax(dim=0)

        assert mask.shape == (256,256)
        gt = torch.from_numpy(gt).to(mask.device)
        union = torch.logical_or(gt, mask)

        if union.sum() == 0:
            continue
        
        iou = compute_iou(gt, mask).cpu().numpy()
        dice = compute_dice(gt, mask).cpu().numpy()

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
