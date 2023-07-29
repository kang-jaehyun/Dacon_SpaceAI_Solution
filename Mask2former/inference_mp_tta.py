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
from mask2former.test_time_augmentation import SemanticSegmentorWithTTA
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

import warnings
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
TTA = 1

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

def process_csv(csv_file, start_idx, end_idx, results):
    # 각 프로세스에서 수행될 함수
    
    cfg.MODEL.DEVICE = f"cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    
    dic = {}
    submission = pd.read_csv(csv_file)
    model = build_model(cfg)
    model = model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    aug = T.ResizeShortestEdge([448, 448], 448)

    if TTA:
        ttamodel = SemanticSegmentorWithTTA(cfg, model, batch_size=1)

    for i, id in tqdm(enumerate(submission['img_id'][start_idx:end_idx]), total=len(submission[start_idx:end_idx])):
        img_path = os.path.join(test_data_folder, id + ".png")
        original_image = cv2.imread(img_path)
        
        original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        target_image = aug.get_transform(original_image).apply_image(original_image)
        target_image = torch.as_tensor(target_image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": target_image, "height": height, "width": width}
        with torch.no_grad():
            if TTA:
                outputs = ttamodel([inputs])
            else:
                outputs = model([inputs])

        mask = outputs[0]["sem_seg"].argmax(dim=0)
        assert mask.shape == (224,224)
        mask = mask.cpu().numpy()
        rle = rle_encode(mask)
        dic[id] = rle if rle else -1
    
        results.update(dic)

if __name__ == '__main__':

    # 전체 submission csv 파일 경로
    csv_file = '/workspace/datasets/dacon/sample_submission.csv'
    # csv_file = '/workspace/datasets/dacon/sample.csv'
    
    # 프로세스 개수
    num_processes = 4
    
    # submission csv를 동등하게 분할하기 위해 필요한 변수
    submission = pd.read_csv(csv_file)
    num_samples = len(submission)
    chunk_size = num_samples // num_processes

    mp.set_start_method('spawn')
    processes = []
    manager = mp.Manager()
    results = manager.dict()
    
    for i in range(num_processes):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_processes - 1 else num_samples
        
        # 프로세스 생성 및 실행
        p = mp.Process(target=process_csv, args=(csv_file, start_idx, end_idx, results))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # 결과 합치기
    dic = dict(results)
    
    # 결과 저장
    res = pd.DataFrame(dic.items(), columns=['img_id', 'mask_rle'])
    res.sort_values(by=['img_id'], inplace=True)
    res.to_csv(f'/workspace/Mask2former-MP/submissions/{EXP_NAME}_{EPOCHS}_{TTA}_{"_".join(map(str,cfg.TEST.AUG.MIN_SIZES))}.csv', index=False)
