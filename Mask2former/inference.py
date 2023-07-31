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
import argparse

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


parser = argparse.ArgumentParser(description="Mask2Former")
parser.add_argument("--config", type=str, default="", required=True)
parser.add_argument("--model", type=str, default="", required=True)
parser.add_argument("--num_process", type=int, default=1, help="Number of processes for parallel execution")
parser.add_argument("--tta", type=int, default=0, help="", required=False)
parser.add_argument("--output", type=str, default="", help="", required=False)

args = parser.parse_args()

CONFIG = args.config
MODEL_WEIGHTS = args.model
NUM_PROCESS = args.num_process
TTA = args.tta
OUTPUT_DIR = '../results'
OUTPUT_FILENAME = args.model.split('/')[-1].split('.')[0]
OUTPUT = args.output
if not args.output:
    OUTPUT = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

test_data_folder = "../datasets/Satellite/img_dir/test"

print("CONFIG:", CONFIG)
print("MODEL_WEIGHTS:", MODEL_WEIGHTS)
print("NUM_PROCESS:", NUM_PROCESS)
print("TTA:", bool(TTA))
print("OUTPUT:", OUTPUT)

# 학습된 모델 파일 경로

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)

cfg.merge_from_file(CONFIG)
cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
cfg.TEST.AUG.MIN_SIZES = [448, 560, 672]

def process_csv(csv_file, start_idx, end_idx, results):
    # 각 프로세스에서 수행될 함수
    
    cfg.MODEL.DEVICE = f"cuda" if torch.cuda.is_available() else "cpu"
    # predictor = DefaultPredictor(cfg)
    
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
    csv_file = '../datasets/sample_submission.csv'
    # csv_file = '/workspace/datasets/dacon/sample.csv'
    
    # 프로세스 개수
    # submission csv를 동등하게 분할하기 위해 필요한 변수
    submission = pd.read_csv(csv_file)[:100]
    num_samples = len(submission)
    chunk_size = num_samples // NUM_PROCESS

    mp.set_start_method('spawn')
    processes = []
    manager = mp.Manager()
    results = manager.dict()
    
    for i in range(NUM_PROCESS):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < NUM_PROCESS - 1 else num_samples
        
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
    res.to_csv(f'{OUTPUT}.csv', index=False)
