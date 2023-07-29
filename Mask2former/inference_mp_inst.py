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

def semantic_segmentation_inference(image_path, model):
    image = cv2.imread(image_path)
    with torch.no_grad():
        outputs = model(image)
    mask = outputs["sem_seg"].argmax(dim=0).byte().cpu().numpy()
    return mask

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    try:
        runs[1::2] -= runs[::2]
    except:
        print(np.unique(mask))
    return ' '.join(str(x) for x in runs)


# 테스트 데이터 폴더 경로
test_data_folder = "/workspace/Mask2former-MP/datasets/dacon/test_img"
# 학습된 모델 파일 경로

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)

target = "inst_tiny_basemapper_0029999" ######

model_weights_path = f"/workspace/Mask2former-MP/checkpoints/{target}.pth"
cfg.merge_from_file('/workspace/Mask2former-MP/output/dacon/inst_tiny/config.yaml')
cfg.MODEL.WEIGHTS = model_weights_path

def process_csv(csv_file, start_idx, end_idx, results):
    # 각 프로세스에서 수행될 함수
    
    cfg.MODEL.DEVICE = f"cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    
    dic = {}
    submission = pd.read_csv(csv_file)
    
    for i, id in tqdm(enumerate(submission['img_id'][start_idx:end_idx]), total=len(submission[start_idx:end_idx])):
        img_path = os.path.join(test_data_folder, id + ".png")
        image = cv2.imread(img_path)
        with torch.no_grad():
            outputs = predictor(image)
        
        score = outputs['instances'].scores
        pred_masks = outputs['instances'].pred_masks

        indices = torch.where(score >= 0.9)[0]
        # _, indices = score.topk(5)

        mask = torch.zeros((224, 224), dtype=torch.float32).to(pred_masks.device)
        for index in indices:
            mask = torch.logical_or(mask, pred_masks[index])
        mask = mask.cpu().numpy().astype(np.uint8)

        # mask = semantic_segmentation_inference(img_path, predictor)
        rle = rle_encode(mask)
        dic[id] = rle if rle else -1
    
    results.update(dic)

if __name__ == '__main__':
    # 전체 submission csv 파일 경로
    csv_file = '/workspace/datasets/dacon/sample_submission.csv'
    
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
    res.to_csv(f'/workspace/Mask2former-MP/submissions/{target}_notta.csv', index=False)
