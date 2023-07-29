import sys
sys.path.append("/workspace/Mask2former-MP")
import os
import cv2
import torch
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
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# 테스트 데이터 폴더 경로
test_data_folder = "/workspace/Mask2former-MP/datasets/dacon/test_img"
# 학습된 모델 파일 경로
model_weights_path = "/workspace/Mask2former-MP/checkpoints/slice_wce75_99999.pth"


cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)

target = "slice_wce75_99999" ######

cfg.merge_from_file("/workspace/Mask2former-MP/configs/dacon/semantic-segmentation/slice/swin/maskformer2_swin_base_IN21k_384_bs16_160k_res640.yaml")
cfg.MODEL.WEIGHTS = model_weights_path
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

submission = pd.read_csv('/workspace/datasets/dacon/sample_submission.csv')
dic = {}
try:
    for i, id in tqdm(enumerate(submission['img_id']), total=len(submission)):
        # if i % 100 == 0:
        #     print(f"{i} / {len(submission)}")

        img_path = os.path.join(test_data_folder, id+".png")
        mask = semantic_segmentation_inference(img_path, predictor)
        rle = rle_encode(mask)
        dic[id] = rle if rle else -1
except:
    res = pd.DataFrame(dic.items(), columns=['img_id', 'mask_rle'])
    res.to_csv(f'/workspace/Mask2former-MP/submissions/{target}.csv', index=False)

res = pd.DataFrame(dic.items(), columns=['img_id', 'mask_rle'])
res.to_csv(f'/workspace/Mask2former-MP/submissions/{target}.csv', index=False)



# for i in range(len(submission)):
#     id = submission.loc[i, "img_id"]
#     img_path = os.path.join(test_data_folder, id+".png")
#     mask = semantic_segmentation_inference(img_path, predictor)
#     rle = rle_encode(mask)
#     submission.loc[i, "mask_rle"] = rle if rle else -1

# submission.to_csv('/workspace/Mask2former-MP/submissions/submission_slice_17999.csv', index=False)