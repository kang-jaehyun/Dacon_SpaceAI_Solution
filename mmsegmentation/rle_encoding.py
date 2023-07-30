import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from glob import glob

# RLE 인코딩 함수
def rle_encode(mask, is_convnext=False):
    pixels = mask.flatten()

    if is_convnext:
        pixels = pixels - 7
    
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

if __name__ == "__main__":

    result_dir_list = glob("mask_inference_result/*")

    for dir_path in result_dir_list:
        submission = pd.read_csv("../datasets/sample_submission.csv")

        is_convnext = False
        if "convnext" in dir_path:
            is_convnext = True

        for idx in tqdm(range(submission.shape[0])):
            img_id = submission['img_id'].iloc[idx]

            mask = cv2.imread(f"{dir_path}/format_results/{img_id}.png", cv2.IMREAD_GRAYSCALE)

            rle_encoding = rle_encode(mask, is_convnext)

            if rle_encoding == '':
                rle_encoding = -1
            
            submission.at[idx, 'mask_rle'] = rle_encoding

        exp_name = dir_path.split("/")[-1]
        submission.to_csv(f"../results/{exp_name}.csv", index=False)