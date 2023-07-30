import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import os

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)



if __name__ == "__main__":

    df = pd.read_parquet("./refined_train.parquet")

    for fold_index in range(5):
        for index in tqdm(range(df.shape[0])):
            train_id, img_path, rle_mask, fold = df.iloc[index]

            # read img and rle_mask from img_path and rle_mask respectively
            img = cv2.imread(f"../original_data/train_img/{train_id}.png")
            gt_mask = rle_decode(rle_mask, (1024, 1024))

            # make img_dir and ann_dir
            os.makedirs(f"./Satellite/img_dir/train_{fold_index}", exist_ok=True)
            os.makedirs(f"./Satellite/ann_dir/train_{fold_index}", exist_ok=True)

            os.makedirs(f"./Satellite/img_dir/val_{fold_index}", exist_ok=True)
            os.makedirs(f"./Satellite/ann_dir/val_{fold_index}", exist_ok=True)

            # save img and ann to img_dir and ann_dir respectively by fold_index
            if fold != fold_index:
                cv2.imwrite(f"./Satellite/img_dir/train_{fold_index}/{train_id}.png", img)
                cv2.imwrite(f"./Satellite/ann_dir/train_{fold_index}/{train_id}.png", gt_mask)

            else:
                cv2.imwrite(f"./Satellite/img_dir/val_{fold_index}/{train_id}.png", img)
                cv2.imwrite(f"./Satellite/ann_dir/val_{fold_index}/{train_id}.png", gt_mask)
    
    
    # copy test images from original_data/test_img to Satellite/img_dir/test
    os.makedirs(f"./Satellite/img_dir/test", exist_ok=True)

    for file_name in tqdm(os.listdir("../original_data/test_img")):
        img = cv2.imread(f"../original_data/test_img/{file_name}")
        cv2.imwrite(f"./Satellite/img_dir/test/{file_name}", img)

    