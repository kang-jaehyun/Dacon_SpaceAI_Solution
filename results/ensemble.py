import pandas as pd
import numpy as np
from tqdm import tqdm

convnext_results = [pd.read_csv(f'results/convnext-base_fold{i}.csv') for i in range(5)]
resnest_results = [pd.read_csv(f'results/resnest_deeplabv3plus_fold{i}.csv') for i in range(5)]
segformer_results = [pd.read_csv(f'results/segformer_fold{i}.csv') for i in range(5)]
mask2former_results = [pd.read_csv(f'results/dinat_mask2former_fold{i}.csv') for i in range(5)]
print("Loaded Results")
foldwise_csvs = [[convnext_results[i], convnext_results[i], resnest_results[i], segformer_results[i], mask2former_results[i]] for i in range(5)]

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()

    if pixels.sum() == 0:
        return -1
    
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def merge(csvs):
    merged_df = None
    for index, csv in enumerate(csvs):
        if merged_df is None:
            merged_df = csv
        else:
            merged_df = merged_df.merge(csv, on='img_id', suffixes=(f'_{index}', f'_{index+1}'))
    return merged_df

def hard_vote(row):
    masks = row[['mask_rle_1', 'mask_rle_2', 'mask_rle_3', 'mask_rle_4', 'mask_rle']].values
    assert len(masks) == 5

    mask_array = np.zeros((224, 224))
    for mask in masks:
        if isinstance(mask, str):
            decoded_mask = rle_decode(mask, (224, 224))  # Replace with your RLE decoding logic
            mask_array += decoded_mask

    mask_array = (mask_array >= 3).astype(int)  # Adjust the threshold for voting agreement
    
    return rle_encode(mask_array)  # Replace with your RLE encoding logic

foldwise_results = []
for f in range(5):
    print("Start Fold Ensemble", f)
    merged_df = merge(foldwise_csvs[f])
    merged_df['final_mask_rle'] = merged_df.apply(hard_vote, axis=1)
    merged_df = merged_df[['img_id', 'final_mask_rle']]
    merged_df.rename(columns = {'final_mask_rle':'mask_rle'}, inplace = True)
    foldwise_results.append(merged_df)
    print("End Fold Ensemble", f)

print("Start Final Ensemble")
final_merged_df = merge(foldwise_results)
final_merged_df['final_mask_rle'] = final_merged_df.apply(hard_vote, axis=1)
final_merged_df = final_merged_df[['img_id', 'final_mask_rle']]
final_merged_df.rename(columns = {'final_mask_rle':'mask_rle'}, inplace = True)
final_merged_df.to_csv('results/ensemble.csv', index=False)
print("End Final Ensemble")