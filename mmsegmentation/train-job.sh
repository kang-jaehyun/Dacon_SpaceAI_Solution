cd mmsegmentation

python tools/train.py configs/convnext-base_upernet_8xb2-ade20k_fold0.py --work-dir _satellite/convnext-base_fold0 --amp
python tools/train.py configs/convnext-base_upernet_8xb2-ade20k_fold1.py --work-dir _satellite/convnext-base_fold1 --amp
python tools/train.py configs/convnext-base_upernet_8xb2-ade20k_fold2.py --work-dir _satellite/convnext-base_fold2 --amp
python tools/train.py configs/convnext-base_upernet_8xb2-ade20k_fold3.py --work-dir _satellite/convnext-base_fold3 --amp
python tools/train.py configs/convnext-base_upernet_8xb2-ade20k_fold4.py --work-dir _satellite/convnext-base_fold4 --amp


python tools/train.py configs/resnest_deeplabv3plus_fold0.py --work-dir _satellite/resnest_deeplabv3plus_fold0 --amp
python tools/train.py configs/resnest_deeplabv3plus_fold1.py --work-dir _satellite/resnest_deeplabv3plus_fold1 --amp
python tools/train.py configs/resnest_deeplabv3plus_fold2.py --work-dir _satellite/resnest_deeplabv3plus_fold2 --amp
python tools/train.py configs/resnest_deeplabv3plus_fold3.py --work-dir _satellite/resnest_deeplabv3plus_fold3 --amp
python tools/train.py configs/resnest_deeplabv3plus_fold4.py --work-dir _satellite/resnest_deeplabv3plus_fold4 --amp


python tools/train.py configs/segformer_fold0.py --work-dir _satellite/segformer_fold0 --amp
python tools/train.py configs/segformer_fold1.py --work-dir _satellite/segformer_fold1 --amp
python tools/train.py configs/segformer_fold2.py --work-dir _satellite/segformer_fold2 --amp
python tools/train.py configs/segformer_fold3.py --work-dir _satellite/segformer_fold3 --amp
python tools/train.py configs/segformer_fold4.py --work-dir _satellite/segformer_fold4 --amp