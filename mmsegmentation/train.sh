PORT=29500 sh tools/dist_train.sh configs/convnext-base_upernet_8xb2-ade20k_fold0.py 4 --work-dir _satellite/convnext-base_fold0 --amp
PORT=29501 sh tools/dist_train.sh configs/convnext-base_upernet_8xb2-ade20k_fold1.py 4 --work-dir _satellite/convnext-base_fold1 --amp
PORT=29502 sh tools/dist_train.sh configs/convnext-base_upernet_8xb2-ade20k_fold2.py 4 --work-dir _satellite/convnext-base_fold2 --amp
PORT=29503 sh tools/dist_train.sh configs/convnext-base_upernet_8xb2-ade20k_fold3.py 4 --work-dir _satellite/convnext-base_fold3 --amp
PORT=29504 sh tools/dist_train.sh configs/convnext-base_upernet_8xb2-ade20k_fold4.py 4 --work-dir _satellite/convnext-base_fold4 --amp


PORT=29500 sh tools/dist_train.sh configs/resnest_deeplabv3plus_fold0.py 4 --work-dir _satellite/resnest_deeplabv3plus_fold0 --amp
PORT=29501 sh tools/dist_train.sh configs/resnest_deeplabv3plus_fold1.py 4 --work-dir _satellite/resnest_deeplabv3plus_fold1 --amp
PORT=29502 sh tools/dist_train.sh configs/resnest_deeplabv3plus_fold2.py 4 --work-dir _satellite/resnest_deeplabv3plus_fold2 --amp
PORT=29503 sh tools/dist_train.sh configs/resnest_deeplabv3plus_fold3.py 4 --work-dir _satellite/resnest_deeplabv3plus_fold3 --amp
PORT=29504 sh tools/dist_train.sh configs/resnest_deeplabv3plus_fold4.py 4 --work-dir _satellite/resnest_deeplabv3plus_fold4 --amp


PORT=29500 sh tools/dist_train.sh configs/segformer_fold0.py 4 --work-dir _satellite/segformer_fold0 --amp
PORT=29501 sh tools/dist_train.sh configs/segformer_fold1.py 4 --work-dir _satellite/segformer_fold1 --amp
PORT=29502 sh tools/dist_train.sh configs/segformer_fold2.py 4 --work-dir _satellite/segformer_fold2 --amp
PORT=29503 sh tools/dist_train.sh configs/segformer_fold3.py 4 --work-dir _satellite/segformer_fold3 --amp
PORT=29504 sh tools/dist_train.sh configs/segformer_fold4.py 4 --work-dir _satellite/segformer_fold4 --amp