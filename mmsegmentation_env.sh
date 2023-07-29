conda create -n mmseg python=3.9
conda activate mmseg

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -U openmim
mim install "mmengine==0.8.2"
mim install "mmcv==2.0.1"
pip install "mmsegmentation==1.1.0"
pip install "mmpretrain==1.0.0"
pip install "cityscapesScripts==2.2.2"
pip install pyarrow
