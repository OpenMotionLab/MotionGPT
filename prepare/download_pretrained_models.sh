mkdir -p checkpoints/
cd checkpoints/
echo -e "The pretrained models will stored in the 'checkpoints' folder\n"
mkdir -p mld_humanml3d_checkpoint/

git lfs install
git clone https://huggingface.co/OpenMotionLab/MotionGPT-base

echo -e "Downloading done!"
