#!/usr/bin/bash
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-06:00:00   # DD-HH:MM:SS
#SBATCH --gres=gpu:1

cd EAT_code

module load StdEnv/2020
module load python/3.10.2
module load gcc opencv/4.8.0
module load scipy-stack

echo 'Done loading modules'

echo 'Creating virtual environment'
virtualenv --no-download "$SLURM_TMPDIR/env" && echo 'Done creating venv'
source $SLURM_TMPDIR/env/bin/activate && echo 'Activated venv'
echo 'Done setting up virtual environment'

pip install --upgrade --no-index pip
pip install --no-index -r requirements.txt

pip install --no-index wheelhouse/*.whl
# export CMAKE_BUILD_PARALLEL_LEVEL=1
# pip install wheelhouse/* --no-index --no-cache-dir -vvv && echo 'Done installing local dependencies'

# Extract mp4 files from test set
tar -xvf lrw-v1.tar -C "EAT_code/preprocess/video" */test/*.mp4

# Preprocess videos
cd EAT_code/preprocess
echo "Starting preprocessing..."
python preprocess.py
echo "Done preprocessing. Moving files to correct directories..."
mkdir ../lrw
mv imgs/* ../lrw/lrw_images && mv latents/* ../lrw/lrw_latent && mv deepfeature32/* ../lrw/lrw_df32 && mv poseimg/* ../lrw/poseimg && mv video_fps25/*.wav ../lrw/lrw_wavs