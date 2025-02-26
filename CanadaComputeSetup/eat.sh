#!/usr/bin/bash
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-01:00:00   # DD-HH:MM:SS
#SBATCH --gres=gpu:1

cd EAT_code

module load StdEnv/2020
module load python/3.10.2
module load gcc opencv/4.8.0
module load scipy-stack

echo 'Done loading modules'

echo 'Creating virtual environment'
virtualenv --no-download $SLURM_TMPDIR/env && echo 'Done creating venv'
source $SLURM_TMPDIR/env/bin/activate && echo 'Activated venv'
echo 'Done setting up virtual environment'

pip install --upgrade --no-index pip
pip install --no-index -r requirements.txt

pip install --no-index wheelhouse/*.whl
# export CMAKE_BUILD_PARALLEL_LEVEL=1
# pip install wheelhouse/* --no-index --no-cache-dir -vvv && echo 'Done installing local dependencies'

python demo.py --src_img obama --emo hap --save_dir ../