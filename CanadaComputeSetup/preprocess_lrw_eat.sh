#!/usr/bin/bash
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=8G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-06:00:00   # DD-HH:MM:SS
#SBATCH --gres=gpu:1

PWD=$(pwd)

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

cd ..

# Extract mp4 files from test set
EXTRACT=false
while getopts "x" opt; do
  case ${opt} in
    e )
      EXTRACT=true
      ;;
    * )
      echo "Usage: $0 [-x] to extract from lrw-v1.tar"
      exit 1
      ;;
  esac
done

if [ "$EXTRACT" = true ]; then
        mkdir preprocess/video # make directory in case it doesn't exist yet
        tar --wildcards -xvf lrw-v1.tar -C "EAT_code/preprocess/video" */test/*.mp4
        find EAT_code/preprocess/video/lipread_mp4 -type f -path '*/test/*.mp4' -exec mv {} EAT_code/preprocess/video \;
        rm -r EAT_code/preprocess/video/lipread_mp4
fi

# Preprocess videos
cd EAT_code/preprocess
echo "Starting preprocessing..."
python preprocess_video.py
echo "Done preprocessing."

cd "$PWD"

source move_pre.sh