# Extract mp4 files from test set
echo "Extracting mp4 files from test set..."
cp lrw_test_subset50/*.mp4 "./preprocess/video"
echo "Done extracting mp4 files."

# Preprocess videos
mkdir -p lrw/lrw_images lrw/lrw_latent lrw/lrw_df32 lrw/poseimg lrw/lrw_wavs
cd preprocess
echo "Starting preprocessing..."
python preprocess_video.py
mv imgs/* ../lrw/lrw_images
mv latents/* ../lrw/lrw_latent
mv deepfeature32/* ../lrw/lrw_df32
mv poseimg/* ../lrw/poseimg
mv video_fps25/*.wav ../lrw/lrw_wavs
echo "Done preprocessing."

# Run on LRW test set
cd ..
echo "Running on LRW test set..."
NAME=deepprompt_eam3d_all_final_313
mkdir -p results_lrw/"$NAME_lrw_norm"
python test_lrw_posedeep_normalize_neutral.py --name $NAME --mode 0 --part=-1
echo "Done running on LRW test set."
