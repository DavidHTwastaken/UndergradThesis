cd EAT_code
mkdir -p lrw/lrw_images lrw/lrw_latent lrw/lrw_df32 lrw/poseimg lrw/lrw_wavs
cd preprocess

mv imgs/* ../lrw/lrw_images
mv latents/* ../lrw/lrw_latent
mv deepfeature32/* ../lrw/lrw_df32
mv poseimg/* ../lrw/poseimg
mv video_fps25/*.wav ../lrw/lrw_wavs