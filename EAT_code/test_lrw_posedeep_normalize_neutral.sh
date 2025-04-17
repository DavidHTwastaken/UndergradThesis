NAME=deepprompt_eam3d_all_final_313
mkdir -p results_lrw/"$NAME_lrw_norm"
touch results_lrw/"$NAME_lrw_norm"/loss.txt
CUDA_VISIBLE_DEVICES=0 python test_lrw_posedeep_normalize_neutral.py --name $NAME --part 0 --mode 0 &
CUDA_VISIBLE_DEVICES=1 python test_lrw_posedeep_normalize_neutral.py --name $NAME --part 2 --mode 0 &
CUDA_VISIBLE_DEVICES=2 python test_lrw_posedeep_normalize_neutral.py --name $NAME --part 1 --mode 0 &
CUDA_VISIBLE_DEVICES=3 python test_lrw_posedeep_normalize_neutral.py --name $NAME --part 3 --mode 0 &
wait