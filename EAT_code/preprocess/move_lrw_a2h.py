import os
import subprocess
import glob

paths = glob.glob('./video/*.mp4')
root_lrw = '../lrw'
os.makedirs(os.path.join(root_lrw, 'lrw_a2h_head'), exist_ok=True)
os.makedirs(os.path.join(root_lrw, 'lrw_a2h_poseimg'), exist_ok=True)

for name in paths:
    name = os.path.basename(name)[:-4]
    # move a2h_head
    os.system(f'cp ./a2h_head/{name}.npy.gz {root_lrw}/lrw_a2h_head/')
    os.system(f'cp ./a2h_poseimg/{name}.npy.gz {root_lrw}/lrw_a2h_poseimg/')
    