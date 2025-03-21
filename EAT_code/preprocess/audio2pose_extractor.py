import glob
from skimage.transform import resize
import numpy as np
import imageio
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import os
import sys
sys.path.append('../')
from ..Audio2Head.modules.audio2pose import get_pose_from_audio
from ..Audio2Head.inference import get_audio_feature_from_audio
import subprocess 
from skimage import io, img_as_float32
import cv2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    pass


def head_pose_sequence(audio_path, img_path, model_path):
    temp_audio = "./results/temp.wav"
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" %
               (audio_path, temp_audio))
    output = subprocess.call(command, shell=True, stdout=None)

    audio_feature = get_audio_feature_from_audio(temp_audio)
    frames = len(audio_feature) // 4

    img = io.imread(img_path)[:, :, :3]
    img = cv2.resize(img, (256, 256))

    img = np.array(img_as_float32(img))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    ref_pose_rot, ref_pose_trans = get_pose_from_audio(
        img, audio_feature, model_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", default='../config/vox-256-spade.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='../ckpt/pretrain_new_274.pth.tar',
                        help="path to checkpoint to restore")
    parser.add_argument("--gen", default="spade",
                        choices=["original", "spade"])
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--part", default=0, type=int, help="part emotion")

    opt = parser.parse_args()
    part = opt.part
    trainlist = glob.glob('./imgs/*')
    trainlist.sort()
    kp_detector, he_estimator = load_checkpoints(
        config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=torch.cuda.is_available() == False)
    if not os.path.exists('./latents/'):
        os.makedirs('./latents')
    #     os.makedirs('./output/latent_evp/test/')
    # for videoname in tqdm(trainlist[part*2850 : (part+1)*2850]):
    for videoname in tqdm(trainlist):
        path_frames = glob.glob(videoname+'/*.jpg')
        path_frames.sort()
        driving_frames = []
        for im in path_frames:
            driving_frames.append(imageio.imread(im))
        driving_video = [resize(frame, (256, 256))[..., :3]
                         for frame in driving_frames]

        kc, he = estimate_latent(driving_video, kp_detector, he_estimator)
        kc = kc['value'].cpu().numpy()
        for k in he:
            he[k] = torch.cat(he[k]).cpu().numpy()
        np.save('./latents/'+os.path.basename(videoname), [kc, he])
    print('=============done==============')
