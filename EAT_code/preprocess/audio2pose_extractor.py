import glob
import gzip
from skimage.transform import resize
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import os
import sys
sys.path.append('../Audio2Head')
from modules.audio2pose import get_pose_from_audio, get_pose_from_audio_raw # referring to modules inside Audio2Head
from inference import get_audio_feature_from_audio # referring to inference.py inside Audio2Head
import subprocess 
from skimage import io, img_as_float32
import cv2
from generate_poseimg import draw_annotation_box, get_rotation_matrix
from modules.util import AntiAliasInterpolation2d

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
down_pose = AntiAliasInterpolation2d(1, 0.25).to(DEVICE)
CHECKPOINT_PATH = '../Audio2Head/checkpoints/audio2head.pth.tar'

def head_pose_sequence(audio_path, img_path, raw=False):
    temp_audio = "./temp.wav"
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" %
               (audio_path, temp_audio))
    output = subprocess.call(command, shell=True, stdout=None)
    audio_feature = get_audio_feature_from_audio(temp_audio)

    img = io.imread(img_path)[:, :, :3]
    img = cv2.resize(img, (256, 256))

    img = np.array(img_as_float32(img))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    if raw:
        ref_pose_rot, ref_pose_trans = get_pose_from_audio_raw(
            img, audio_feature, CHECKPOINT_PATH)
    else:
        ref_pose_rot, ref_pose_trans = get_pose_from_audio(
            img, audio_feature, CHECKPOINT_PATH)
    return ref_pose_rot, ref_pose_trans
    

def get_pose_img(rot: np.ndarray, trans: np.ndarray):
    rot = torch.from_numpy(rot).to(DEVICE).to(torch.float32)
    yaw = rot[:, 0]
    # print(yaw)
    pitch = rot[:, 1]
    roll = rot[:, 2]
    rot = get_rotation_matrix(yaw, pitch, roll).cpu().numpy().astype(np.double)
    # print(rot)
    t = trans.astype(np.double)
    # print(t)
    poseimgs = []
    for i in range(rot.shape[0]):
        ri = rot[i]
        ti = t[i]
        img = np.zeros([256, 256])
        draw_annotation_box(img, ri, ti)
        # print(img)
        # show image to user
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        poseimgs.append(img)
    poseimgs = torch.from_numpy(np.array(poseimgs))
    down_poseimgs = down_pose(poseimgs.unsqueeze(1).to(DEVICE).to(torch.float))
    return down_poseimgs

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
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--dry", action="store_true",
                        help="dry run with fake data")


    opt = parser.parse_args()
    part = opt.part
    # trainlist = glob.glob('./imgs/*')
    trainlist = glob.glob('./video/*')
    trainlist.sort()
    
    if opt.save_path is not None and not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists('./a2h_poseimg/'):
        os.makedirs('./a2h_poseimg/')
    if not os.path.exists('./a2h_head/'):
        os.makedirs('./a2h_head/')

    for videoname in tqdm(trainlist):
        videoname = os.path.basename(videoname)[:-4]
        path_frames = glob.glob(f'./imgs/{videoname}/*.jpg')
        path_frames.sort()
        img_path = path_frames[0] # take the first frame
        audio_path = os.path.join('video_fps25', os.path.basename(videoname) + '.wav')
        if opt.dry:
            head = np.tile([32, 32, 32, 0, 0, 0], (90, 1))
        else:
            rot, trans = head_pose_sequence(audio_path, img_path, raw=opt.raw)
            head = np.concatenate([rot, trans], axis=1)
        
        poseimg = get_pose_img(head[:, :3], head[:, 3:]).cpu().numpy()
        if opt.save_path is not None:
            poseimg_out = os.path.join(opt.save_path, f'poseimg_{os.path.basename(videoname)}.npy.gz')
            head_out = os.path.join(opt.save_path, f'head_{os.path.basename(videoname)}.npy.gz')
        else:
            poseimg_out = os.path.join('a2h_poseimg',
                                    f'{os.path.basename(videoname)}.npy.gz')
            head_out = os.path.join('a2h_head',
                                f'{os.path.basename(videoname)}.npy.gz')
           
        f = gzip.GzipFile(f'{poseimg_out}', "w")
        np.save(file=f, arr=poseimg)
        f = gzip.GzipFile(f'{head_out}', "w")
        np.save(file=f, arr=head)
    print('=============done==============')
