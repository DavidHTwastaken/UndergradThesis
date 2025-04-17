from modules import audioencoder, prompt
import argparse
import torch
import imageio
import numpy as np
import os
import gzip
from modules.transformer import headpose_pred_to_degree
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# from ..Audio2Head.modules.audio2pose import get_pose_from_audio
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']
emo_label_full = ['angry',  'contempt',  'disgusted',
                  'fear',  'happy',  'neutral',  'sad',  'surprised']
latent_dim = 16

def test_emo_mapper(emotype: str):
    mapper = audioencoder.MappingDeepNetwork(latent_dim=16, style_dim=128, num_domains=8, hidden_dim=512)
    y_trg = emo_label.index(emotype)
    y_trg = torch.tensor(y_trg, dtype=torch.long).unsqueeze(0)
    z_trg = torch.randn(latent_dim).unsqueeze(0)
    print(y_trg.shape)
    print(z_trg.shape)
    print(z_trg)
    s_trg = mapper(z_trg, y_trg)
    print(s_trg.shape)

def test_emo_prompt(emotype: str):
    emotionprompt = prompt.EmotionPrompt()
    y_trg = emo_label.index(emotype)
    y_trg = torch.tensor(y_trg, dtype=torch.long).unsqueeze(0)
    z_trg = torch.randn(latent_dim).unsqueeze(0)
    print(y_trg.shape)
    print(z_trg.shape)
    print(z_trg)
    s_trg = prompt({'z_trg': z_trg, 'y_trg': y_trg})
    print(s_trg.shape)

def test_mimsave(video_path, predictions_gen):
    '''
    predictions_gen: list(np.array((num_frames, 256, 256, 3), dtype=np.uint8))
    '''
    imageio.mimsave(video_path, predictions_gen, fps=25.0)
    os.remove(video_path)

def test_latent_extractor():
    import glob
    from preprocess.latent_extractor import resize
    lat = 'preprocess/latents/head_movements.npy'
    kp_cano, he_driving = np.load(lat, allow_pickle=True)
    print('yaw', he_driving['yaw'])
    print('translation',he_driving['t'])
    # videoname = 'preprocess/imgs/self_vid'
    # path_frames = glob.glob(videoname+'/*.jpg')
    # path_frames.sort()
    # driving_frames = []
    # for im in path_frames:
    #     driving_frames.append(imageio.imread(im))
    # driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_frames]

    # driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(
    #     np.float32)).permute(0, 4, 1, 2, 3)
    # print(driving.shape)

def compare_latent_a2h():
    preprocess_root = 'preprocess/'
    filename = '01-01-03-02-01-02-01'
    latents = np.load(os.path.join(preprocess_root, 'latents', f'{filename}.npy'), allow_pickle=True)

    gzip_path = os.path.join(preprocess_root, 'a2h_head', f'{filename}.npy.gz')
    with gzip.open(gzip_path, 'rb') as f:
        a2h_latents = np.load(f, allow_pickle=True)
    kp_cano, he_driving = latents

    keys = ['pitch', 'yaw', 'roll']
    for key in keys:
        he_driving[key] = torch.from_numpy(he_driving[key]).to(DEVICE)
        he_driving[key] = headpose_pred_to_degree(he_driving[key])
    he_driving['t'] = torch.from_numpy(he_driving['t']).to(DEVICE)

    # plot the features from he_driving and a2h_latents on the same graph (in pairs)
    plt.figure(figsize=(10, 5))
    for i, key in enumerate(keys):
        plt.subplot(1, 4, i + 1)
        plt.plot(he_driving[key].cpu().numpy(), label='he_driving')
        plt.plot(a2h_latents[:, i], label='a2h_latents')
        plt.title(key)
        plt.legend()
    plt.subplot(1, 4, 4)
    plt.plot(he_driving['t'].cpu().numpy(), label=[f'he_driving {i}' for i in ['x','y','z']])
    plt.plot(a2h_latents[:, 3:], label=[
             f'a2h_latents {i}' for i in ['x', 'y', 'z']])
    plt.title('translation')
    plt.legend()
    plt.tight_layout()
    plt.savefig('a2h_vs_he_driving.png')
    # print(he_driving['yaw'])
    # print(a2h_latents[:,0])
    # difference
    # print(torch.sum(he_driving['yaw'] - a2h_latents[:,0]))
    # print(he_driving['t'])
    # print(a2h_latents[:,3:])
    # poseimg_path = os.path.join(preprocess_root, 'poseimg', f'{filename}.npy.gz')
    # with gzip.open(poseimg_path, 'rb') as f:
    #     poseimg = np.load(f, allow_pickle=True)
    # print(poseimg.shape)
    # print('poseimg', poseimg)

    # a2h_poseimg_path = os.path.join(preprocess_root, 'a2h_poseimg', f'{filename}.npy.gz')
    # with gzip.open(a2h_poseimg_path, 'rb') as f:
    #     a2h_poseimg = np.load(f, allow_pickle=True)
    # print(a2h_poseimg.shape)
    # print('a2h_poseimg', a2h_poseimg)

def graph_a2h():
    filename = '01-01-03-02-01-02-01'
    gzip_path = os.path.join('test_samples', f'head_{filename}.npy.gz')
    with gzip.open(gzip_path, 'rb') as f:
        a2h_latents = np.load(f, allow_pickle=True)
    minv_og = np.array([-0.639, -0.501, -0.47, -102.6, -32.5, 184.6], dtype=np.float32)
    maxv_og = np.array([0.411, 0.547, 0.433, 159.1, 116.5, 376.5], dtype=np.float32)
    minv_eat = np.array([0, 0, 0, -1, -1, -1], dtype=np.float32)
    maxv_eat = np.array([65, 65, 65, 1, 1, 1], dtype=np.float32)
    def calc_poses(poses, minv, maxv): 
        return (poses+1)/2*(maxv-minv)+minv

    og = calc_poses(a2h_latents, minv_og, maxv_og)
    eat = calc_poses(a2h_latents, minv_eat, maxv_eat)
    raw = a2h_latents
    
    keys = ['pitch', 'yaw', 'roll', 'tx', 'ty', 'tz']
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.plot(raw[:, i], label='raw')
        plt.plot(og[:, i], label='og')
        plt.plot(eat[:, i], label='eat')
        plt.title(keys[i])
        plt.legend()

    plt.tight_layout()
    plt.savefig('a2h_pose_scaling_comparison.png')

def test_audio2pose_extractor():
    from preprocess.audio2pose_extractor import get_pose_from_audio

def main():
    # argparser = argparse.ArgumentParser()
    # # argparser.add_argument('--emotype', type=str, default='neu')
    # # args = argparser.parse_args()
    # # test_emo_mapper(args.emotype)
    # argparser.add_argument('--intensity', type=float, default=None)
    # args = argparser.parse_args()
    
    # print(args.intensity)
    # test_mimsave(f'test{args.intensity}.mp4', np.random.randint(255,size=(12,256,256,3), dtype=np.uint8))
    # test_latent_extractor()
    # compare_latent_a2h()
    graph_a2h()

if __name__ == '__main__':
    main()