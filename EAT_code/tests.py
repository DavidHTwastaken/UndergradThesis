from modules import audioencoder, prompt
import argparse
import torch
import imageio
import numpy as np
import os

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

def main():
    # argparser = argparse.ArgumentParser()
    # # argparser.add_argument('--emotype', type=str, default='neu')
    # # args = argparser.parse_args()
    # # test_emo_mapper(args.emotype)
    # argparser.add_argument('--intensity', type=float, default=None)
    # args = argparser.parse_args()
    
    # print(args.intensity)
    # test_mimsave(f'test{args.intensity}.mp4', np.random.randint(255,size=(12,256,256,3), dtype=np.uint8))
    test_latent_extractor()

if __name__ == '__main__':
    main()