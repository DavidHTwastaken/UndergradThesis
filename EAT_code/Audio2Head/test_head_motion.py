from inference import get_audio_feature_from_audio
from modules.audio2pose import get_pose_from_audio
import os
import cv2, torch
import numpy as np
from skimage import io, img_as_float32

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    img_path = os.path.join('demo','img','baiden.jpg')
    # 'eat_example.wav'
    audio_path = os.path.join('demo', 'audio', 'intro.wav')
    model_path = os.path.join('checkpoints', 'audio2head.pth.tar')

    img = io.imread(img_path)[:, :, :3]
    img = cv2.resize(img, (256, 256))

    img = np.array(img_as_float32(img))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    audio = get_audio_feature_from_audio(audio_path)
    rot, trans = get_pose_from_audio(img, audio, model_path)
    print(rot, trans)

if __name__ == '__main__':
    main()