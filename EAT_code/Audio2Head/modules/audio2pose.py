import torch
import torch.nn as nn
from modules.util import MyResNet34
import numpy as np

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class audio2poseLSTM(nn.Module):
    def __init__(self):
        super(audio2poseLSTM,self).__init__()

        self.em_audio = MyResNet34(256, 1)
        self.em_img = MyResNet34(256, 3)

        self.lstm = nn.LSTM(512,256,num_layers=2,bias=True,batch_first=True)
        self.output = nn.Linear(256,6)

    def forward(self,x):
        img_em = self.em_img(x['img']) # encoding image as one of 256 classes (image embedding, shape=(batch_size, 256))
        result = [self.output(img_em).unsqueeze(1)] # (batch_size, 1, 6) -> unsqueeze adds a dimension; this is first frame
        bs,seqlen,_,_ = x["audio"].shape
        zero_state = torch.zeros((2,bs,256),requires_grad=True).to(img_em.device)
        cur_state = (zero_state,zero_state)
        audio = x["audio"].reshape(-1, 1, 4, 41)
        audio_em = self.em_audio(audio).reshape(bs, seqlen, 256)
        for i in range(seqlen):
            # for each frame, combine audio and image embeddings and pass through LSTM
            img_em,cur_state = self.lstm(torch.cat((audio_em[:,i:i+1],img_em.unsqueeze(1)),dim=2),cur_state)
            img_em = img_em.reshape(-1, 256)
            result.append(self.output(img_em).unsqueeze(1))
        res = torch.cat(result,dim=1)
        return res


def get_pose_from_audio(img, audio, model_path="./checkpoints/audio2head.pth.tar"):
    num_frame = len(audio) // 4
    # Changed minv and maxv to match the values in EAT_code
    # minv = np.array([-0.639, -0.501, -0.47, -102.6, -32.5, 184.6], dtype=np.float32)
    # maxv = np.array([0.411, 0.547, 0.433, 159.1, 116.5, 376.5], dtype=np.float32)
    minv = np.array([0, 0, 0, -1, -1, -1], dtype=np.float32)
    maxv = np.array([65, 65, 65, 1, 1, 1], dtype=np.float32)

    generator = audio2poseLSTM().to(DEVICE)

    ckpt_para = torch.load(model_path, map_location=DEVICE)

    generator.load_state_dict(ckpt_para["audio2pose"])
    generator.eval()

    audio_seq = []
    for i in range(num_frame):
        audio_seq.append(audio[i*4:i*4+4])

    audio = torch.from_numpy(np.array(audio_seq,dtype=np.float32)).unsqueeze(0).to(DEVICE)

    x = {}
    x["img"] = img
    x["audio"] = audio
    poses = generator(x)

    # print('audio2poseLSTM output: ', poses.shape)
    poses = poses.cpu().data.numpy()[0]

    # Simple min-max scaling
    # poses = (poses+1)/2*(maxv-minv)+minv
    # print('rotations: ',poses[:,:3])
    poses = (poses+1)/2*(maxv-minv)+minv
    rot,trans =  poses[:,:3].copy(),poses[:,3:].copy()
    return rot,trans

def get_pose_from_audio_raw(img, audio, model_path="./checkpoints/audio2head.pth.tar"):
    num_frame = len(audio) // 4
    # Changed minv and maxv to match the values in EAT_code
    # minv = np.array([-0.639, -0.501, -0.47, -102.6, -32.5, 184.6], dtype=np.float32)
    # maxv = np.array([0.411, 0.547, 0.433, 159.1, 116.5, 376.5], dtype=np.float32)
    # minv = np.array([0, 0, 0, -1, -1, -1], dtype=np.float32)
    # maxv = np.array([65, 65, 65, 1, 1, 1], dtype=np.float32)

    generator = audio2poseLSTM().to(DEVICE)

    ckpt_para = torch.load(model_path, map_location=DEVICE)

    generator.load_state_dict(ckpt_para["audio2pose"])
    generator.eval()

    audio_seq = []
    for i in range(num_frame):
        audio_seq.append(audio[i*4:i*4+4])

    audio = torch.from_numpy(
        np.array(audio_seq, dtype=np.float32)).unsqueeze(0).to(DEVICE)

    x = {}
    x["img"] = img
    x["audio"] = audio
    poses = generator(x)

    # print('audio2poseLSTM output: ', poses.shape)
    poses = poses.cpu().data.numpy()[0]

    # Simple min-max scaling
    # poses = (poses+1)/2*(maxv-minv)+minv
    # print('rotations: ',poses[:,:3])
    # poses = (poses+1)/2*(maxv-minv)+minv
    rot, trans = poses[:, :3].copy(), poses[:, 3:].copy()
    return rot, trans