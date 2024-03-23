import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Conv2D_Voice_Model
import torchaudio
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
labels_dict=['background','yes', 'no', 'up', 'down', 'left', 'right', 'forward', 'backward']
model = Conv2D_Voice_Model(n_input=1, n_output=len(labels_dict))

model.to(device)
model.load_state_dict(torch.load("Weight/Pytorch_Pretrain/Conv2D_Voice_Model_ACC87%.pt"))
waveform, sample_rate = torchaudio.load("SpeechCommands/speech_commands_v0.02/forward/0a2b400e_nohash_1.wav", normalize=False)
waveform = waveform.to(device)
model.eval()
with torch.no_grad():
    output = model(((waveform.reshape(1,1,1,16000)))/32767.0)
    output = output.squeeze()
    print(output)
    print(torch.argmax(output, dim=0))
    print(labels_dict[torch.argmax(output, dim=0)])

