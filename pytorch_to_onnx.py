import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Conv2D_Voice_Model
import torchaudio
import os
labels_dict=['background','yes', 'no', 'up', 'down', 'left', 'right', 'forward', 'backward']

model = Conv2D_Voice_Model(n_input=1, n_output=len(labels_dict))
model.load_state_dict(torch.load("Weight/Pytorch_Pretrain/Conv2D_Voice_Model_ACC87%.pt"))
model.eval()
dummy_input = torch.randn(1, 1,1, 16000)  # one sample, 10 features (same as the model's input size)
save_path = "Weight/Onnx"
if (os.path.isdir(save_path) == False):
    os.makedirs(save_path)
onnx_file = "Weight/Onnx/Conv2D_Voice_Model_ACC87%.onnx"
torch.onnx.export(model, dummy_input, onnx_file, verbose=True, input_names=['input'], output_names=['output'])