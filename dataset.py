
# from torch.utils.data import Dataset

# class speech_commands_dataset(Dataset):
    
#     def __init__(self,data,filtered_label,label_dict):
#         self.data = data
#         self.filtered_label = filtered_label
#         self.label_dict = label_dict 
            
#     def __len__(self):
#         return len(self.data)    
    
#     def __getitem__(self,idx):
#         waveform = self.data[idx]
#         out_labels = self.label_dict.index(self.filtered_label[idx])
#         return waveform, out_labels
    

import torchaudio  # Assuming torchaudio has been imported and tqdm if needed
from torch.utils.data import DataLoader
from pathlib import Path
import os
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm
import numpy as np
import shutil

# Define the dataset class
class speech_commands_dataset(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self,filtered_dataset_save_path):
        super(speech_commands_dataset,self).__init__('.', url='speech_commands_v0.02',
                         folder_in_archive='SpeechCommands', download=True)
        self.labels_list = ['background', 'yes', 'no', 'up', 'down', 'left', 'right', 'forward', 'backward']
        self.entire_dataset = super(speech_commands_dataset,self)
        self.filtered_dataset_save_path = filtered_dataset_save_path
        self.filtered_indices = self._get_filtered_indices()
        
    def _get_filtered_indices(self):
        indices = []
        if (not os.path.isdir(self.filtered_dataset_save_path)):
            os.makedirs(self.filtered_dataset_save_path)
        else:
            indices = np.load(os.path.join(self.filtered_dataset_save_path,"filted_dataset_indices.npy"))
            return indices

        for i in tqdm(range(self.entire_dataset.__len__())):
            waveform = self.entire_dataset.__getitem__(i)[0]
            label = self.entire_dataset.__getitem__(i)[2]
            
            if label in self.labels_list and waveform.shape == (1,16000):
                indices.append(i)
        np.save(os.path.join(self.filtered_dataset_save_path,"filted_dataset_indices.npy"),np.array(indices))
        return indices

    def __getitem__(self, index):
        index = self.filtered_indices[index]
        waveform = self.entire_dataset.__getitem__(index)[0]
        label = self.entire_dataset.__getitem__(index)[2]
        label = self.labels_list.index(label)
        return waveform, label

    def __len__(self):
        return len(self.filtered_indices)