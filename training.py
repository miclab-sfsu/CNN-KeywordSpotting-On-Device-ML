from torch.utils.data import DataLoader,random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import speech_commands_dataset
from model import Conv2D_Voice_Model
import torchaudio
from tqdm import tqdm
import os

# Load the dataset
filtered_dataset = speech_commands_dataset(filtered_dataset_save_path="filted_dataset/")

# Define the size of the split
train_size = int(0.8 * len(filtered_dataset))
test_size = len(filtered_dataset) - train_size

# Split the dataset into training and testing datasets
train_set, test_set = random_split(filtered_dataset, [train_size, test_size])


train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=256,
    shuffle=True,
    num_workers=4,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=256,
    shuffle=False,
    num_workers=4,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Conv2D_Voice_Model(n_input=1, n_output=len(filtered_dataset.labels_list))
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
criterion = nn.CrossEntropyLoss()

def train(model, epoch, log_interval):
    model.train()
    with tqdm(total=len(train_loader)) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)

            data = data.reshape(-1,1,1,16000)
            output = model(data)

            loss = criterion(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            pbar.update()


def test(model, epoch):
    model.eval()
    correct = 0
    with tqdm(total=len(test_loader)) as pbar:
        for data, target in test_loader:

            data = data.to(device)
            target = target.to(device)


            data = data.reshape(-1,1,1,16000)
            output = model(data)

            pred = output.argmax(dim=-1)
            correct += pred.squeeze().eq(target).sum().item()

            pbar.set_description(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
            pbar.update()



n_epoch = 50

for epoch in range(1, n_epoch + 1):
    train(model, epoch, 20)
    test(model, epoch)
    scheduler.step()
save_path = "Weight/Pytorch_Pretrain"
if (os.path.isdir(save_path) == False):
    os.makedirs(save_path)
torch.save(model.state_dict(), "Weight/Pytorch_Pretrain/Conv2D_Voice_Model.pt")