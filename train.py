import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

labels_dataframe = pd.read_csv('train.csv')
labels_name = sorted(list(set(labels_dataframe['label'])))
n = len(labels_name)
class_num = dict(zip(labels_name, range(n)))
num_class = {v : k for k, v in class_num.items()}

class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256, random_state=42):
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.file_path = file_path
        self.mode = mode

        # 读取csv文件
        self.data_info = pd.read_csv(csv_path)  # 去掉 header=None
        self.data_len = len(self.data_info.index)  # 计算数据长度

        # 根据mode生成对应的数据集
        if mode == 'train' or mode == 'valid':
            # 划分训练集和验证集
            X = self.data_info.iloc[:, 0]  # 第一列是图像文件名
            y = self.data_info.iloc[:, 1]  # 第二列是标签
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=valid_ratio, random_state=random_state)
            self.image_arr = self.X_train if mode == 'train' else self.X_val
            self.label_arr = self.y_train if mode == 'train' else self.y_val

        elif mode == 'test':
            self.test_image = self.data_info.iloc[:, 0]  # 测试集只有图像文件名
            self.image_arr = self.test_image

        # 打印数据集长度
        if mode == 'train':
            print(f'Finished reading the {mode} set of Leaves Dataset ({len(self.X_train)} training samples found)')
            # for i in range(min(10, len(self.X_train))):
            #     print(os.path.join(self.file_path, self.X_train.iloc[i]))
        elif mode == 'valid':
            print(f'Finished reading the {mode} set of Leaves Dataset ({len(self.X_val)} valid samples found)')
        elif mode == 'test':
            print(f'Finished reading the {mode} set of Leaves Dataset ({len(self.test_image)} samples found)')

    def __getitem__(self, index):
        if self.mode == 'train':
            image_name = self.X_train.iloc[index]  # 获取训练集图像文件名
            label = self.y_train.iloc[index] 
            number_labels = class_num[label] # 获取训练集标签
            img = Image.open(os.path.join(self.file_path, image_name))  # 读取图像

            # 训练集数据增强
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img = transform(img)
            return img, torch.tensor(number_labels)

        elif self.mode == 'valid':
            image_name = self.X_val.iloc[index]  # 获取验证集图像文件名
            label = self.y_val.iloc[index]
            number_labels = class_num[label]  # 获取验证集标签
            img = Image.open(os.path.join(self.file_path, image_name))  # 读取图像

            # 验证集不做数据增强
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img = transform(img)
            return img, torch.tensor(number_labels)

        elif self.mode == 'test':
            image_name = self.test_image.iloc[index]  # 获取测试集图像文件名
            img = Image.open(os.path.join(self.file_path, image_name))  # 读取图像

            # 测试集不做数据增强
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img = transform(img)
            return img

    def __len__(self):
        if self.mode == 'train':
            return len(self.X_train)
        elif self.mode == 'valid':
            return len(self.X_val)
        elif self.mode == 'test':
            return len(self.test_image)

train_path = 'train.csv'
test_path = 'test.csv'
img_path = ''
random_state = 42
valid_ratio = 0.2
train_dataset = LeavesData(train_path, img_path, mode='train', valid_ratio = valid_ratio , random_state=random_state)
val_dataset = LeavesData(train_path, img_path, mode='valid', valid_ratio=valid_ratio, random_state=random_state)
test_dataset = LeavesData(test_path, img_path, mode='test')
print(train_dataset)
print(val_dataset)
print(test_dataset)

def load_data(mode="train", batch_size=32):
    if mode == "train":
        return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0))
    elif mode == "test":
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        return None
    
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
        
class ResNet32(nn.Module):
    def __init__(self, num_classes=176):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.blks = nn.ModuleList([])
        block_num = [3, 4, 6, 6]
        for i, num_blocks in enumerate(block_num):
            if i == 0:
                self.blks.append(self._make_layer(64, 64, num_blocks, 1, is_shortcut=False))
            else:
                self.blks.append(self._make_layer(64 * 2**(i-1), 64 * 2**i, num_blocks, 2))
        # self.layer1 = self._make_layer(inchannel=64, outchannel=64, block_num=3, stride=1, is_shortcut=False)
        # self.layer2 = self._make_layer(inchannel=64, outchannel=128, block_num=4, stride=2)
        # self.layer3 = self._make_layer(inchannel=128, outchannel=256, block_num=6, stride=2)
        # self.layer4 = self._make_layer(inchannel=256, outchannel=512, block_num=6, stride=2)
        self.back = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), 
                                  nn.Linear(512, num_classes))

    def _make_layer(self, inchannel, outchannel, block_num, stride, is_shortcut=True):
        if is_shortcut:
            shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        else:
            shortcut = None
        
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for _ in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        net = nn.Sequential(self.pre, *self.blks, self.back)
        return net(x)



# 看一下是在cpu还是GPU上
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
print(device)

model = ResNet32(176)# 176种叶子
model = model.to(device)
model.device = device
print(model)

l = nn.CrossEntropyLoss()
lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
epochs = 10
batch_size = 32
train_data, val_data = load_data("train", batch_size)

best_acc = 0.0

def train(model, l, optimizer, epochs, train_data, val_data, device):
    best_acc = 0.0
    def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    model.apply(init_weight)
    for epoch in range(epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train() 
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        # Iterate the training set by batches.
        for batch in tqdm(train_data):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs)
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = l(logits, labels)
            
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            # Compute the gradients for parameters.
            loss.backward()
            # Update the parameters with computed gradients.
            optimizer.step()
            
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            
        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        
        
        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        
        # Iterate the validation set by batches.
        for batch in tqdm(val_data):
            imgs, labels = batch
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))
                
            # We can still compute the loss (but not the gradient).
            loss = l(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            
        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        # if the model improves, save a checkpoint at this epoch
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), "model_path.pth")
            print('saving model with acc {:.3f}'.format(best_acc))

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, valid_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, valid_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Validation Accuracy')

    plt.show()

train(model, l, optimizer, epochs, train_data, val_data, device)


        print(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        # if the model improves, save a checkpoint at this epoch
        if valid_acc > best_acc:
           asdfjjsaoidfj
           asdhfasdhfui
          a =+
asdffjoasidjfioajs
jerbekacneit

        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []