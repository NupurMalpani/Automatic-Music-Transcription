from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def one_layer(insize,kernel,m):
    w = insize[0] - kernel  + 1
    h = insize[1] - kernel + 1
    final_w = w // m
    if w % m :
        final_w += 1
    final_h = h // m
    if h % m:
        final_h += 1
    return final_w,final_h

class CNN(nn.Module):
    def __init__(self,insize):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512,3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2))
#         self.fc = nn.Linear(128*32*32, 8)
        # self.fc = nn.Linear(256*16*16, 8)
        w,h = one_layer(insize,3,2)
        for x in range(5):
            w,h = one_layer((w,h),3,2)
        self.fc = nn.Linear(512*w*h, 50)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



