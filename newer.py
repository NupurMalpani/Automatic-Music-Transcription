import os
import glob
import torch
import torch.backends.cudnn as cudnn
import torchaudio
from torch.utils.data.sampler import Sampler
import shlex
from img2arr import give_img_arr
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from os import listdir
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from os.path import isfile, join
import subprocess
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math
from mq import mysimpl
import socket
from other_feat import all_feat
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import librosa
import scipy.signal as signal
import librosa.display as display
random.seed(115)
np.random.seed(115)
torch.manual_seed(115)
torch.cuda.manual_seed(115)
classes = {'Guitar': 1, 'Violin': 0, 'Piano': 2, 'Flute': 3}
revclasses = {1: 'Guitar', 0: 'Violin', 2: 'Piano', 3: 'Flute'}
insize =(480,640)
from cnnmodel import *
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
'''


#
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# s.bind(("127.0.0.1", 10999 ))
# print("socket done")

transform = transforms.Compose(
    [transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
class DriveData(Dataset):
    __xs = []
    __ys = []

    def __init__(self, data, fs, transform=None):
        # super(DriveData, self).__init__()
        self.transform = transform
        self.data = data
        self.__xs = [i[0] for i in data]
        # f = open('f1.txt','a')
        # print(_xs,file=f)
        self.__ys = [i[1] for i in data]
        # print(_ys,file=f)
        # f.close()
        self.fs = fs
        # self.freqs =np.array([(i + 1) * self.fs/2048 for i in range(2048)])

    def give_xs_ys(self):
        return self.__xs, self.__ys

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        # filename = os.path.splitext(self.__xs[index])[0] + ".npy"
        # print("working on ",filename)
        # sound, sample_rate = torchaudio.load(self.__xs[index])
        # signal = sound.numpy()
        # if signal.shape[1] == 2 :
        #     signal = (signal[:,0] + signal[:,1]) / 2
        #     print("stereo to mono")
        # else:
        #     signal = np.ascontiguousarray(np.reshape(signal,(signal.shape[0],)))
        # # print("len",signal.shape)
        # # print(give_num_frames(signal.shape[0]))
        # num_frames = give_num_frames(signal.shape[0])
        # #
        # # np.save(filename,signal)
        # # # print("saving input in file")
        # # call = 'python2 /home/ce21/mag.py ' + filename
        # # subprocess.call(shlex.split(call))
        # # print("simpl done")
        # # features = np.load(filename)
        # # os.remove(filename)
        # frames_features = {}
        # print("frames,",num_frames)
        filename = self.__xs[index]
        print(filename)
        y, sr = get_y_sr(filename)
        # stfts = librosa.stft(y, window=signal.get_window('blackman', 2048))
        # stfts = np.transpose(stfts)
        frames = librosa.util.frame(np.pad(y,1024,mode='reflect'))
        frames = np.transpose(frames)
        sp, c, roll, cr, flux = all_feat(filename)
        num_frames = cr.shape[1]
        print("num of frames",frames.shape[0],num_frames)
        # print(np.count_nonzero(sp.imag),np.count_nonzero(flux.imag))
        features = [0 for x in range(num_frames)]
        # frames_features = {}
        # for frame in range(num_frames):
        #   frames_features[frame] = []
        # for x in features:
        # print(x[0],x[1],x[2]) x[1] = framenumber x[0] amp x[2] freq
        # frames_features[int(x[1])].append((x[0],x[2]))
        frame_freq_bins = []
        for x in range(num_frames):
            img = give_img_arr(frames[x])
            transformed = transform(img)
            # os.remove(imgfile)
            #print("done with frame",x)
            freq_bins = np.zeros(5)
            features[x] = []
            freq_bins[0] = sp[x]
            freq_bins[1] = c[0][x]
            freq_bins[2] = roll[0][x]
            freq_bins[3] = cr[0][x]
            freq_bins[4] = flux[x]
            features[x].append(transformed)
            features[x].append(torch.from_numpy(freq_bins))
        # dict with key as freqbin
        # to_be_added ={}
        # for y in frames_features[x]:
        #     index_i = np.abs(self.freqs-y[1]).argmin();
        #     if(y[1] < self.freqs[index_i]):
        #         index_i -=1;
        #     if index_i not in to_be_added.keys():
        #         to_be_added[index_i] = []
        #     to_be_added[index_i].append(y[0])
        # all_non_zero_bins = to_be_added.keys()
        # for x in all_non_zero_bins:
        #     amp_array =to_be_added[x]
        #     amp_array = np.array(amp_array)
        #     avg_amp = np.mean(amp_array)
        #     freq_bins[x] += avg_amp
        # # freq_bins = torch.LongTensor(freq_bins)
        # frame_freq_bins.append(freq_bins)
        # frame_freq_bins = np.array(frame_freq_bins)
        # frame_freq_bins = torch.from_numpy(frame_freq_bins)
        # print("frame freq bins")
        label_i = self.__ys[index]
        # label = np.zeros(4,dtype='double')
        # label[classes[label_i]] += 1
        label = np.array([classes[label_i]])
        label = torch.from_numpy(label)
        # return features,label
        print("done")
        # print(features,"before")
        return [features, label,num_frames]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)


def give_num_frames(data):
    return 3 + math.ceil((data - 2048) / 512)


# Getting data
# ----------------------------------
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_data(data_dir="/home/ce21/new_dataset"):
    paths = get_immediate_subdirectories(data_dir)
    data = []
    for x in paths:
        nowpath = data_dir + '/' + x + '/'
        files = [join(nowpath, f) for f in listdir(nowpath) if isfile(join(nowpath, f))]
        data.append((files, x))
    return data


def get_y_sr(filename):
    y, sr = torchaudio.load(filename)
    if (y.shape[1] > 1):
        d = y.shape[1]
        y = y.numpy()
        y = np.sum(y, axis=1) / d
    else:
        y = y.view(-1, )
        y = y.numpy()
    return y, sr


#
# class BucketingSampler(Sampler):
#     def __init__(self, data_source, batch_size=1):
#         """
#         Samples batches assuming they are in order of size to batch similarly sized samples together.
#         """
#         super(BucketingSampler, self).__init__(data_source)
#         self.data_source = data_source
#         ids = list(range(0, len(data_source)))
#         self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
#
#     def __iter__(self):
#         for ids in self.bins:
#             np.random.shuffle(ids)
#             yield ids
#
#     def __len__(self):
#         return len(self.bins)
#
#     def shuffle(self):
#         np.random.shuffle(self.bins)
'''
def my_collate(batch):
    def func(p):
        return p[0].size(1)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow([1, 0, seq_length]).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.LongTensor(targets)
    return inputs, targets, input_percentages, target_sizes
'''


def mycollate(batch):
    features = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    frames = [x[2] for x in batch]
    lengths = [len(x) for x in features]
    # print(features,labels,lengths)
    nl = np.array(lengths)
    inds = nl.argsort()
    nf = features
    features = []
    for i in range(len(nf)):
        features.append(nf[inds[i]])
    nl = labels
    labels = []
    for i in range(len(nl)):
        labels.append(nl[inds[i]])
    nf = frames
    frames = []
    for i in range(len(nf)):
        frames.append(nf[inds[i]])
    # labels = [x for _,x in sorted(zip(lengths,labels))]
    # features = [x for _,x in sorted(zip(lengths,features))]
    lengths.sort(reverse=True)
    labels.reverse()
    features.reverse()
    frames.reverse()
    frames = torch.Tensor(frames)
    maxlen = max(lengths)
    print("max length of batch", maxlen)
    lengths = np.array(lengths)
    diff = maxlen - lengths
    diff = torch.from_numpy(diff)
    # new_features = torch.zeros(, maxlen, 2))
    # for i, x in enumerate(features):
    #     to_cat = np.zeros((diff[i], 2053))
    #     to_cat = torch.from_numpy(to_cat)
    #     new_features[i] = torch.cat([x, to_cat])
    #     print(new_features[i].shape, "new length", i)
    # dummy = torch.zeros(620,480)
    norm_features = [[] for x in range(len(batch))]
    image_tensors = [[] for x in range(len(batch))]
    # print(features)
    #print(diff)
    for i , x in enumerate(features):
        myimgx = torch.stack([y[0] for y in x],0)
        mynormx =torch.stack([y[1] for y in x],0)
        extra = int(diff[i])
        to_cat = torch.zeros(extra,3,480,640)
        # print(x[0].type())
        #print(myimgx.size(),to_cat.size())
        image_tensors[i] = torch.cat((myimgx ,to_cat),0)
        norm_features[i] = torch.cat((mynormx.float() ,torch.zeros(extra,5)),0)
    image_tensors = torch.stack(image_tensors,0)
    norm_features = torch.stack(norm_features,0)
    # features = [x for _,x in sorted(zip(lengths,features))]
    # lengths.sort()
    # features.reverse()
    # lengths.reverse()
    # print(lengths,"length")
    # print(features,"features")
    # print(np.array(features))
    # print(torch.from_numpy(np.array(features)))
    # print(Variable(torch.from_numpy(np.array(features))))
    #print(norm_features.shape, "new features shape")
    image_tensors = pack_padded_sequence(Variable(image_tensors), lengths, batch_first=True)
    norm_features = pack_padded_sequence(Variable(norm_features),lengths,batch_first=True)
    new_l = torch.LongTensor(batch_size, 1)
    torch.cat(labels, out=new_l)
    # new_l= new_l.view(batch_size,4)
    return [image_tensors,norm_features, new_l,frames]


def random_permutate(data, train_percent=0.7, percent=1.0):
    rptrain_data = []
    rptest_data = []
    for x in data:
        # type = x[1]
        num_of_files = len(x[0])
        z = np.random.permutation(np.arange(num_of_files))
        percent_limit = int((len(z) * percent))
        z = z[:percent_limit]
        #         print(len(z))
        limit = int((len(z) * train_percent))
        train = z[:limit]
        test = z[limit:]
        if len(train) % 2 != 0:
            curr = len(train)
            train = train[:-(curr % 2)]
        if len(test) % 2 != 0:
            curr = len(test)
            test = test[:-(curr % 2)]
        for i in train:
            rptrain_data.append((x[0][i], x[1]))
        for i in test:
            rptest_data.append((x[0][i], x[1]))
    random.shuffle(rptrain_data)
    random.shuffle(rptest_data)
    return rptrain_data, rptest_data


dataset = get_data()
train_data, test_data = random_permutate(dataset)
# f = open('f.txt','w')
# print(train_data,file=f)
# f.close()
train_dataset = DriveData(train_data, 44100)
print(len(train_dataset.data))
test_dataset = DriveData(test_data, 44100)
train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=mycollate, num_workers=1, batch_size=1,
                          pin_memory=True)
# print(train_loader)
test_loader = DataLoader(test_dataset, shuffle=True, collate_fn=mycollate, pin_memory=True, batch_size=2, num_workers=2)

n = 55
hidden_size = 20
num_layers = 10
batch_size = 1
dense1_o = 5


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn1 = CNN(insize)
        self.rnn1 = nn.GRU(input_size=n,
                           hidden_size=hidden_size,
                           num_layers=num_layers)
        self.dense1 = nn.Linear(hidden_size, dense1_o, bias=True)
        self.dense2 = nn.Linear(dense1_o, 4, bias=True)
        self.layers = nn.ModuleList([self.cnn1,self.rnn1,self.dense1,self.dense2])

    def forward(self, image,feat, diff, hidden,frames):
        # self.hidden = hidden
        finals = []
        self.output = []
        self.input = []
        frame = int(frames.data[0])
        #print(feat.size())
        self.all_cnns = []
        for i in range(batch_size):
            #final = torch.Tensor([]).cuda()
            final = []
            for x in range(frame):
                out = self.cnn1(image[x])
                out= out.cpu()
                out.retain_grad()
                self.all_cnns.append(out)
                # self.save_for_backward(out)
                #out.requires_grad = True
                #print(feat[x].size())
                # print("frame",x , out.requires_grad)
                torch.cuda.empty_cache()
                out2 = feat[x]
                # out2.requires_grad = True
                #print(out.requires_grad,out2.requires_grad)
                curr = torch.cat((out.data,out2.data),1)
                print("frame",x)
                final.append(curr)
            finals.append(torch.cat(final))
            # self.all_cnns.append(cnn_outputs)
        self.output.append(self.all_cnns)
        final = torch.stack(finals,1)
        print(final.size())
        final = final.cuda()
        print(final.requires_grad)
        final.requires_grad = True
#print(hidden)
        #final = Variable(final.cuda())
        #print(hidden,final)
        self.input.append(final)
        x, hidden = self.rnn1(final, hidden)
        torch.cuda.empty_cache()
        # print(x,"prev")
        # self.hidden = Variable(hidden.data,requires_grad= True)
        # x = x.data
        # print(x)
        mylist = []
        for i in range(batch_size):
            curr = x[-1 + diff[i]][i][:].view(-1, hidden_size)
            # curr = curr.view(1,hidden_size)
            mylist.append(curr)
            # print(curr.size())
        newx = torch.cat(tuple(mylist), 0)
        self.output.append(newx)
        # for i in range(4):
        # x[]
        # print(diff,newx,hidden)
        # # print(x,"after")
        # x = x.contiguous().view(-1, hidden_size)
        # # print(x,"after")
        # x = Variable(x,requires_grad= True)
        newx = Variable(newx.data,requires_grad = True)
        self.input(newx)
        x = self.dense1(newx)
        self.output.append(x)
        #first = nn.BatchNorm1d(dense1_o)
        #x = first(x.cpu()).cuda()
        # x.requires_grad = True
        x = Variable(x.data,requires_grad = True)
        self.input.append(x)
        x = self.dense2(x)
        self.output.append(x)
        #second = nn.BatchNorm1d(4)
        #x = second(x.cpu()).cuda()
        # x.requires_grad = True
        x = Variable(x.data,requires_grad = True)
        self.input(x)
        x = F.softmax(F.sigmoid(x))
        self.output(x)
        return x, hidden

    def init_hidden(self, mylength):
        weight = next(self.parameters()).data
        hidden = Variable(weight.new(num_layers, batch_size, hidden_size).zero_().cuda())
        return hidden
    def backward(self,g):
        self.input.reverse()
        for i,output in reversed(list(enumerate(self.output[:-1]))):
            if i == 0:
                output.backward(g)
            else:
                output.backward(self.input[i - 1].grad.data)
                print(i,self.input[i-1].grad.data,"Putting calculated gradient")
        final_grads = self.input[-1].grad.data
        cnn_grads = final_grads[:,:,:-5].view(-1,50).contiguous()
        cnns = self.output[-1]
        cnn_optim = optim.Adam(model.parameters(), lr= 0.001)
        for y in range(len(cnns)):
            cnn_optim.zero_grad()
            cnns[y].backward(cnn_grads[y])
            cnn_optim.step()

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def training(epochs, net, loader):
    torch.cuda.empty_cache()
    print("Number of batches to go", len(loader))
    for epoch in range(epochs):
        running_loss = 0.0
        print(epoch)
        net.train()
        criterion = nn.NLLLoss().cuda()
        for i, data in enumerate(loader):
            print("loop", i + 1)
            imagetensors,feature_tensors,labels,frames = data
            print('in loop')
            # get the inputs
            # wrap them in Variable
            # print(inputs)
            # zero the parameter gradients
            # inputs = inputs.cuda()
            # labels = labels.long().cuda()
            output_images, lengths = pad_packed_sequence(imagetensors)
            output_norm_features,lengths = pad_packed_sequence(feature_tensors)
            del imagetensors,feature_tensors
            maxlength = max(lengths)
            diff = [j - maxlength for j in lengths]
            print("before net")
            if (i == 0 and epoch == 0):
                hidden = net.init_hidden(maxlength)
            # else:
            # hidden = net.hidden
            # print(hidden)
            # diff = Variable(diff).cuda()
            # forward + backward + optimize
            hidden = repackage_hidden(hidden)
            # net.zero_grad()
            outputs, hidden = net(output_images,output_norm_features, diff, hidden,Variable(frames))
            # _,mylabels = torch.max(outputs,1)
            # mylabels = mylabels.view(batch_size,1)
            # labels = labels.view(batch_size,1)
            # print(outputs)
            # _, predicted = torch.max(outputs, 1)
            print("output done---------------------------------------------------------")
            # print(labels.cpu(),"labels")
            # print(outputs.cpu(),"outputs")
            # print(mylabels.cpu())
            labels = Variable(labels.long()).cuda()
            print(outputs, labels)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            del outputs
            del labels
            # loss = loss
            print(loss.data[0])
            # loss = loss.cpu()
            loss.backward()
            for param in net.parameters():
                if param.grad is not None:
                    print(param.grad.data.sum())
                else:
                    print("Its None")
            optimizer.step()
            # del hidden
            # print statistics
            torch.cuda.synchronize()
            running_loss += loss.data[0]
            print("BATCH", i + 1, "Done-------------------------------------------------------")
            if i % 2 == 1:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
                print("in loop")
            # hidden = Variable(old_hidden.data,volatile = True)
            net.train()
            # net.hidden = Variable(net.hidden.data,requires_grad = True)
            torch.cuda.empty_cache()
        print('getting out of epoch')
        del loss
        torch.cuda.empty_cache()
        if epoch != epochs - 1:
            del net
            net = Net().cuda()
        else:
            return net, hidden
def testing(net, loader, hidden):
    correct = 0
    total = 0
    class_correct = [0.0 for i in range(4)]
    class_total = [0.0 for i in range(4)]
    net.eval()
    for i, data in enumerate(loader):
        print("batch", i + 1)
        inputs, labels = data
        labels = labels.long().cuda()
        output, lengths = pad_packed_sequence(inputs)
        del inputs
        output = output.cuda()
        maxlength = max(lengths)
        diff = [j - maxlength for j in lengths]
        outputs, hidden = net(output, diff, hidden)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        c = (predicted == labels).squeeze()
        torch.cuda.synchronize()
        for i in range(2):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
        del labels
        hidden = repackage_hidden(hidden)
        # net.hidden = Variable(net.hidden.data,requires_grad = True)
    print("accuracy is ", 100 * correct / total)
    for i in range(4):
        print("Accuracy of %s:%f %%" % (revclasses[i], 100 * class_correct[i] / class_total[i]))
        # ef backward(self,g):

# print('Finished Training')

# libs = cudnn._libcudnn()
# print(libs)
model = Net()
model.cuda()
lr = 0.0001
# output, hidden = model(X_train_1, Variable(hidden.data))

# print(criterion)
optimizer = optim.Adam(model.parameters(), lr=lr)
# myhidden = model.init_hidden()
torch.cuda.empty_cache()
mynet, myhidden = training(1, model, train_loader)
# torch.save(mynet.state_dict(),'/home/ce21/mymodel.pth.tar')
# newnet = Net()
# mynet_state_dict = torch.load('/home/ce21/mymodel.pth.tar')
# newnet.load_state_dict(mynet_state_dict)
# testing(newnet,test_loader)
testing(mynet, test_loader, myhidden)
# f = open('f.txt','w')
# print(train_loader.__sizeof__())
