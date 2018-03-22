import os
import glob
import torch
import torch.backends.cudnn as cudnn
import torchaudio
from torch.utils.data.sampler import Sampler
import shlex
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from os import listdir
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
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
random.seed(499)
np.random.seed(499)
torch.manual_seed(499)
torch.cuda.manual_seed(499)
classes ={'Guitar':1,'Violin':0,'Piano':2,'Flute':3}
revclasses ={1:'Guitar',0:'Violin',2:'Piano',3:'Flute'}
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


class DriveData(Dataset):
    __xs = []
    __ys = []
    def __init__(self, data,fs,transform=None):
        # super(DriveData, self).__init__()
        self.transform = transform
        self.data = data
        self.__xs = [i[0] for i in data]
        # f = open('f1.txt','a')
        # print(_xs,file=f)
        self.__ys =[i[1] for i in data]
        # print(_ys,file=f)
        # f.close()
        self.fs = fs
        self.freqs =np.array([(i + 1) * self.fs/2048 for i in range(2048)])

    def give_xs_ys(self):
        return self.__xs,self.__ys
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
        features,num_frames = mysimpl(filename)
        print(num_frames)
        sp,c,roll,cr,flux = all_feat(filename)
        num_frames =  cr.shape[1]
        print(np.count_nonzero(sp.imag),np.count_nonzero(flux.imag))
        frames_features = {}
        for frame in range(num_frames):
            frames_features[frame] = []
        # for x in features:
            # print(x[0],x[1],x[2]) x[1] = framenumber x[0] amp x[2] freq
            # frames_features[int(x[1])].append((x[0],x[2]))
        frame_freq_bins =[]
        for x in range(num_frames):
            freq_bins = np.zeros(2053)
            freq_bins[2048] = sp[x]
            freq_bins[2049] = c[0][x]
            freq_bins[2050] = roll[0][x]
            freq_bins[2051] = cr[0][x]
            freq_bins[2052] = flux[x]
            #dict with key as freqbin
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
            frame_freq_bins.append(freq_bins)
            
        frame_freq_bins = np.array(frame_freq_bins)
        frame_freq_bins = torch.from_numpy(frame_freq_bins)
        print("frame freq bins")
        label_i = self.__ys[index]
        # label = np.zeros(4,dtype='double')
        # label[classes[label_i]] += 1
        label = np.array([classes[label_i]])
        label = torch.from_numpy(label)
        #return features,label
        print("done")
        return [frame_freq_bins,label]
    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)
def give_num_frames(data):
    return 3 + math.ceil((data - 2048)/512)

#Getting data
#----------------------------------
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_data(data_dir = "/home/ce21/dataset"):
    paths = get_immediate_subdirectories(data_dir)
    data =[]
    for x in paths:
        nowpath = data_dir + '/' + x + '/'
        files = [join(nowpath,f) for f in listdir(nowpath) if isfile(join(nowpath, f))]
        data.append((files,x))
    return data
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
    # labels = [x for _,x in sorted(zip(lengths,labels))]
    # features = [x for _,x in sorted(zip(lengths,features))]
    lengths.sort(reverse=True)
    labels.reverse()
    features.reverse()
    maxlen = max(lengths)
    print("max length of batch",maxlen)
    lengths = np.array(lengths)
    diff = maxlen - lengths
    new_features = torch.zeros((lengths.shape[0],maxlen,2053))
    for i,x in enumerate(features):
        to_cat = np.zeros((diff[i],2053))
        to_cat = torch.from_numpy(to_cat)
        new_features[i] = torch.cat([x,to_cat])
        print(new_features[i].shape,"new length",i)
    # features = [x for _,x in sorted(zip(lengths,features))]
    # lengths.sort()
    # features.reverse()
    # lengths.reverse()
    # print(lengths,"length")
    # print(features,"features")
    # print(np.array(features))
    # print(torch.from_numpy(np.array(features)))
    # print(Variable(torch.from_numpy(np.array(features))))
    print(new_features.shape,"new features shape")
    inputs = pack_padded_sequence(Variable(new_features), lengths, batch_first=True)
    new_l =  torch.LongTensor(batch_size,1)
    torch.cat(labels,out=new_l)
    # new_l= new_l.view(batch_size,4)
    return [inputs,new_l]
def random_permutate(data,train_percent=0.7,percent=1.0):
    rptrain_data =[]
    rptest_data = []
    for x in data:
        # type = x[1]
        num_of_files = len(x[0])
        z = np.random.permutation(np.arange(num_of_files))
        percent_limit = int((len(z) * percent))
        z = z[:percent_limit]
#         print(len(z))
        limit = int((len(z) * train_percent))
        train =z[:limit]
        test = z[limit:]
        if len(train) % 2 != 0:
            curr = len(train)
            train = train[:-(curr%2)]
        if len(test) % 2 != 0:
            curr = len(test)
            test = test[:-(curr % 2)]
        for i in train:
            rptrain_data.append((x[0][i],x[1]))
        for i in test:
            rptest_data.append((x[0][i],x[1]))
    random.shuffle(rptrain_data)
    random.shuffle(rptest_data)
    return rptrain_data,rptest_data

dataset = get_data()
train_data,test_data = random_permutate(dataset)
# f = open('f.txt','w')
# print(train_data,file=f)
# f.close()
train_dataset = DriveData(train_data,44100)
print(len(train_dataset.data))
test_dataset = DriveData(test_data,44100)
train_loader = DataLoader(train_dataset,shuffle=True,collate_fn=mycollate, num_workers=2,batch_size=2,pin_memory=True)
# print(train_loader)
test_loader = DataLoader(test_dataset,shuffle=True,collate_fn=mycollate,pin_memory=True, batch_size=2, num_workers=2)

n = 2053
hidden_size = 75
num_layers =2
batch_size =2
dense1_o =5
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn1 = nn.GRU(input_size=n,
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        self.dense1 = nn.Linear(hidden_size,dense1_o,bias=True)
        self.dense2 = nn.Linear(dense1_o,4,bias=True)
    def forward(self,x,diff,hidden):
        # self.hidden = hidden
        x, hidden = self.rnn1(x.cuda(),hidden)
        torch.cuda.empty_cache()
        # print(x,"prev")
        # self.hidden = Variable(hidden.data,requires_grad= True)
        # x = x.data
        # print(x)
        mylist = []
        for i in range(2):
            curr = x[-1 + diff[i]][i][:].view(-1,hidden_size)
            # curr = curr.view(1,hidden_size)
            mylist.append(curr)
            # print(curr.size())
        newx = torch.cat(tuple(mylist),0)
        # for i in range(4):
            # x[]
        # print(diff,newx,hidden)
        # # print(x,"after")
        # x = x.contiguous().view(-1, hidden_size)
        # # print(x,"after")
        # x = Variable(x,requires_grad= True)
        x = self.dense1(newx)
        first =  nn.BatchNorm1d(dense1_o)
        x = first(x.cpu()).cuda()
        # x.requires_grad = True
        x = self.dense2(x)
        second = nn.BatchNorm1d(4)
        x= second(x.cpu()).cuda()
        # x.requires_grad = True
        return F.softmax(F.sigmoid(x)),hidden
    def init_hidden(self,mylength):
        weight = next(self.parameters()).data
        hidden =  Variable(weight.new(num_layers, batch_size, hidden_size).zero_().cuda())
        return hidden
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
def training(epochs,net,loader):
        torch.cuda.empty_cache()
        print("Number of batches to go",len(loader))
        for epoch in range(epochs):
            running_loss = 0.0
            print(epoch)
            net.train()
            criterion = nn.NLLLoss().cuda()
            for i,data in enumerate(loader):
                print("loop",i+1)
                inputs,labels = data
                print('in loop')
                # get the inputs
                # wrap them in Variable
                # print(inputs)
                # zero the parameter gradients
                # inputs = inputs.cuda()
                # labels = labels.long().cuda()
                output,lengths = pad_packed_sequence(inputs)
                del inputs
                maxlength = max(lengths)
                diff = [ j - maxlength for j in lengths]
                print("before net")
                if(i==0 and epoch == 0):
                    hidden = net.init_hidden(maxlength)
                # else:
                    # hidden = net.hidden
                # print(hidden)
                # diff = Variable(diff).cuda()
                # forward + backward + optimize
                hidden = repackage_hidden(hidden)
                # net.zero_grad()
                outputs,hidden = net(output,diff,hidden)
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
                print(outputs,labels)
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
                print("BATCH",i+1,"Done-------------------------------------------------------")
                if i % 2 == 1:    # print every 10 mini-batches
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
                return net,hidden

def testing(net,loader,hidden):
    correct = 0
    total = 0
    class_correct = [0.0 for i in range(4)]
    class_total = [0.0 for i in range(4)]
    net.eval()
    for i,data in enumerate(loader):
        print("batch",i + 1)
        inputs,labels = data
        labels = labels.long().cuda()
        output,lengths = pad_packed_sequence(inputs)
        del inputs
        output = output.cuda()
        maxlength = max(lengths)
        diff = [ j - maxlength for j in lengths]
        outputs,hidden= net(output,diff,hidden)
        _,predicted = torch.max(outputs.data,1)
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
    print("accuracy is ",100* correct /total)
    for i in range(4):
        print("Accuracy of %s:%f %%" %(revclasses[i],100 * class_correct[i] / class_total[i]))
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
mynet,myhidden = training(1,model,train_loader)
# torch.save(mynet.state_dict(),'/home/ce21/mymodel.pth.tar')
# newnet = Net()
# mynet_state_dict = torch.load('/home/ce21/mymodel.pth.tar')
# newnet.load_state_dict(mynet_state_dict)
# testing(newnet,test_loader)
testing(mynet,test_loader,myhidden)
# f = open('f.txt','w')
# print(train_loader.__sizeof__())
