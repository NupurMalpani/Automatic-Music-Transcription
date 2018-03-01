from copy import deepcopy
import  os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import librosa
from for_others import run as runothers
import socket
from for_others import init_processes as init_p


#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#s.bind(("127.0.0.1",10999))


def init_processes(port,addrs,rank,size,fn,scatterlist,sr,backend='tcp'):
    init_addr =  backend + "://" + addrs +":" + str(port)
    dist.init_process_group(backend=backend,init_method=init_addr, rank=rank, world_size=size, group_name="mygroup")
    # for rank in range(1,size):
    #     init_p(port,addrs[rank],rank,size)
    output = fn(size,scatterlist,sr)

def run(number,scatter_list,sr):
    ranks =list(range(number))
    lengths = []
    examples = len(scatter_list)
    for a in scatter_list:
        lengths = lengths + list(a.size())
    maxlen = max(lengths)
    for a in range(len(lengths)):
        zero = torch.zeros(maxlen - lengths[a])
        scatter_list[a] = torch.cat((scatter_list[a],zero,sr),0)
    # ind = lengths.index(maxlen)
    win_length = 2048
    group = dist.new_group(ranks)
    src = dst =0
    scatter_op =dist.scatter(torch.zeros(lengths),scatter_list=scatter_list,group = group)
    #assume scatter_o_p is a list
    while True:
        all_done = True
        for i in scatter_op:
            if i.is_completed == False:
                all_done= False
                break;
        if(all_done):
            break;
    frames_size = librosa.util.frame(scatter_list[0]).shape[0]
    gather_list_ele = torch.zeros(frames_size,win_length)
    gather_list = []
    for i in range(len(scatter_list)):
        gather_list.append(deepcopy(gather_list_ele))
    dist.gather(gather_list=gather_list,dst=dst,group=group)
    return gather_list

def init_main(scatter_list,sr):
    size = 10;
    processes = []
    port = 29500
    # output = init_processes(port,'127.0.0.1',0,size,run,scatter_list,sr)
    p = Process(target=init_processes, args=(port,'127.0.0.1',0, size, run,scatter_list,sr))
    # init_processes(port,'127.0.0.1',0,size,run,scatter_list,sr)
    p.start()
    # output = out_q.get()
    # return output
