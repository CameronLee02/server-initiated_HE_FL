import copy
import torch

def FedAvg(weight_list):
    num = len(weight_list)
    w_avg = copy.deepcopy(weight_list[0])
    for k in w_avg.keys():
        for i in range(1, num):
            w_avg[k] += weight_list[i][k]
        w_avg[k] = torch.div(w_avg[k], num)
    return w_avg