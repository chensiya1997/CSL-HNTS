import torch
import math

def auto_thre(p,w_adj):
    max_value = torch.max(w_adj) * (1 - p)
    step = 0.001
    number = int(max_value / step)
    min = 10000
    final_thre = 0
    for i in range(number):
        cum = 0
        count = 0
        thre = (i + 1) * step
        for j in range(w_adj.shape[0]):
            for k in range(w_adj.shape[1]):
                if w_adj[j][k] > thre:
                    cum = cum + (w_adj[j][k] - thre)
                    count = count + 1
        detection_rate = count / (w_adj.shape[0] * w_adj.shape[1])
        judge = math.fabs(detection_rate - p)+ cum / count
        if judge < min:
            min = judge
            final_thre = thre


    return  final_thre



