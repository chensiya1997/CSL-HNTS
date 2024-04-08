# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy.random
import torch
import torch.nn as nn
import logging
import os
import argparse
import math
import random
import tqdm
import scipy.sparse as ss
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch.optim as optim
import torch.utils as utils
import dataloader,utility,earlystopping
from model import models
from matrix_copy import matrix_trans


def set_env(seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=50, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--n_his', type=int, default=6)
    parser.add_argument('--n_pred', type=int, default=1,
                        help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=1)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv',
                        choices=['cheb_graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap',
                        choices=['sym_norm_lap'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    args = parser.parse_args()
    set_env(args.seed)
    if args.enable_cuda and torch.cuda.is_available():

        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([16, 16, 16])
    if Ko == 0:
        blocks.append([16])
    elif Ko > 0:
        blocks.append([16, 16])
    blocks.append([1])
    return args, device, blocks


def data_preparate(args,x,input_dim,vadj,device):
    n_vertex = input_dim
    a = torch.zeros((vadj.shape[0], vadj.shape[1]))
    max_value=torch.abs(vadj).max()
    for i in range(vadj.shape[0]):
        for j in range(vadj.shape[1]):
            a[i][j] =  math.fabs(vadj[i][j]) / max_value
    adj = ss.csc_matrix(vadj.detach().numpy())

    gso = utility.calc_gso(adj, args.gso_type)

    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    train=x



    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)

    return n_vertex, x_train,y_train


def prepare_model(args, blocks, n_vertex,device):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    if args.graph_conv_type == 'cheb_graph_conv':
        modelST = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)


    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(modelST.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, modelST, optimizer, scheduler


class STGCNEncoder(nn.Module):
    def __init__(self, device) -> None:
        super(STGCNEncoder, self).__init__()
        self.device = device


    def forward(self, x,adj) -> torch.Tensor:

        args, device, blocks = get_parameters()
        n_vertex, x_train, y_train= data_preparate(args, x.view(len(x), x.shape[1]).detach().numpy(), x.shape[1],adj,device)
        loss, es, modelST2, optimizer, scheduler = prepare_model(args, blocks, n_vertex, self.device)
        modelST2.train()
        y_pred = modelST2(x_train).view(len(x_train), -1)

        return y_pred.reshape(y_pred.shape[0], y_pred.shape[1], 1)



class MLP(nn.Module):
    """
    Feed-forward neural networks----MLP

    """

    def __init__(self, input_dim, layers, units, output_dim,
                 activation=None, device=None) -> None:
        super(MLP, self).__init__()
        # self.desc = desc
        self.input_dim = input_dim
        self.layers = layers
        self.units = units
        self.output_dim = output_dim
        self.activation = activation
        self.device = device

        mlp = []
        for i in range(layers):
            input_size = units
            if i == 0:
                input_size = input_dim
            weight = nn.Linear(in_features=input_size,
                               out_features=self.units,
                               bias=True,
                               device=self.device)
            mlp.append(weight)
            if activation is not None:
                mlp.append(activation)
        out_layer = nn.Linear(in_features=self.units,
                              out_features=self.output_dim,
                              bias=True,
                              device=self.device)
        mlp.append(out_layer)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x) -> torch.Tensor:

        x_ = x.reshape(-1, self.input_dim)
        output = self.mlp(x_)

        return output.reshape(x.shape[0], -1, self.output_dim)


class AutoEncoder(nn.Module):

    def __init__(self, d, timestep,input_dim, hidden_layers=3, hidden_dim=989,
                 activation=nn.ReLU(), device=None):
        super(AutoEncoder, self).__init__()
        self.d = d
        self.timestep=timestep
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.device = device
        self.encoder=STGCNEncoder(
            device=self.device,
        )


        self.decoder = MLP(input_dim=self.input_dim,
                           layers=self.hidden_layers,
                           units=self.hidden_dim,
                           output_dim=self.input_dim,
                           activation=self.activation,
                           device=self.device)

        w = torch.nn.init.uniform_(torch.empty(self.d, self.d),
                                   a=-0.1, b=0.1)     #初始化神经网络参数
        w_det=torch.nn.init.uniform_(torch.empty(self.d),
                                   a=-0.1, b=0.1)
        self.w = torch.nn.Parameter(w.to(device=self.device))
        self.w_det=torch.nn.Parameter(w_det.to(device=self.device))



    def forward(self, x):
        self.w_adj=self.w
        self.w_det=self.w_det
        adj = matrix_trans(self.w_adj, self.w_det.detach().numpy())
        unit_angle=torch.triu(torch.ones(self.w_adj.shape[0],self.w_adj.shape[1]))
        change = torch.mul(self.w_adj, self.w_det)
        change=torch.mul(change,unit_angle)
        change=matrix_trans(change, self.w_det.detach().numpy())
        x_est = self.decoder(change)
        mse_loss = torch.square(torch.norm(x[x.shape[0] - x_est.shape[0]:, :, :] - x_est, p=2))

        return mse_loss, self.w_adj,change

