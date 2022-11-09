#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 06:00:09 2021
using example: model = EmbeddingNet(in_sz=1000,
                     out_sz=32,
                     emb_szs=[256],projection=False)
@author: xiaokangyu
"""
import torch
import torch.nn as nn
import numpy as np
import sys

class EmbeddingNet(nn.Module):
    def __init__(self, in_sz=1000, out_sz=32, emb_szs=[256],projection=False,project_dim=2,use_dropout=False,dp_list=None,use_bn=False, actn=nn.ReLU()):
        super(EmbeddingNet, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.emb_szs=emb_szs
        self.projection=projection
        self.project_dim=project_dim
        self.use_dropout=use_dropout
        self.dp_list=dp_list
        self.use_bn=use_bn
        self.actn=actn
        if self.projection:
            #self.emb_szs.append(out_sz)
            self.out_sz=self.project_dim
        self.n_embs = len(self.emb_szs) - 1  
        if self.use_dropout:
            if(self.dp_list is None):
                sys.exit("Error: can't find dropout value for Dropout Layers, please provide a list of dropout value if you want to use Dropout!!")
            else:
                ps=self.dp_list
        else:
            ps = np.zeros(self.n_embs)#
        # input layer
        layers = [nn.Linear(self.in_sz, self.emb_szs[0]),
                  self.actn]#
        # hidden layers
        for i in range(self.n_embs):#
            layers += self.bn_drop_lin(n_in=self.emb_szs[i], n_out=self.emb_szs[i+1], bn=self.use_bn, p=ps[i], actn=self.actn)
        # output layer
        layers.append(nn.Linear(self.emb_szs[-1], self.out_sz))#
        self.fc = nn.Sequential(*layers)#
        
    def bn_drop_lin(self, n_in:int, n_out:int, bn:bool=True, p:float=0., actn:nn.Module=None):
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else [] 
        if p != 0: layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None: layers.append(actn)
        return layers
              
    def forward(self, x):
        output = self.fc(x)
        return output
