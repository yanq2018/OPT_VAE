# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:27:00 2019

@author: Qing Yan
"""

import torch
#import math
import torch.nn as nn
import torch.nn.functional as F
#import numpy as np
from torch.autograd import Variable

class OPT_VAE(nn.Module):

    def __init__(self, x_h, x_w, h_dim, z_dim, z_mu, z_var,device):
        super(OPT_VAE, self).__init__()

        self.x_dim = x_h * x_w # height * width
        self.h = x_h
        self.w = x_w
        self.h_dim = h_dim # hidden layer
        self.z_dim = z_dim # generic latent variable
        self.dv = device
        
        self.z_mu = Variable(z_mu,requires_grad=True)
        self.z_var = Variable(z_var,requires_grad=True)
        
        """
        encoder: two fc layers
       
        self.x2h = nn.Linear(self.x_dim, h_dim)
#        self.x2h = nn.Sequential(
#            nn.Linear(self.x_dim, h_dim),
#            nn.ReLU(),
#            nn.Linear(self.h_dim, self.h_dim)
#            )

        self.h2zmu = nn.Linear(h_dim, z_dim)
        self.h2zvar = nn.Linear(h_dim, z_dim)
        """
        """
        decoder: two fc layers
        """
        self.z2h = nn.Linear(z_dim, h_dim)
        self.h2x = nn.Linear(self.h_dim, self.x_dim)
        
    def update_z(self,z_mu,z_var):
        self.z_mu.data = z_mu
        self.z_var.data = z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(self.dv)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = F.relu(self.z2h(z))
        x = F.sigmoid(self.h2x(h))
        return x

    def forward(self, inputs):
        
        #z_mu, z_var = self.encode(inputs.view(-1, self.x_dim))
        z = self.reparameterize(self.z_mu, self.z_var)
        x = self.decode(z)
        x = torch.clamp(x, 1e-6, 1-1e-6)

        return x, self.z_mu, self.z_var,z
