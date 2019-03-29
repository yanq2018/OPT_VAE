# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:25:55 2019

@author: Qing Yan
"""

import torch
from torch import optim
from torchvision import transforms, datasets
from model import VAE
from opt_model import OPT_VAE
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import pow
import numpy as np

import argparse

parser = argparse.ArgumentParser(
    description='OPT-VAE on MNIST'
)

parser.add_argument('--nepoch', type=int, default=50, help='number of training epochs')
#parser.add_argument('--lamda',type=float,default=1,help='balancing parameter in front of classification loss')
parser.add_argument('--gpu', default=True, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed (default: 123)')

args = parser.parse_args()

use_gpu = args.gpu and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if use_gpu else "cpu")

'''Parameters, adjust here'''
mb_size = 200 # batch size
h = 28
w = 28
x_dim = h*w
epochs = args.nepoch
log_interval = 100 # for reporting


kwargs = {'num_workers': 4, 'pin_memory': True} if use_gpu else {}
# add 'download=True' when use it for the first time
mnist_tr = datasets.MNIST(root='../MNIST/', download=True, transform=transforms.ToTensor())
mnist_te = datasets.MNIST(root='../MNIST/', download=True, train=False, transform=transforms.ToTensor())
tr = torch.utils.data.DataLoader(dataset=mnist_tr,
                                batch_size=mb_size,
                                shuffle=False,
                                drop_last=True, **kwargs)
te = torch.utils.data.DataLoader(dataset=mnist_te,
                                batch_size=1,
                                shuffle=False,
                                drop_last=True, **kwargs)
def loss_V(recon_x, x, mu, std):
    '''loss = reconstruction loss + KL_z + KL_u'''
    BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, x_dim), x.view(-1, x_dim), size_average=False)
    KLD = -0.5 * torch.sum(1 + 2*torch.log(std) - mu**2 - std**2) # z
    return BCE, KLD

def train_vae():
    model = VAE(h,w,256,50,device)
    optimizer = optim.Adadelta(model.parameters())
    l2 = lambda epoch: pow((1.-1.*epoch/epochs),0.9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l2)
    for epoch in range(epochs):
        scheduler.step()
        model.train()
        tr_recon_loss = 0
        for batch_idx, (data, target) in enumerate(tr):
            data = data.to(device)
            optimizer.zero_grad()
    
            recon_batch, zmu, zvar,_= model(data)
            recon_loss, kl = loss_V(recon_batch, data, zmu,torch.exp(0.5*zvar))
        
            loss = recon_loss + kl 
            loss.backward()
            tr_recon_loss += recon_loss.item()
        
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tReconstruction-Loss: {:.4f}, KL: {:.4f}'.format(
                        epoch, batch_idx * len(data), len(mnist_tr),
                        100. * batch_idx / len(tr),
                        recon_loss / len(data),kl/len(data)))

        print('====> Epoch: {} Reconstruction loss: {:.4f}'.format(
                epoch, tr_recon_loss / (len(tr)*mb_size)))
        test(epoch,model)
    return model

def NLL_test_loss(recon_x, x, mu, var,z):
    #NLL just for testing 
    log_PxGh = -torch.sum(F.binary_cross_entropy(recon_x.squeeze().view(-1, x_dim), x.view(-1,x_dim), reduction='none'), -1)
    log_QhGx = torch.sum(-0.5*((z-mu)/var)**2 - torch.log(var), -1)  
    #log_QuGx =  torch.sum(-0.5*((u-mu2)/var2)**2 - torch.log(var2), -1) + torch.sum(-0.5*((glb['u']-glb['mu'])/glb['var'])**2 - torch.log(glb['var']), -1)
    log_Ph = torch.sum(-0.5*z**2, -1)
    #log_Pu = torch.sum(-0.5*u**2, -1) + torch.sum(-0.5*glb['u']**2, -1)
    log_weight = log_Ph  + log_PxGh - log_QhGx   
    weight = torch.exp(log_weight-torch.min(log_weight))
    NLL = -torch.log(torch.mean(weight,0))-torch.min(log_weight) 
    return NLL

def test(epoch,model):
    model.eval()
    test_recon_loss = 0
    
    with torch.no_grad():
        for _, (data, target) in enumerate(te):
            data = data.to(device)
            #sample 1000 times from posterior to compute NLL (IWAE paper uses 5000)
            data = data.expand(3,1,28,28)
            recon_batch, zmu, zvar,z= model(data)
            NLL = NLL_test_loss(recon_batch, data, zmu, torch.exp(0.5*zvar),z)
            test_recon_loss += NLL.item()
            
    test_recon_loss /= (len(te))
    print('====> Epoch:{} NLL: {:.4f}'.format(epoch, test_recon_loss))

#visualize
'''
for _, (data, target) in enumerate(te):
            data = data.to(device)
            break
'''

'''
Opt
'''
#Ite = 5
#mul = [30,20,10,10,10]
epochs = 100
def train(epoch,img,model,optimizer):
    model.train()
    data = img.to(device)
    optimizer.zero_grad()
    recon,zmu,zvar,z = model(data)
    recon_loss, kl = loss_V(recon, data, zmu,torch.exp(0.5*zvar))
    loss = recon_loss +kl 
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss.item()

def train_opt_vae():
    model = OPT_VAE(h, w, 256, 50, torch.zeros(mb_size,50), 
                    torch.zeros(mb_size,50), device).to(device)
    optimizer1 = optim.Adadelta([model.z_mu,model.z_var])
    optimizer2 = optim.Adadelta(model.parameters())
    zmu_dict = torch.zeros(len(tr),mb_size,50).to(device)
    zvar_dict = torch.zeros(len(tr),mb_size,50).to(device)
    #optimizer = optim.Adam([model.u,model.glbu],lr=0.1)
    #l2 = lambda epoch: pow((1.-1.*epoch/epochs),0.9)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=10000)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10000)
    for epoch in range(epochs):
        loss_tr = 0.0
        for batch_idx, (data, target) in enumerate(tr):
            data = data.to(device)
            model.update_z(zmu_dict[batch_idx,:,:],zvar_dict[batch_idx,:,:])
            for num in range(20):
                scheduler1.step()
                ls = train(epoch,data,model,optimizer1)
            #print('====> Epoch: {} Index {} Ite {} Reconstruction loss (after updating z): {:.4f}'.format(epoch,batch_idx,ite,
            #      tr_recon_loss/data.shape[0]/mul[ite]))
            scheduler2.step()
            ls = train(epoch,data,model,optimizer2)
            loss_tr += ls
            zmu_dict[batch_idx,:,:] = model.z_mu.data
            zvar_dict[batch_idx,:,:] = model.z_var.data
        print('====> Epoch: {} Reconstruction loss: {:.4f}'.format(epoch,loss_tr/len(tr)/mb_size)) 
    return model

model = train_opt_vae()