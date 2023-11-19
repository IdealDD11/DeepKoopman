# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:27:17 2023

@author: Ideal
"""

from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import array, linalg, transpose, diag, dot, ones, zeros, unique, power, prod, exp, log, divide, real, iscomplex, any, ones_like
from numpy import concatenate as npconcatenate
import numpy as np
from itertools import combinations_with_replacement, permutations
from .utils import differentiate_vec
from .basis_functions import BasisFunctions
from dynamics.linear_system_dynamics import LinearSystemDynamics
from controllers.constant_controller import ConstantController
from .diffeomorphism_net import DiffeomorphismNet
from .coupling_diffeomorphism_net import CouplingDiffeomorphismNet
from .aekoopman import AEKoopman
from .aekoopman import HNNKoopman
import torch
from torch import nn, cuda, optim, from_numpy, manual_seed, no_grad, save, load, cat, transpose as t_transpose
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from torch.autograd.gradcheck import zero_gradients
import gym
from scipy.integrate import odeint
import control


class DeepKoopman(BasisFunctions):
    """
    Class for construction and lifting using Koopman eigenfunctions
    """
    def __init__(self,nx,nu,NN,model_name = "vdp3", data_mode="system",hidden_dim = 3, stable_dim=64,batch_size=128,steps=0.01,time=5,traj=200,epoch=100):
        """KoopmanEigenfunctions 
        
        Arguments:
            BasisFunctions {basis_function} -- function to lift the state
            n {integer} -- number of states
            max_power {integer} -- maximum number to exponenciate each original principal eigenvalue
            A_cl {numpy array [Ns,Ns]} -- closed loop matrix in continuous time
            BK {numpy array [Ns,Nu]} -- control matrix 
        """
        self.model_name = model_name
        # self.env = gym.make(env_name)
        # self.state_dim = self.env.observation_space.shape[0]+1
        self.state_dim=nx
        self.hidden_dim = hidden_dim
        self.NN = NN
        # self.action_dim = self.env.action_space.shape[0]
        self.action_dim=nu
        self.stable_dim=stable_dim
        if self.model_name=='vdp':
            self.system=self.vdp
        if self.model_name=='toy':
            self.system=self.toy
        if self.model_name=='pendulum':
            self.system=self.pendulum
        if self.model_name=='robot':
            self.system=self.robot
        if self.model_name=='duffing':
            self.system=self.duffing
        if self.model_name=='new_system':
            pass
        if self.model_name=='unknown_system':
            pass
        self.diffeomorphism_model = None
        self.lambda1 = 0.6
        self.lambda2 = 0.2
        self.lambda3 = 0.1
        
        self.dt = steps
        
        self.batch_size=batch_size
        
        # self.replay_buffer = ReplayBuffer(100000) 
        
        self.nsim=int(time/steps)  
        self.ntraj=traj 
        self.epoch=epoch
          
##  added content start
### add system
### toy system
    def toy(self, P, steps, sets):
        x, y = P
        # miu, lamda = sets
        u=sets
        dx = -0.05*x
        dy = -1 * (y - x**2) + u**2
        return [x + dx * steps, y + dy * steps]
### pend system
    def pend(self,P,time,steps):       
        t = np.linspace(0, time, steps)
        p1=odeint(self.pendulum,P,t)
        return p1
    def pendulum(self,y, t):
        x, xdot = y
        ydot = [xdot, - np.sin(x)]
        return ydot
### robot system
    def robot(self,P, steps, sets):
        x, y, z = P
        # miu, lamda = sets
        u, v=sets
        dx = np.cos(z)*u
        dy = np.sin(z)*u
        dz = v
        return [x + dx * steps, y + dy * steps, z+dz*steps]
### vdp system
    def vdp(self,P, steps, sets):
        x, y= P
        # miu, lamda = sets
        u =sets
        dy = 2*(1-x**2)*y-x +u
        y=y + dy * steps
        dx = y
        return [x + dx * steps, y]
#### duffing system    
    def duffing(self,z, steps, sets):
        x,y=z
        u=sets       
        yd=x-x**3+u
        y=y+yd*steps
        dx=y
        return [x + dx * steps, y]
        
     
### train data   
    def random_rollout(self):
        ##parameters
        nx=self.state_dim
        nu=self.action_dim
        nsim=self.nsim
        ntraj=self.ntraj
        steps=self.dt
        setss=4*np.random.rand(ntraj,nsim,nu)-2
        ##parameters
        
        ##position iteration
        P0 = 4*np.random.rand(ntraj,nx)-2
        d = []
        for i in range(ntraj):
            p0=P0[i]
            ### other system
            P1=[p0]
            for j in range(nsim):
                sets=setss[i,j,:]
                p1 = self.system(p0, steps, sets)
                P1.append(p1)
                p0=p1
            ### pendulum system
            # P1=self.pend(p0,time,steps)
            ### pend end
            d.append(P1)
        dnp = np.asarray(d,'float64')
        x=dnp
        u=setss
        return x, u
###add batch size
    def sample(self,batch_size,x_train,u_train):
        ##parameters
        nsim=self.nsim
        ntraj=self.ntraj
        ##paraneters
        # x, u= self.random_rollout()
        x, u=x_train,u_train
        ############################################################## old samples
        ind=np.random.randint(0, nsim, size = batch_size)
        i=np.random.randint(0, ntraj, size=1)
        X,U,Y=[],[],[]
        for j in ind:
            x0=x[i,j,:]
            x1=x[i,j+1,:]
            u0=u[i,j,:]
            X.append(x0)
            U.append(u0)
            Y.append(x1)
        return np.array(X).reshape(batch_size, -1), np.array(U).reshape(batch_size, -1), np.array(Y).reshape(batch_size, -1)                                                                                                                                        
####unknown system
    def get_data(self,path_x,path_u):
        x_train=np.load(path_x)
        u_train=np.load(path_u)
        return x_train, u_train
### lqr procedure
    def policy_rollout(self):
        A, B = self.get_system()
        Q = np.eye(self.hidden_dim)
        R = np.array([[0.01]])
        K, _, _ = control.lqr(A, B, Q, R)
        # ref = torch.FloatTensor([[0.0, 0.0, 0.1, 0., 0.]])
        ref = torch.FloatTensor([[0.0, 1.0]])
        # ref = torch.FloatTensor([[np.pi/2, 1.0, 0.0, 0., 0.]])
        ref = self.encoder(ref).detach().numpy()
        # obs_old = self.transform_state(self.env.reset())
        obs_old=[0.5,-0.5]
        #obs_old[2] = obs_old[2] / 8.0
        sat=[obs_old]
        for _ in range(200):
            state = torch.FloatTensor(obs_old.reshape((1, -1)))
            y = self.encoder(state).detach().numpy()
            action = -np.dot(K, (y-ref).T)
            action = np.clip(np.array([action.item()]), -1., 1.)
            steps=0.1
            obs=self.toy(obs_old,steps,action)
            obs_old = obs
            sat.append(obs)
        sat=np.array(sat)
        return sat
### pre
    def pre(self,obs_old0=[-0.5,0.5],kk=1000):
    # ##################################################predictor
    # obs_old=[-0.5,0.5]
        obs_old=np.asarray(obs_old0,'float64')
        sat_pre=[obs_old]
        setss=4*np.random.rand(kk,1)-2
        for i in range(kk):
            state = torch.FloatTensor(obs_old.reshape((1, -1)))
            y = ((self.encoder(state)+self.encoder1(state[:,0].reshape(1,-1))+self.encoder2(state[:,1].reshape(1,-1)))/3)
            ut=np.array(setss[i]).reshape(1,1)
            ut = torch.FloatTensor(ut)
            y1dt = self.propagate(torch.cat((y, ut), axis = -1))
            y1 = y + self.dt*y1dt
            xt1 = self.decoder(y1).detach().numpy()
            obs_old = xt1
            sat_pre.append(np.squeeze(obs_old))
        sat_pre=np.asarray(sat_pre,'float64')
        ####################################################actual
        obs_old=np.asarray(obs_old0,'float64')
        sat_true=[obs_old]
        for _ in range(kk):    
            # steps=0.1
            action=np.array(setss[_]).reshape(1,1)
            obs=self.system(obs_old,self.dt,action)
            obs_old = np.asarray(obs,'float64')
            sat_true.append(obs)
        sat_true=np.asarray(sat_true,'float64')
        ##sat
        from sklearn.metrics import mean_squared_error
        error1=[]
        for i in range(kk-1):
            error1.append(mean_squared_error(sat_pre[:i+1,:],sat_true[:i+1,:]))
            
        from statsmodels import robust
        # np.mean(robust.mad(sat_pre-sat_true, axis=0))
        error2=[]
        for i in range(kk-1):
            error2.append(np.mean(robust.mad(sat_pre[:i+1,:]-sat_true[:i+1,:], axis=0)))
        
        error1=np.array(error1)
        error2=np.array(error2)
        return sat_true,sat_pre, error1, error2
    
        
    def pre_plot(self,sat_true,sat_pre, error1, error2):
        
        import matplotlib.pyplot as plt   
        kk=sat_true.shape[1]
        nsim=sat_true.shape[0]
        t_eval = self.dt * np.arange(nsim)
        t_pred=t_eval.squeeze()
        fig, ax = plt.subplots()        
        for ii in range(kk):
            plt.subplot(kk, 1, ii+1)
            plt.plot(t_pred,sat_true[:,ii], linewidth=2, label='Nominal', color='tab:gray')
            # fill_between(t_pred,e_mean_nom[ii,:]-e_std_nom[ii,:], e_mean_nom[ii,:]+e_std_nom[ii,:], alpha=0.2, color='tab:gray')           
            plt.plot(t_pred,sat_pre[:,ii], linewidth=2, label='Nominal', color='tab:green')
            # fill_between(t_pred,e_mean_edmd[ii,:]-e_std_edmd[ii,:], e_mean_edmd[ii,:]+e_std_nom[ii,:], alpha=0.2, color='tab:green')
            
            plt.xlabel("$T$",fontsize=12)
            plt.ylabel("$x_%d$"%(ii+1),fontsize=12)
            plt.grid()
            plt.xlim(0, t_pred[-1])
            if ii==0:
                # plt.title("Mean prediction error (+/-1 std)",fontsize=12)
                plt.legend(['Nominal','Prediction'],fontsize=12)

####  added controllability
    def controllability(self,A,B):
        AB={}
        AB[0]=B
        for i in range(1,A.shape[0]):
            AB[i]=A.dot(AB[i-1])
        q=np.column_stack((AB.values()))
        z=q.dot(q.T)
        return np.linalg.matrix_rank(z)   

    def build_DNN_model(self, jacobian_penalty=1., n_hidden_layers=2, layer_width=50, batch_size=64, dropout_prob=0.1):
        """build_NN_model 
        
        Keyword Arguments:
            n_hidden_layers {int} --  (default: {2})
            layer_width {int} --  (default: {50})
            batch_size {int} --  (default: {64})
            dropout_prob {float} --  (default: {0.1})
        """

        self.A_cl = from_numpy(self.A_cl)
        if self.NN=="AE":
            self.NN_model = AEKoopman(self.state_dim, self.action_dim, self.model_name, self.hidden_dim)
        if self.NN=="HNN":
            self.NN_model = HNNKoopman(self.state_dim, self.action_dim, self.model_name,  self.hidden_dim)
    
    def fit_DNN_model(self, max_iter, lr =0.001,learning_decay=0.95,l2=1e1,path1=None,path2=None,path3=None,path4=None):
        mseloss = nn.MSELoss()
        l1loss = nn.L1Loss()
        
        # encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = lr)
        # decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr = lr)
        # propagate_optimizer = torch.optim.Adam(self.propagate.parameters(), lr = lr)
        optimizer = optim.Adam(self.NN_model.parameters(),lr=lr,weight_decay=l2)
        lambda1 = lambda epoch: learning_decay ** max_iter
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = lr)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr = lr)
        encoder1_optimizer = torch.optim.Adam(self.encoder1.parameters(), lr = lr)
        encoder2_optimizer = torch.optim.Adam(self.encoder2.parameters(), lr = lr)
        encoder11_optimizer = torch.optim.Adam(self.encoderdiff11.parameters(), lr = lr)
        encoder12_optimizer = torch.optim.Adam(self.encoderdiff12.parameters(), lr = lr)
        encoder13_optimizer = torch.optim.Adam(self.encoderdiff13.parameters(), lr = lr)
        encoder21_optimizer = torch.optim.Adam(self.encoderdiff21.parameters(), lr = lr)
        encoder22_optimizer = torch.optim.Adam(self.encoderdiff22.parameters(), lr = lr)
        encoder23_optimizer = torch.optim.Adam(self.encoderdiff23.parameters(), lr = lr)
        propagate_optimizer = torch.optim.Adam(self.propagate.parameters(), lr = lr)

        for it in range(max_iter):
               loss_hist = []
               if self.data_mode=="system":
                   x_train,u_train=self.random_rollout()
                   x_test,u_test=self.random_rollout()
               else:
                   x_train,u_train=self.get_data(path1,path2)   
                   x_test,u_test=self.get_data(path3,path4)
               for _ in range(self.epoch):
                   xt, ut, xt1 = self.sample(self.batch_size,x_train,u_train)
                  
                   xt = torch.FloatTensor(xt)
                   ut = torch.FloatTensor(ut)
                   xt1 = torch.FloatTensor(xt1)
   
                   gt, gt1, xt_, xt1_ = self.NN_model.forward(xt, ut)
   
                   total_loss=self.NN_model.loss(gt, gt1, xt_, xt1_ )
                   
                   total_loss.backward()
                   optimizer.step()
                   optimizer.zero_grad()
                   
                   loss_hist.append(total_loss.detach().numpy())
               print("epoch: %d, loss: %2.5f" % (it, np.mean(loss_hist)))
               
               with torch.no_grad():
                   total_test_loss=[]
                   for _ in range(self.epoch):
                       xt, ut, xt1 = self.sample(self.batch_size,x_test,u_test)                  
                       xt = torch.FloatTensor(xt)
                       ut = torch.FloatTensor(ut)
                       xt1 = torch.FloatTensor(xt1)   
                       gt, gt1, xt_, xt1_ = self.NN_model.forward(xt, ut)
                       loss=self.NN_model.loss(gt, gt1, xt_, xt1_ )                   
                       total_test_loss.append(loss.detach().numpy())
               print("epoch: %d, validation loss: %2.5f" % (it, np.mean(total_test_loss)))
               
    def train(self, max_iter, lr =0.01):
         mseloss = nn.MSELoss()
         # l1loss = nn.L1Loss()
         
         encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = lr)
         decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr = lr)
         encoder1_optimizer = torch.optim.Adam(self.encoder1.parameters(), lr = lr)
         encoder2_optimizer = torch.optim.Adam(self.encoder2.parameters(), lr = lr)
         encoder11_optimizer = torch.optim.Adam(self.encoderdiff11.parameters(), lr = lr)
         encoder12_optimizer = torch.optim.Adam(self.encoderdiff12.parameters(), lr = lr)
         encoder13_optimizer = torch.optim.Adam(self.encoderdiff13.parameters(), lr = lr)
         encoder21_optimizer = torch.optim.Adam(self.encoderdiff21.parameters(), lr = lr)
         encoder22_optimizer = torch.optim.Adam(self.encoderdiff22.parameters(), lr = lr)
         encoder23_optimizer = torch.optim.Adam(self.encoderdiff23.parameters(), lr = lr)
         propagate_optimizer = torch.optim.Adam(self.propagate.parameters(), lr = lr)
 
 
         for it in range(max_iter):
             loss_hist = []
             x_train,u_train=self.random_rollout()
             for _ in range(100):
                 # xt, ut, xt1 = self.replay_buffer.sample(64)
                 xt, ut, xt1 = self.sample(self.batch_size,x_train,u_train)
 
                 xt = torch.FloatTensor(xt)
                 ut = torch.FloatTensor(ut)
                 xt1 = torch.FloatTensor(xt1)
 
                 gt, gt1_p0, gt1_pg,deltp1_,deltp2_,xt_,xt1_ = self.forward(xt, ut, xt1)
 
                 ae_loss = self.scale_loss(gt1_p0+deltp1_+deltp2_, gt1_pg)
                 pred_loss = self.scale_loss(deltp1_, gt1_pg-gt1_p0)               
                 pred_loss2= self.scale_loss(deltp2_,gt1_pg-(gt1_p0+deltp1_))
                 ae_loss1 = mseloss(xt_, xt)
                 pred_loss1 = mseloss(xt1_, xt1)
                 A,B=self.get_system()
                 l2=A.shape[0]-self.controllability(A,B)
                 # p2=gt1_p0+deltp1_+deltp2_
                 # metric_loss = l1loss(torch.norm(gt1_p0-gt, dim=1), torch.norm(xt1-xt, dim=1))
                 metric_loss = mseloss(torch.norm(gt1_p0-gt, dim=1), torch.norm(xt1-xt, dim=1))
                 #reg_loss = torch.norm(self.propagate.weight.data[:, self.hidden_dim:])
                 total_loss = ae_loss + ae_loss1 + self.lambda1*pred_loss1 + self.lambda2*pred_loss + self.lambda2*pred_loss2+self.lambda2*metric_loss+self.lambda3*l2
                 # total_loss = ae_loss + ae_loss1 + self.lambda1*pred_loss1 + self.lambda2*pred_loss + self.lambda2*pred_loss2+self.lambda3*l2
                 
 
                 encoder_optimizer.zero_grad()
                 decoder_optimizer.zero_grad()
                 encoder1_optimizer.zero_grad()
                 encoder2_optimizer.zero_grad()
                 encoder11_optimizer.zero_grad()
                 encoder12_optimizer.zero_grad()
                 encoder13_optimizer.zero_grad()
                 encoder21_optimizer.zero_grad()
                 encoder22_optimizer.zero_grad()
                 encoder23_optimizer.zero_grad()
                 propagate_optimizer.zero_grad()
                 
                 
                 total_loss.backward()
                 
                 encoder_optimizer.step()
                 decoder_optimizer.step()
                 encoder1_optimizer.step()
                 encoder2_optimizer.step()
                 encoder11_optimizer.step()
                 encoder12_optimizer.step()
                 encoder13_optimizer.step()
                 encoder21_optimizer.step()
                 encoder22_optimizer.step()
                 encoder23_optimizer.step()
                 propagate_optimizer.step()
                 loss_hist.append(total_loss.detach().numpy())
             print("epoch: %d, loss: %2.5f" % (it, np.mean(loss_hist)))



    def save_NN_model(self):
        # save(self.diffeomorphism_model.state_dict(), filename)
        self.NN_model.save()

    def load_NN_model(self):
        # self.diffeomorphism_model.load_state_dict(load(filename))
        self.NN_model.load()

    