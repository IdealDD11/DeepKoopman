# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:31:07 2023

@author: Ideal
"""



import numpy as np
import torch
import torch.nn as nn
import control
import os
import argparse
import sys  
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class DeepKoopman():
    def __init__(self,nx,nu,model_name = "vdp", hidden_dim = 3, stable_dim=64,batch_size=128,steps=0.01,time=5,ntraj=200):
        self.model_name = model_name
        self.state_dim=nx
        self.hidden_dim = hidden_dim
        self.action_dim=nu
        self.stable_dim=stable_dim
        if self.model_name=='vdp':
            self.system=self.vdp
        if self.model_name=='toy':
            self.system=self.toy
        if self.model_name=='pendulum':
            self.system=self.pendulum
        if self.model_name=='duffing':
            self.system=self.duffing
        if self.model_name=='new_system':
            pass
        if self.model_name=='unknown_system':
            pass
        self.encoder = nn.Sequential(nn.Linear(self.state_dim, self.stable_dim),
                                      nn.PReLU(),
                                      nn.Linear(self.stable_dim, self.stable_dim),
                                      nn.PReLU(),
                                      nn.Linear(self.stable_dim, self.hidden_dim))
        
        self.decoder = nn.Sequential(nn.Linear(self.hidden_dim, self.stable_dim),
                                      nn.PReLU(),
                                      nn.Linear(self.stable_dim, self.stable_dim),
                                      nn.PReLU(),
                                      nn.Linear(self.stable_dim, self.state_dim))
        
        self.propagate = nn.Linear(self.hidden_dim+self.action_dim, self.hidden_dim, bias = False)
        
        self.lambda1 = 0.6
        self.lambda2 = 0.2
        self.lambda3 = 0.1
        
        self.dt = steps
        
        self.batch_size=batch_size
        
        self.nsim=int(time/steps)  
        self.ntraj=ntraj
    
    def get_system(self):
        weight = self.propagate.weight.data.numpy()
        A = weight[:, :self.hidden_dim]
        B = weight[:, self.hidden_dim:]
        return A, B
    def get_data(self,path_x,path_u):
        x_train=np.load(path_x)
        u_train=np.load(path_u)
        return x_train, u_train
    
    def forward(self, xt, ut,xt1):
        gt = self.encoder(xt)
        gt1_ = self.encoder(xt1)
        xt_ = self.decoder(gt)
        gtdot = self.propagate(torch.cat((gt, ut), axis = -1))
        gt1 = gt + self.dt*gtdot
        xt1_ = self.decoder(gt1)
        return gt, gt1, gt1_,xt_, xt1_
    
    def save(self):
        if not os.path.exists("weights/"):
            os.mkdir("weights/")
        file_name = "weights/" + self.model_name + ".pt"
        # file_name = "weights/" + 'robot-ae' + ".pt"
        torch.save({"encoder" : self.encoder.state_dict(),
                    "decoder" : self.decoder.state_dict(),
                    "propagate" : self.propagate.state_dict()}, file_name)
        print("save model to " + file_name)
    
    def load(self):
        try:
            if not os.path.exists("weights/"):
                os.mkdir("weights/")
            file_name = "weights/" + self.model_name + ".pt"
            checkpoint = torch.load(file_name)
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
            self.propagate.load_state_dict(checkpoint["propagate"])
            print("load model from " + file_name)
        except:
            print("fail to load model!")

###  added content start
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
        P0 = 4*np.random.rand(ntraj,nx)-2
        d = []
        for i in range(ntraj):
            p0=P0[i]
            ### other system
            P1=[p0]
            for j in range(nsim):
                sets=setss[i,j,:]
                p1 = self.system(p0, steps, sets)
                P1.append(np.array(p1).squeeze())
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
### lqr procedure
    def policy_rollout(self,Q,R,ref,obs_old,nsim):
        A, B = self.get_system()
        K, _, _ = control.lqr(A, B, Q, R)
        ref=np.array(ref,'float64')
        ref = torch.FloatTensor(ref)
        ref = self.encoder(ref).detach().numpy()
        sat=[np.array(obs_old)]  
        uu=[]
        for _ in range(nsim):
            obs_old=np.array(obs_old,'float64')
            state = torch.FloatTensor(obs_old.reshape((1, -1)))
            y = self.encoder(state).detach().numpy()
            action = -np.dot(K, (y-ref).T)
            action = np.clip(np.array([action.item()]), -1., 1.)
            obs=self.system(obs_old,self.dt,action)
            obs_old = obs
            sat.append(np.array(obs).squeeze())
            uu.append(action)
        sat=np.asarray(sat,'float64')
        uu=np.asarray(uu,'float64')
        return K,sat,uu
### policy plot
    def policy_plot(self,sat,uu):               
        nn_=sat.shape[1]
        nsim=sat.shape[0]
        mm_=uu.shape[1]
        t_eval = self.dt * np.arange(nsim)
        t_pred=t_eval.squeeze()
        fig, ax = plt.subplots()        
        for ii in range(nn_):
            plt.subplot(nn_+mm_, 1, ii+1)
            plt.plot(t_pred,sat[:,ii], linewidth=2, label='Nominal', color='tab:gray')
            plt.xlabel("$T$",fontsize=12)
            plt.ylabel("$x_%d$"%(ii+1),fontsize=12)    
            plt.grid()
            plt.xlim(0, t_pred[-1])
        for ii in range(mm_):
            plt.subplot(nn_+mm_, 1, nn_+ii+1)
            plt.plot(t_pred[:-1],uu[:,ii], linewidth=2, label='Nominal', color='tab:green')
            plt.xlabel("$T$",fontsize=12)
            plt.ylabel("$u$",fontsize=12)
            plt.grid()
            plt.xlim(0, t_pred[-1])           
### pre
    def pre(self,obs_old0=[-0.5,0.5],kk=1000):
    # ##################################################predictor
    # obs_old=[-0.5,0.5]
        obs_old=np.array(obs_old0,'float64')
        sat_pre=[obs_old]
        setss=4*np.random.rand(kk,1)-2
        for i in range(kk):
            state = torch.FloatTensor(obs_old.reshape((1, -1)))
            y = self.encoder(state)
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
            sat_true.append(np.array(obs).squeeze())
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
           
        nn_=sat_true.shape[1]
        nsim=sat_true.shape[0]
        t_eval = self.dt * np.arange(nsim)
        t_pred=t_eval.squeeze()
        fig, ax = plt.subplots()        
        for ii in range(nn_):
            plt.subplot(nn_, 1, ii+1)
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
        q=np.column_stack(list(AB.values()))
        z=q.dot(q.T)
        return np.linalg.matrix_rank(z)   
#### self definedloss
    def scale_loss(self,x, y):
        z=x-y
        loss=0
        for i in range(z.shape[1]):
            for j in range(i+1,z.shape[1]):
                loss=loss+(torch.pow((z[:,i] - z[:,j]), 2))
        return torch.mean(loss)   
    def total_loss_fn(self, xt, ut, xt1, gt, gt1, gt1_,xt_, xt1_):
        mseloss = nn.MSELoss()
        l1loss = nn.L1Loss()
        ae_loss = mseloss(xt_, xt)
        pred_loss = mseloss(xt1_, xt1)
        metric_loss = l1loss(gt1,gt1_)
        # metric_loss = l1loss(torch.norm(gt1-gt, dim=1), torch.norm(xt1-xt, dim=1))
        #reg_loss = torch.norm(self.propagate.weight.data[:, self.hidden_dim:])
        A,B=self.get_system()
        l2=A.shape[0]-self.controllability(A,B)

        total_loss = ae_loss + self.lambda1*pred_loss + self.lambda2*metric_loss+self.lambda3*l2
        # total_loss = ae_loss + self.lambda1*pred_loss 
        return total_loss
                
    
    def train(self, max_iter, epoch, lr =0.001,path1=None,path2=None,path3=None,path4=None):
        
        
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = lr)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr = lr)
        propagate_optimizer = torch.optim.Adam(self.propagate.parameters(), lr = lr)

       
        if self.model_name=="unknown_system":
            x_train,u_train=self.get_data(path1,path2)   
            x_test,u_test=self.get_data(path3,path4)
        else:
            x_train,u_train=self.random_rollout()
            x_test,u_test=self.random_rollout()
        for it in range(max_iter):
            loss_hist = []
            for _ in range(epoch):

                xt, ut, xt1 = self.sample(self.batch_size,x_train,u_train)

                xt = torch.FloatTensor(xt)
                ut = torch.FloatTensor(ut)
                xt1 = torch.FloatTensor(xt1)

                gt, gt1, gt1_,xt_, xt1_ = self.forward(xt, ut,xt1)
                total_loss=self.total_loss_fn( xt, ut, xt1,gt, gt1, gt1_,xt_, xt1_)
               
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                propagate_optimizer.zero_grad()
                
                total_loss.backward()
                
                encoder_optimizer.step()
                decoder_optimizer.step()
                propagate_optimizer.step()
                loss_hist.append(total_loss.detach().numpy())
            print("epoch: %d, loss: %2.5f" % (it, np.mean(loss_hist)))
            with torch.no_grad():
                    total_test_loss=[]
                    for _ in range(epoch):
                        xt, ut, xt1 = self.sample(self.batch_size,x_test,u_test)                  
                        xt = torch.FloatTensor(xt)
                        ut = torch.FloatTensor(ut)
                        xt1 = torch.FloatTensor(xt1)   
                        gt, gt1, xt_, xt1_ = self.forward(xt, ut,xt1)
                        loss=self.total_loss_fn( xt, ut, xt1,gt, gt1, gt1_,xt_, xt1_ )                   
                        total_test_loss.append(loss.detach().numpy())
            print("epoch: %d, validation loss: %2.5f" % (it, np.mean(total_test_loss)))

if __name__ == "__main__":
    """
    ====###Case1###==== default systems: vdp;duffing;pendulum;toy;robot  
    """
    """
    build NNKOOPMAN model for system with equations
    parameters:
                model_name: systems:vdp;duffing;toy;pendulum;
                max_iter: max iterations
                epoch: training epoch in one iteration
                hidden_dim: dimension of the lifted state
                stable_dim: size of hidden layers
                batch_size: batch size
                nx: dimension of the original state
                nu: dimension of the control input
                time: sampling period
                steps: sampling interval
                ntaj: number of trajectories
                mode: training or load model                            
    """ 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='vdp')    
    parser.add_argument("--max_iter", default=50)
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--hidden_dim", default=8, type=int)
    parser.add_argument("--stable_dim", default=64, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--nx", default=2, type=int)
    parser.add_argument("--nu", default=1, type=int)
    parser.add_argument("--time", default=50, type=int)
    parser.add_argument("--steps", default=0.01)
    parser.add_argument("--ntaj", default=200)
    parser.add_argument("--mode", default="train")
    # parser.add_argument("--mode", '-false')
    args = parser.parse_args()
    model = DeepKoopman(args.nx,args.nu,args.model_name, args.hidden_dim)  
    """
    train DEEPKOOPMAN model for system with equations
    save model; 
    load model;
    build LQR controller;
    """
    if args.mode== "train":
        model.train(args.max_iter, args.epoch,0.001)
        model.save()
    else:
        model.load()
        sat_true,sat_pre, error1, error2=model.pre()
        model.pre_plot(sat_true,sat_pre, error1, error2)
        #################LQR
        """
        LQR control: 
            K gain matrix; xx closed-system states
        """
        # A, B = model.get_system()
        Q = np.eye(args.hidden_dim)
        R = np.array([[0.1]])
        # K, _, _ = control.lqr(A, B, Q, R)    
        ref=[0.0, 0.0]
        x_0=[0.5,-0.5]
        nsim=200
        K,xx,uu=model.policy_rollout(Q, R, ref, x_0, nsim)
        model.policy_plot(xx,uu)
    
        
        
    """
    ====###Case2###==== unknown dynamics, please provide training data and test data 
    """        
    """
    build NNKOOPMAN model for system with equations
    model name:
        unknown system  
    provide data paths:
        path1 state_trainning_data
        path2 controlinput_training_data
        path3 state_test_data
        path4 controlinput_test_data  
    """  
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='unknown_system')   
    parser.add_argument("--max_iter", default=50)
    parser.add_argument("--epoch", default=50)
    parser.add_argument("--hidden_dim", default=8, type=int)
    parser.add_argument("--stable_dim", default=64, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--nx", default=2, type=int)
    parser.add_argument("--nu", default=1, type=int)
    parser.add_argument("--time", default=50, type=int)
    parser.add_argument("--steps", default=0.01)
    parser.add_argument("--ntaj", default=200)
    parser.add_argument("--mode", default="train")
    # parser.add_argument("--mode", '-false')
    args = parser.parse_args()
    path1='path_for_state_data'
    path2='path_for_controlinput_data'
    path3='path_for_state_data'
    path4='path_for_controlinput_data'
    model = DeepKoopman(args.nx,args.nu,args.model_name, args.hidden_dim) 
    """
    train DEEPKOOPMAN model for unknown systen with measurements
    save model weights; 
    load model weights;
    build LQR controller;
    """
    if args.mode == "train":
        model.train(args.max_iter, 0.001, path1,path2, path3,path4,)
        model.save()
    else:
        model.load()
        sat_true,sat_pre, error1, error2=model.pre()
        model.pre_plot(sat_true,sat_pre, error1, error2)
        model.save_weights()
        ##################LQR
        A, B = model.get_system()
        Q = np.eye(args.hidden_dim)
        R = np.array([[0.1]])
        K, _, _ = control.lqr(A, B, Q, R)
        
        



















