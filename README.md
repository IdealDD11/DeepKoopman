
# DeepKoopman

<img src="https://github.com/IdealDD11/DeepKoopman/blob/main/DeepKoopman/PNG/2.png" width="600px">


Package requirements: 
* Torch 1.7.0
* Numpy 1.23.0
* Scipy 1.4.1
* Matplotlib 3.2.3
* Argparse 1.1
* Control	0.8.3


Installation
```
pip install deepKoopman

import deepKoopman
from deepKoopman import DeepKoopman
```

Modelling
```
model=DeepKoopman(args...)
model.train(args...)
model.save()
model.load()
model.pre()
```

Control
```
model.policy_rollout(Q,R,...)
```

Parameters for embeded systems
```
"""
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
```

Parameters for unknown systems
```
"""
parameters:
            model name:
                unknown system  
            provide data paths:
                path1 state_trainning_data
                path2 controlinput_training_data
                path3 state_test_data
                path4 controlinput_test_data
"""                        
```

Example
```
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
```

Documentation  

* [Instruction document (CN)](https://github.com/IdealDD11/DeepKoopman/blob/f2f1dea6f99933591d36ecc50af93a3e5a931ce4/Instruction%20source%20document.pdf)

  [Process steps](https://github.com/IdealDD11/DeepKoopman/blob/main/DeepKoopman/PNG/3.png)

* [Instruction document (EN)](https://github.com/IdealDD11/DeepKoopman/blob/0abcb60cba3b40cf6db51a2c3360ddc43e8f2f7d/DeepKoopman/Instruction%20source%20document_EN.pdf)

  [Process steps](https://github.com/IdealDD11/DeepKoopman/blob/main/DeepKoopman/PNG/4.png)



Theoretical  

* [S. Sinha, S. P. Nandanoori, J. Drgona, and D. Vrabie, “Data-driven stabilization of discrete-time control-affine nonlinear systems: A Koopman operator approach,” in 2022 European Control Conference (ECC), 2022, pp. 552–559.](https://doi.org/10.23919/ECC55457.2022.9837986)
* [S. Daniel-Berhe and H. Unbehauen, “Experimental physical parameter estimation of a thyristor driven DC-motor using the HMF-method,” Control Engineering Practice, vol. 6, no. 5, pp. 615–626, 1998.](https://doi.org/10.1016/S0967-0661(98)00036-7)
* [S. Klus, F. Nuske, S. Peitz, J. H. Niemann, and C. Schutte, “Data-driven approximation of the Koopman generator: Model reduction, system identification, and control,” Physica D: Nonlinear Phenomena, vol. 406, p. 132416, 2020.](https://doi.org/10.1016/j.physd.2020.132416)
* [N. Takeishi, Y. Kawahara, and T. Yairi, “Learning Koopman invariant subspaces for dynamic mode decomposition,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017, pp. 1130–1140.](https://api.semanticscholar.org/CorpusID:22736336)
* [B. Lusch, J. Kutz, and S. Brunton, “Deep learning for universal linear embeddings of nonlinear dynamics,” Nature Communications, vol. 9, p. 4950, 2018.](https://doi.org/10.1038/s41467-018-07210-0)
* [S. L. Brunton, B. W. Brunton, J. L. Proctor, E. Kaiser, and J. N. Kutz,“Chaos as an intermittently forced linear system,” Nature Communications, vol. 8, no. 1, p. 19, 2017.](https://doi.org/10.1038/s41467-017-00030-8})
* [Ch. 2 - The Simple Pendulum (mit.edu)](http://underactuated.mit.edu/pend.html)
* [S. L. Brunton, B. W. Brunton, J. L. Proctor, K. J. Nathan, and H. A.Kestler, “Koopman invariant subspaces and finite linear representationsof nonlinear dynamical systems for control,” Plos One, vol. 11, no. 2, p. e0150171, 2016. 
](https://doi.org/10.1371/journal.pone.0150171)
* [P. J. Schmid and J. Sesterhenn, “Dynamic mode decomposition of numerical and experimental data,” Journal of Fluid Mechanics, vol. 656, no. 10, pp. 5–28, 2010. ](https://doi.org/10.1017/S0022112010001217)
* [M. Korda and I. Mezic, “Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control,” Automatica, vol. 93, pp. 149–160, 2016. ](https://doi.org/10.1016/j.automatica.2018.03.046)
* [J. L. Proctor, S. L. Brunton, and J. N. Kutz, “Generalizing Koopman theory to allow for inputs and control,” SIAM Journal on Applied Dynamical Systems, vol. 17, p. 909–930, 2016.](https://doi.org/10.1137/16M1062296)
