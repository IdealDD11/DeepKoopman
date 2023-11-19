# DeepKoopman

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

Document  
[Instruction document](https://github.com/IdealDD11/DeepKoopman/blob/49015e54d49e640eadd942560a5e109bdf3e33e8/Instruction%20source%20document.docx)
![]()
