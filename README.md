# G2SAT: Learning to Generate SAT Formulas
This repository is the official PyTorch implementation of "G2SAT: Learning to Generate SAT Formulas".

[Jiaxuan You*](https://cs.stanford.edu/~jiaxuan/), [Haoze Wu*](https://anwu1219.github.io/), [Clark Barrett](https://theory.stanford.edu/~barrett/), [Raghuram Ramanujan](https://www.davidson.edu/people/raghu-ramanujan), [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html), [Position-aware Graph Neural Networks](http://proceedings.mlr.press/v97/you19b/you19b.pdf), NeurIPS 2019.

## Installation

- Install PyTorch (tested on 1.0.0), please refer to the offical website for further details
```bash
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
- Install PyTorch Geometric (tested on 1.1.2), please refer to the offical website for further details
```bash
pip install --verbose --no-cache-dir torch-scatter
pip install --verbose --no-cache-dir torch-sparse
pip install --verbose --no-cache-dir torch-cluster
pip install --verbose --no-cache-dir torch-spline-conv (optional)
pip install torch-geometric
```
- Install networkx (tested on 2.3), make sure you are not using networkx 1.x version!
```bash
pip install networkx
```
- Install tensorboardx
```bash
pip install tensorboardX
```


## Example Run


1. Preprocess data
```bash
python 
```

2. Train G2SAT
```bash
python main_train.py --epoch_num 201
```

3. Use G2SAT to generate Formulas
```bash
python main_test.py --epoch_load 200
```

4. Analyze results
```bash
python
```


You are highly encouraged to tune all kinds of hyper-parameters to get better performance. We only did very limited hyper-parameter tuning.

We recommend using tensorboard to monitor the training process. To do this, you may run
```bash
tensorboard --logdir runs
```

## Citation
If you find this work useful, please cite our paper:
```latex

```