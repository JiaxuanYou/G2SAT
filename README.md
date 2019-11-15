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

You can try out the following 4 steps one by one.

1. Preprocess data
```bash
python conversion.py --src dataset/train_formulas/ -s dataset/train_set/
python conversion.py --src dataset/test_formulas/ -s dataset/test_set/
```

2. Train G2SAT
```bash
python main_train.py --epoch_num 201
```
After this step, trained G2SAT models will be saved in `model/` directory.

3. Use G2SAT to generate Formulas
```bash
python main_test.py --epoch_load 200
```
After this step, generated graphs will be saved to `graphs/` directory. 1 graph is generated out of 1 template.

Graphs will be saved in 2 formats: a single `.dat` file containing all the generated graphs; a directory where each generated graph is saved as a single `.dat` file. 

(It may take fairly long time: Runing G2SAT is fast, but updating networkx takes the majority of time in current implementation.)

We can then generate CNF formulas from the generated graphs
```bash
python conversion.py --src graphs/GCN_3_32_preTrue_dropFalse_yield1_019501.120000_0.dat --store-dir formulas --action=lcg2sat
```
4. Analyze results
We make use of this script to compute the scale-free structure (http://www.iiia.csic.es/~levy/software/scalefree.cpp)
```bash
g++ -o eval/scale_free eval/scale_free.cpp
python eval/evaluate_formulas.py -s eval/scalefree -d formulas/ 
```
This will print out the mean/std of the graph statistics of formulas in the formulas/ directory. You could also dump the raw statistics by adding the following flag: -o graph_statistics.csv

You are highly encouraged to tune all kinds of hyper-parameters to get better performance. We only did very limited hyper-parameter tuning.

We recommend using tensorboard to monitor the training process. To do this, you may run
```bash
tensorboard --logdir runs
```

## Citation
If you find this work useful, please cite our paper:
```latex
@article{you2019g2sat,
  title={G2SAT: Learning to Generate SAT Formulas},
  author={You, Jiaxuan and Wu, Haoze and Barrett, Clark and Ramanujan, Raghuram and Leskovec, Jure},
  journal={NeurIPS},
  year={2019}
```