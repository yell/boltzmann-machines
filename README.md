# Hierarchical Deep Models

## Models
* RBM (Bernoulli, Multinomial, Gaussian)
* Easy to add new type of RBM (implement new type of stochastic units or create new RBM from existing types of units)
* DBM with arbitrary number of layers of any types

## Features
* Serialization (tf saver + python class hyperparams + RNG state)
* Reproducible (random seeds)
* All models support both `float32` and `float64` precision

## RBM and DBM features
* momentum
* L2 weight decay
* maxnorm weight decay
* ***TODO***: rest
* ***TODO***: sparsity targets
* ***TODO***: dropout

## How to install
By default, the following commands install (among others) **tensorflow-gpu~=1.3.0**. If you want to install tensorflow without GPU support, replace corresponding line in [requirements.txt](requirements.txt). If you have already tensorflow installed, comment that line but note that for [edward](http://edwardlib.org/) to work correctly, you must have tf>=1.2.0rc installed.
```bash
git clone https://github.com/monsta-hd/hd-models
cd hd-models/
pip install -r requirements.txt
```
After installation, tests can be run with:
```bash
make test
```
