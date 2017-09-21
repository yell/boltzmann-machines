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
