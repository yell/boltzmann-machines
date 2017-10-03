# Hierarchical Deep Models

## Features
### Restricted Boltzmann Machine (RBM)
* *k-step Contrastive Divergence*: *variable* learning rate and momentum, L2 weight decay, maxnorm, ***TODO***: rest, ***TODO***: dropout, ***TODO***: sparsity targets
* *different types of RBMs*: Bernoulli, Multinomial, Gaussian;
* *easy to add new type of RBM*: implement new type of stochastic units or create new RBM from existing types of units
* *visualization in Tensorboard*: learning curves (reconstruction RMSE, pseudo log-likelihood, free energy gap, L2 loss), distribution of weights and weights updates in TensorBoard ***TODO***: hidden activations and weight filters
* variable number of Gibbs steps withing training is not yet supported, but possible (need to implement `tf.while_loop` with variable number of steps)

### Deep Boltzmann Machine (DBM)
* arbitrary number of layers of any types
* initialize from greedy layer-wise pretrained RBMs and jointly fine-tune using PCD + mean-field approximation

### Hierarchical Dirichlet Prior (HDP)
***TODO***

### Stick-breaking Variational Autoencoder (SB-VAE)
***TODO***

### General
* easy to use `sklearn`-like interface
* serialization (tf saver + python class hyperparams + RNG state), easy to save and to load
* reproducible (random seeds)
* all models support both `float32` and `float64` precision
* easy to resume training; note that changing parameters other than placeholders or python-level parameters (such as `batch_size`, `learning_rate`, `momentum`, `sample_v_states` etc.) between `fit` calls have no effect as this would require altering the computation graph, which is not yet supported
* *visualization*: python routines to display images, learned filters, confusion matrices etc.

## Examples (***TODO*** add demo images, download models)
### RBM MNIST ([script](examples/rbm_mnist.py), *[notebook](notebooks/rbm_mnist.ipynb)*)
Train RBM on MNIST dataset and use it for classification.

| <div align="center">Algorithm</div> | Test Accuracy, % |
| :--- | :---: |
| RBM features + Logistic Regression | **97.26** |
| RBM features + k-NN | **96.90** |
| RBM + discriminative finetuning | **98.49** |

### DBM MNIST (~~[script]()~~, ~~[notebook]()~~)
### DBM CIFAR-10 Naïve (~~[script]()~~, ~~[notebook]()~~)
### DBM CIFAR-10 (~~[script]()~~, ~~[notebook]()~~)
### HDP (~~[script]()~~, ~~[notebook]()~~)
### HDP-DBM (~~[script]()~~, ~~[notebook]()~~)
### SB-VAE (~~[script]()~~, ~~[notebook]()~~)
### Usage
Use **script**s for training models from scratch, for instance
```bash
$ python rbm_mnist.py -h

Train Bernoulli-Bernoulli RBM on MNIST dataset.

Momentum is initially 0.5 and gradually increases to 0.9.
Training time is approx. 2.5 times faster using single-precision rather than double
with negligible difference in reconstruction error, pseudo log-lik is more noisy though.

usage: rbm_mnist.py [-h] [--n-train N] [--n-val N] [--n-hidden N] [--vb-init]
                    [--n-gibbs-steps N] [--lr LR] [--epochs N]
                    [--batch-size N] [--l2 L2] [--sample-v-states] [--dtype D]
                    [--model-dirpath DIRPATH]

optional arguments:
  -h, --help            show this help message and exit
  --n-train N           number of training examples (default: 55000)
  --n-val N             number of validation examples (default: 5000)
  --n-hidden N          number of hidden units (default: 1024)
  --vb-init             initialize visible biases as logit of mean values of
                        features, otherwise zero init (default: True)
  --n-gibbs-steps N     number of Gibbs steps per weight update (default: 1)
  --lr LR               initial learning rates (default: 0.01)
  --epochs N            number of epochs to train (default: 100)
  --batch-size N        input batch size for training (default: 10)
  --l2 L2               L2 weight decay (default: 1e-05)
  --sample-v-states     sample visible states, otherwise use probabilities w/o
                        sampling (default: False)
  --dtype D             datatype precision to use, {'float32', 'float64'}
                        (default: float32)
  --model-dirpath DIRPATH
                        directory path to save the model (default:
                        ../models/rbm_mnist/)
```
or download pretrained ones with default parameters using `models/fetch_models.sh`, 
</br>
and check **notebook**s for corresponding testing / visualization etc.

## How to install
By default, the following commands install (among others) **tensorflow-gpu~=1.3.0**. If you want to install tensorflow without GPU support, replace corresponding line in [requirements.txt](requirements.txt). If you have already tensorflow installed, comment that line but note that for [edward](http://edwardlib.org/) to work correctly, you must have tf>=1.2.0rc installed.
```bash
git clone https://github.com/monsta-hd/hd-models
cd hd-models/
pip install -r requirements.txt
```
See [here](docs/virtualenv.md) how to run from a ***virtual environment***.
</br>
See [here](docs/docker.md) how to run from a ***docker container***.

After installation, tests can be run with:
```bash
make test
```
All the necessary data can be downloaded with:
```bash
make data
```
### Common installation issues
**ImportError: libcudnn.so.6: cannot open shared object file: No such file or directory**.<br/>
TensorFlow 1.3.0 assumes cuDNN v6.0 by default. If you have different one installed, you can create symlink to `libcudnn.so.6` in `/usr/local/cuda/lib64` or `/usr/local/cuda-8.0/lib64`. More details [here](https://stackoverflow.com/questions/42013316/after-building-tensorflow-from-source-seeing-libcudart-so-and-libcudnn-errors).

## Tensorboard visualization
***TODO***

## Contributing
***TODO***
