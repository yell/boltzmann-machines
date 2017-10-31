# Boltzmann Machines

## Implemented
### Restricted Boltzmann Machines (RBM)
* *k-step Contrastive Divergence*: *variable* learning rate and momentum, L2 weight decay, maxnorm, dropout, sparsity targets, ***TODO***: rest;
* *different types of RBMs*: Bernoulli, Multinomial, Gaussian;
* *easy to add new type of RBM*: implement new type of stochastic units or create new RBM from existing types of units
* *visualization in Tensorboard*: learning curves (reconstruction RMSE, pseudo log-likelihood, free energy gap, L2 loss), distribution of weights and weights updates in TensorBoard; hidden activations and weight filters
* variable number of Gibbs steps withing training is not yet supported, but possible (need to implement `tf.while_loop` with variable number of steps) + see `init_from` method

### Deep Boltzmann Machines (DBM)
* arbitrary number of layers of any types
* initialize from greedy layer-wise pretrained RBMs and jointly fine-tune using PCD + mean-field approximation
* one can use `DBM` class with 1 hidden layer to train **RBM** with this more efficient algorithm + generating samples after training + AIS
* visualize filters and hidden activations for all layers
* sparsity targets
* visualize norms of weights in each layer
* visualize visible negative particles
* implemented Annealed Importance Sampling to estimate log partition function

## Features
* easy to use `sklearn`-like interface
* serialization (tf saver + python class hyperparams + RNG state), easy to save and to load
* reproducible (random seeds)
* all models support both `float32` and `float64` precision
* choose metrics to display during learning
* easy to resume training; note that changing parameters other than placeholders or python-level parameters (such as `batch_size`, `learning_rate`, `momentum`, `sample_v_states` etc.) between `fit` calls have no effect as this would require altering the computation graph, which is not yet supported; **however**, one can build model with new desired TF graph, and initialize weights and biases from old model by using `init_from` method
* *visualization*: python routines to display images, learned filters, confusion matrices etc.

## Examples (***TODO*** add demo images, download models)
### 1) RBM MNIST ([script](examples/rbm_mnist.py), *[notebook](notebooks/rbm_mnist.ipynb)*)
Train RBM on MNIST dataset and use it for classification.

| <div align="center">Algorithm</div> | Test Accuracy, % |
| :--- | :---: |
| RBM features + Logistic Regression | **98.21** |
| RBM features + k-NN | **96.96** |
| RBM + discriminative finetuning | **98.67** |

Also, [one-shot learning idea]:

| Number of labeled data pairs (train + val) | RBM + fine-tuning | random initialization |
| :---: | :---: | :---: |
| 60k (55k + 5k) | 98.67% (**+0.41%**) | 98.26% |
| 10k (9k + 1k) | 97.21% (**+2.48%**) | 94.73% |
| 1k (900 + 100) | 93.52% (**+4.82%**) | 88.70% |
| 100 (90 + 10) | 81.37% (**+5.35%**) | 76.02% |

How to reproduce the last table see [here](docs/rbm_discriminative.md).

### 2) DBM MNIST ([script](examples/dbm_mnist.py), *[notebook](notebooks/dbm_mnist.ipynb)*)

| Number of intermediate distributions | log(Z_mean) | log(Z-sigma), log(Z+sigma) | Avg. test ELBO |
| :---: | :---: | :---: | :---: |
| DBM paper (20'000) | 356.18 | 356.06, 356.29 | **-84.62** |
| 200'000 | 1040.39 | 1040.18, 1040.58 | **-86.37** |
| 20'000 | 1040.55 | 1039.71, 1041.23 | -86.70 |

### 3) DBM CIFAR-10 Naïve (~~[script]()~~, ~~[notebook]()~~)
### 4) DBM CIFAR-10 (~~[script]()~~, ~~[notebook]()~~)
### Usage
Use **script**s for training models from scratch, for instance
```
$ python rbm_mnist.py -h

Train Bernoulli-Bernoulli RBM on MNIST dataset.

Momentum is initially 0.5 and gradually increases to 0.9.
Training time is approx. 2.5 times faster using single-precision rather than double
with negligible difference in reconstruction error, pseudo log-lik is more noisy though.

usage: rbm_mnist.py [-h] [--n-train N] [--n-val N] [--n-hidden N] [--vb-init]
                    [--hb-init HB] [--n-gibbs-steps N] [--lr LR [LR ...]]
                    [--epochs N] [--batch-size N] [--l2 L2]
                    [--sample-v-states] [--dropout P] [--sparsity-target T]
                    [--sparsity-cost C] [--sparsity-damping D] [--dtype T]
                    [--model-dirpath DIRPATH]

optional arguments:
  -h, --help            show this help message and exit
  --n-train N           number of training examples (default: 55000)
  --n-val N             number of validation examples (default: 5000)
  --n-hidden N          number of hidden units (default: 1024)
  --vb-init             initialize visible biases as logit of mean values of
                        features, otherwise zero init (default: True)
  --hb-init HB          initial hidden bias (default: 0.0)
  --n-gibbs-steps N     number of Gibbs updates per iteration (default: 1)
  --lr LR [LR ...]      learning rate(s) (default: 0.05)
  --epochs N            number of epochs to train (default: 100)
  --batch-size N        input batch size for training (default: 10)
  --l2 L2               L2 weight decay coefficient (default: 1e-05)
  --sample-v-states     sample visible states, otherwise use probabilities w/o
                        sampling (default: False)
  --dropout P           probability of visible units being on (default: None)
  --sparsity-target T   desired probability of hidden activation (default:
                        0.1)
  --sparsity-cost C     controls the amount of sparsity penalty (default:
                        1e-05)
  --sparsity-damping D  decay rate for hidden activations probs (default: 0.9)
  --dtype T             datatype precision to use, {'float32', 'float64'}
                        (default: float32)
  --model-dirpath DIRPATH
                        directory path to save the model (default:
                        ../models/rbm_mnist/)
```
or download pretrained ones with default parameters using `models/fetch_models.sh`, 
</br>
and check **notebook**s for corresponding inference / visualization etc.

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

To run some notebooks you also need to install [**JSAnimation**](https://github.com/jakevdp/JSAnimation):
```bash
git clone https://github.com/jakevdp/JSAnimation
cd JSAnimation
python setup.py install
```
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

## TODO
* Centering trick
* ELBO and AIS for arbitrary DBM (again, visible and topmost hidden units can be analytically summed out)
