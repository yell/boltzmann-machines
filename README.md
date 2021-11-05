<p float="left">
  <img src="img/dbm_mnist/rbm1.png" width="129" />
  <img src="img/dbm_mnist/samples.png" width="132" />
  <img src="img/dbm_cifar2/rbm_small_0.png" width="129" />
  <img src="img/dbm_cifar2/grbm.png" width="129" />
  <img src="img/dbm_cifar2/mrbm.png" width="129" />
  <img src="img/dbm_cifar/samples.png" width="129" />
</p>

# Boltzmann Machines
This repository implements generic and flexible RBM and DBM models with lots of features and reproduces some experiments from *"Deep boltzmann machines"* [**[1]**](#1), *"Learning with hierarchical-deep models"* [**[2]**](#2), *"Learning multiple layers of features from tiny images"* [**[3]**](#3), and some others.

## Table of contents
* [What's Implemented](#whats-implemented)
   * [Restricted Boltzmann Machines (RBM)](#restricted-boltzmann-machines-rbm)
   * [Deep Boltzmann Machines (DBM)](#deep-boltzmann-machines-dbm)
   * [Common features](#common-features)
* [Examples](#examples)
   * [#1 RBM MNIST: <a href="examples/rbm_mnist.py">script</a>, <a href="https://nbviewer.jupyter.org/github/monsta-hd/boltzmann-machines/blob/master/notebooks/rbm_mnist.ipynb">notebook</a>](#1-rbm-mnist-script-notebook)
   * [#2 DBM MNIST: <a href="examples/dbm_mnist.py">script</a>, <a href="https://nbviewer.jupyter.org/github/monsta-hd/boltzmann-machines/blob/master/notebooks/dbm_mnist.ipynb">notebook</a>](#2-dbm-mnist-script-notebook)
   * [#3 DBM CIFAR-10 "Naïve": <a href="examples/dbm_cifar_naive.py">script</a>, <a href="https://nbviewer.jupyter.org/github/monsta-hd/boltzmann-machines/blob/master/notebooks/dbm_cifar_naive.ipynb">notebook</a>](#3-dbm-cifar-10-naïve-script-notebook)
   * [#4 DBM CIFAR-10: <a href="examples/dbm_cifar.py">script</a>, <a href="https://nbviewer.jupyter.org/github/monsta-hd/boltzmann-machines/blob/master/notebooks/dbm_cifar.ipynb">notebook</a>](#4-dbm-cifar-10-script-notebook)
   * [How to use examples](#how-to-use-examples)
   * [Memory requirements](#memory-requirements)
* [Download models and stuff](#download-models-and-stuff)
* [TeX notes](#tex-notes)
* [How to install](#how-to-install)
   * [Common installation issues](#common-installation-issues)
* [Possible future work](#possible-future-work)
* [Contributing](#contributing)
* [References](#references)

## What's Implemented
### Restricted Boltzmann Machines (RBM) 
* [[computational graph]](img/tensorboard_rbm/tf_graph.png)
* k-step Contrastive Divergence;
* whether to sample or use probabilities for visible and hidden units;
* *variable* learning rate, momentum and number of Gibbs steps per weight update;
* *regularization*: L2 weight decay, dropout, sparsity targets;
* *different types of stochastic layers and RBMs*: implement new type of stochastic units or create new RBM from existing types of units;
* *predefined stochastic layers*: Bernoulli, Multinomial, Gaussian;
* *predefined RBMs*: Bernoulli-Bernoulli, Bernoulli-Multinomial, Gaussian-Bernoulli;
* initialize weights randomly, from `np.ndarray`-s or from another RBM;
* can be modified for greedy layer-wise pretraining of DBM (see [notes](#tex-notes) or [**[1]**](#1) for details);
* *visualizations in Tensorboard* (hover images for details) and more:
<p align="center">
  <img src="img/tensorboard_rbm/msre.png" height="156" title="Mean squared reconstruction error" />
  <img src="img/tensorboard_rbm/pll.png" height="156" title="Pseudo log-likelihood" />
  <img src="img/tensorboard_rbm/feg.png" height="156" title="Free energy gap [4]" />
</p>  

<p align="center">
  <img src="img/tensorboard_rbm/l2_loss.png" height="156" title="L2 loss (weight decay cost times 0.5||W||^2)" />
  <img src="img/tensorboard_rbm/dist_W.png" width="256" title="Distribution of weights and biases" />
  <img src="img/tensorboard_rbm/dist_hb.png" width="256" title="Distribution of weights and biases" />
</p>

<p align="center">
  <img src="img/tensorboard_rbm/dist_dW.png" width="258" title="Distribution of weights and biases updates" />
  <img src="img/tensorboard_rbm/dist_dvb.png" width="258" title="Distribution of weights and biases updates" />
  <img src="img/tensorboard_rbm/hist_W.png" width="258" title="Histogram of weights and biases" />
</p>

<p align="center">
  <img src="img/tensorboard_rbm/hist_hb.png" width="258" title="Histogram of weights and biases" />
  <img src="img/tensorboard_rbm/hist_dW.png" width="258" title="Histogram of weights and biases updates" />
  <img src="img/tensorboard_rbm/hist_dvb.png" width="258" title="Histogram of weights and biases updates" />
</p>

<p align="center">
  <img src="img/tensorboard_rbm/hidden_activations.gif" height="232" title="Hidden activations probabilities (means)" />
  <img src="img/tensorboard_rbm/mnist_5.gif" height="224" title="Weight filters" />
  <img src="img/tensorboard_rbm/mnist_8.gif" height="224" title="Weight filters" />
</p>

<p align="center">
  <img src="img/tensorboard_rbm/cifar_small_6_1.gif" width="148" title="Weight filters" />
  <img src="img/tensorboard_rbm/cifar_small_8_6.gif" width="148" title="Weight filters" />
  <img src="img/tensorboard_rbm/cifar_6.gif"  width="148" title="Weight filters" />
  <img src="img/tensorboard_rbm/cifar_18.gif" width="148" title="Weight filters" />
  <img src="img/tensorboard_rbm/cifar_9.gif"  width="148" title="Weight filters" />
</p>

### Deep Boltzmann Machines (DBM) 
* [[computational graph]](img/tensorboard_dbm/tf_graph.png)
* EM-like learning algorithm based on PCD and mean-field variational inference [**[1]**](#1);
* arbitrary number of layers of any types;
* initialize from greedy layer-wise pretrained RBMs (no random initialization for now);
* whether to sample or use probabilities for visible and hidden units;
* *variable* learning rate, momentum and number of Gibbs steps per weight update;
* *regularization*: L2 weight decay, maxnorm, sparsity targets;
* estimate partition function using Annealed Importance Sampling [**[1]**](#1);
* estimate variational lower-bound (ELBO) using logẐ (currently only for 2-layer binary BM);
* generate samples after training;
* initialize negative particles (visible and hidden in all layers) from data;
* `DBM` class can be used also for training RBM and its features: more powerful learning algorithm, estimating logẐ and ELBO, generating samples after training;
* *visualizations in Tensorboard* (hover images for details) and more:
<p align="center">
  <img src="img/tensorboard_dbm/msre.png"         height="157" title="Mean squared reconstruction error" />
  <img src="img/tensorboard_dbm/n_mf_updates.png" height="157" title="Actual number of mean-field updates" />
  <img src="img/tensorboard_dbm/W_norm.png"       height="157" title="Maximum absolute value of weight matrix (for each layer)" />
</p>  

<p align="center">
  <img src="img/tensorboard_dbm/dist_W.png"   width="258" title="Distribution of weights and biases (in each layer)" />
  <img src="img/tensorboard_dbm/dist_W2.png"  width="258" title="Distribution of weights and biases (in each layer)" />
  <img src="img/tensorboard_dbm/dist_hb2.png" width="258" title="Distribution of weights and biases (in each layer)" />
</p>

<p align="center">
  <img src="img/tensorboard_dbm/dist_dW.png"  width="258" title="Distribution of weights and biases updates (in each layer)" />
  <img src="img/tensorboard_dbm/dist_dvb.png" width="258" title="Distribution of weights and biases updates (in each layer)" />
  <img src="img/tensorboard_dbm/dist_mu2.png" width="258" title="Distribution of variational parameters (in each layer)" />
</p>

<p align="center">
  <img src="img/tensorboard_dbm/hist_W.png"  width="258" title="Histogram of weights and biases (in each layer)" />
  <img src="img/tensorboard_dbm/hist_dW.png" width="258" title="Histogram of weights and biases updates (in each layer)" />
  <img src="img/tensorboard_dbm/hist_mu.png" width="258" title="Histogram of variational parameters (in each layer)" />
</p>

<p align="center">
  <img src="img/tensorboard_dbm/mnist_filter_L1_5.gif" width="148" title="Weight filters (in each layer)" />
  <img src="img/tensorboard_dbm/mnist_filter_L2_6.gif" width="148" title="Weight filters (in each layer)" />
  <img src="img/tensorboard_dbm/cifar_filter_L1_6.gif" width="148" title="Weight filters (in each layer)" />
  <img src="img/tensorboard_dbm/cifar_filter_L2_2.gif" width="148" title="Weight filters (in each layer)" />
  <img src="img/tensorboard_dbm/cifar_filter_L2_6.gif" width="148" title="Weight filters (in each layer)" />
</p>

<p align="center">
  <img src="img/tensorboard_dbm/mnist_particle_L1_2.gif" width="148" title="Negative particles (in each layer)" />
  <img src="img/tensorboard_dbm/mnist_particle_L1_4.gif" width="148" title="Negative particles (in each layer)" />
  <img src="img/tensorboard_dbm/cifar_particle_L1_1.gif" width="148" title="Negative particles (in each layer)" />
  <img src="img/tensorboard_dbm/cifar_particle_L1_2.gif" width="148" title="Negative particles (in each layer)" />
  <img src="img/tensorboard_dbm/cifar_particle_L1_4.gif" width="148" title="Negative particles (in each layer)" />
</p>

<p align="center">
  <img src="img/tensorboard_dbm/mnist_particles_L23.gif" height="208" title="Negative particles (in each layer)" />
</p>

### Common features
* easy to use with `sklearn`-like interface;
* easy to load and save models;
* easy to reproduce (`random_seed` make reproducible both TensorFlow and numpy operations inside the model);
* all models support any precision (tested `float32` and `float64`);
* configure metrics to display during learning (which ones, frequency, format etc.);
* easy to resume training (note that changing parameters other than placeholders or python-level parameters (such as `batch_size`, `learning_rate`, `momentum`, `sample_v_states` etc.) between `fit` calls have no effect as this would require altering the computation graph, which is not yet supported; **however**, one can build model with new desired TF graph, and initialize weights and biases from old model by using `init_from` method);
* *visualization*: apart from TensorBoard, there also plenty of python routines to display images, learned filters, confusion matrices etc and more.

## Examples
### #1 RBM MNIST: [script](examples/rbm_mnist.py), [notebook](https://nbviewer.jupyter.org/github/monsta-hd/boltzmann-machines/blob/master/notebooks/rbm_mnist.ipynb)
Train Bernoulli RBM with 1024 hidden units on MNIST dataset and use it for classification.

| <div align="center">algorithm</div> | test error, % |
| :--- | :---: |
| RBM features + k-NN | **2.88** |
| RBM features + Logistic Regression | **1.83** |
| RBM features + SVM | **1.80** |
| RBM + discriminative fine-tuning | **1.27** |

<p float="left">
  <img src="img/rbm_mnist/filters.png" width="244" />
  <img src="img/rbm_mnist/filters_finetuned.png" width="244" /> 
  <img src="img/rbm_mnist/confusion_matrix.png" width="288" />
</p>

Another simple experiment illustrates main idea of *one-shot learning* approach proposed in [**[2]**](#2): to train generative neural network (RBM or DBM) on large corpus of unlabeled data and after that to *fine-tune* model only on limited amount of labeled data. Of course, in [**[2]**](#2) they do much more complex things than simply pre-training RBM or DBM, but the difference is already noticeable:

| number of labeled data pairs (train + val) | RBM + fine-tuning | random initialization | gain |
| :---: | :---: | :---: | :---: |
| 60k (55k + 5k) | 98.73% | 98.20% | **+0.53%** |
| 10k (9k + 1k) | 97.27% | 94.73% | **+2.54%** |
| 1k (900 + 100) | 93.65% | 88.71% | **+4.94%** |
| 100 (90 + 10) | 81.70% | 76.02% | **+5.68%** |

How to reproduce this table see [here](docs/rbm_discriminative.md). 
In these experiments only RBM was tuned to have high pseudo log-likelihood on a held-out validation set.
Even better results can be obtained if one will tune MLP and other classifiers.

---

### #2 DBM MNIST: [script](examples/dbm_mnist.py), [notebook](https://nbviewer.jupyter.org/github/monsta-hd/boltzmann-machines/blob/master/notebooks/dbm_mnist.ipynb)
Train 784-512-1024 Bernoulli DBM on MNIST dataset with pre-training and:
* use it for classification;
* generate samples after training;
* estimate partition function using AIS and average ELBO on the test set.

| algorithm | # intermediate distributions | proposal (p<sub>0</sub>) | logẐ | log(Ẑ &plusmn; &#963;<sub>Z</sub>) | avg. test ELBO | tightness of test ELBO |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [**[1]**](#1) | 20'000 | base-rate? [**[5]**](#5) | 356.18 | 356.06, 356.29 | **-84.62** | about **0.5** nats |
| this example | 200'000 | uniform | 1040.39 | 1040.18, 1040.58 | **-86.37** | &mdash; |
| this example | 20'000 | uniform | 1040.58 | 1039.93, 1041.03 | **-86.59** | &mdash; |

One can probably get better results by tuning the model slightly more. 
Also couple of nats could have been lost because of single-precision (for both training and AIS estimation).

<p float="left">
  <img src="img/dbm_mnist/rbm1.png" width="258" />
  <img src="img/dbm_mnist/W1_joint.png" width="258" /> 
  <img src="img/dbm_mnist/W1_finetuned.png" width="258" />
</p>
<p float="left">
  <img src="img/dbm_mnist/rbm2.png" width="258" />
  <img src="img/dbm_mnist/W2_joint.png" width="258" /> 
  <img src="img/dbm_mnist/W2_finetuned.png" width="258" />
</p>
<p float="left">
  <img src="img/dbm_mnist/mnist.png" width="253" />
  <img src="img/dbm_mnist/samples.png" width="258" /> 
  <img src="img/dbm_mnist/samples.gif" width="273" />
</p>

| number of labeled data pairs (train + val) | DBM + fine-tuning | random initialization | gain |
| :---: | :---: | :---: | :---: |
| 60k (55k + 5k) | 98.68% | 98.28% | **+0.40%** |
| 10k (9k + 1k) | 97.11% | 94.50% | **+2.61%** |
| 1k (900 + 100) | 93.54% | 89.14% | **+4.40%** |
| 100 (90 + 10) | 83.79% | 76.24% | **+7.55%** |

How to reproduce this table see [here](docs/dbm_discriminative.md).

Again, MLP is not tuned. With tuned MLP and slightly more tuned generative model in [**[1]**](#1) they achieved **0.95%** error on full test set.
<br>
Performance on full training set is slightly worse compared to RBM because of harder optimization problem + possible vanishing gradients. Also because the optimization problem is harder, the gain when not much datapoints are used is typically larger.
<br>
Large number of parameters is one of the most crucial reasons why one-shot learning is not (so) successful by utilizing deep learning only. Instead, it is much better to combine deep learning and hierarchical Bayesian modeling by putting HDP prior over units from top-most hidden layer as in [**[2]**](#2).

---

### #3 DBM CIFAR-10 "Naïve": [script](examples/dbm_cifar_naive.py), [notebook](https://nbviewer.jupyter.org/github/monsta-hd/boltzmann-machines/blob/master/notebooks/dbm_cifar_naive.ipynb)

(Simply) train 3072-5000-1000 Gaussian-Bernoulli-Multinomial DBM on "smoothed" CIFAR-10 dataset (with 1000 least
significant singular values removed, as suggested in [**[3]**](#3)) with pre-training and:
* generate samples after training;
* use pre-trained Gaussian RBM (G-RBM) for classification.

<p float="left">
  <img src="img/dbm_cifar_naive/grbm.png" width="194" />
  <img src="img/dbm_cifar_naive/W1_joint.png" width="194" />
  <img src="img/dbm_cifar_naive/mrbm.png" width="194" />
  <img src="img/dbm_cifar_naive/W2_joint.png" width="194" />
</p>
<p float="left">
  <img src="img/dbm_cifar_naive/cifar10_smoothed.png" width="255" />
  <img src="img/dbm_cifar_naive/samples.png" width="255" />
  <img src="img/dbm_cifar_naive/samples.gif" width="275" />
</p>

Despite poor-looking G-RBM features, classification performance after discriminative fine-tuning is much larger than reported backprop from random initialization [**[3]**](#3), and is 5% behind best reported result using RBM (with twice larger number of hidden units). Note also that G-RBM is *modified* for DBM pre-training ([notes](#tex-notes) or [**[1]**](#1) for details):

| <div align="center">algorithm</div> | test accuracy, % |
| :--- | :---: |
| *Best known MLP w/o data augmentation*: 8 layer ZLin net [**[6]**](#6) | **69.62** |
| *Best known method using RBM (w/o data augmentation?)*: 10k hiddens + fine-tuning [**[3]**](#3) | **64.84** |
| Gaussian RBM + discriminative fine-tuning (this example) | **59.78** |
| Pure backprop 3072-5000-10 on smoothed data (this example) | **58.20** |
| Pure backprop 782-10k-10 on PCA whitened data [**[3]**](#3) | **51.53** |

<p float="left">
  <img src="img/dbm_cifar_naive/grbm.png" width="246" />
  <img src="img/dbm_cifar_naive/grbm_finetuned.png" width="246" /> 
  <img src="img/dbm_cifar_naive/grbm_confusion_matrix.png" width="285" />
</p>

---

### #4 DBM CIFAR-10: [script](examples/dbm_cifar.py), [notebook](https://nbviewer.jupyter.org/github/monsta-hd/boltzmann-machines/blob/master/notebooks/dbm_cifar.ipynb)

Train 3072-7800-512 G-B-M DBM with pre-training on CIFAR-10, 
augmented (x10) using shifts by 1 pixel in all directions and horizontal mirroring and using more advanced training of G-RBM which is initialized from pre-trained 26 small RBM on patches of images, as in [**[3]**](#3).
<br>
Notice how some of the particles are already resemble natural images of horses, cars etc. and note that the model is trained only on augmented CIFAR-10 (490k images), compared to 4M images that were used in [**[2]**](#2).

<p float="left">
  <img src="img/dbm_cifar/rbm_small_0.png" width="194" />
  <img src="img/dbm_cifar/rbm_small_2.png" width="194" /> 
  <img src="img/dbm_cifar/rbm_small_10.png" width="194" />
  <img src="img/dbm_cifar/rbm_small_20.png" width="194" />
</p>
<p float="left">
  <img src="img/dbm_cifar/grbm.png" width="255" />
  <img src="img/dbm_cifar/mrbm.png" width="255" /> 
  <img src="img/dbm_cifar/samples.png" width="255" hspace="10" /> 
</p>
<p float="left">
  <img src="img/dbm_cifar/W1_joint.png" width="255" />
  <img src="img/dbm_cifar/W2_joint.png" width="255" /> 
  <img src="img/dbm_cifar/samples.gif" width="275" /> 
</p>

I also trained for longer with
```bash
python dbm_cifar.py --small-l2 2e-3 --small-epochs 120 --small-sparsity-cost 0 \
                    --increase-n-gibbs-steps-every 20 --epochs 80 72 200 \
                    --l2 2e-3 0.01 1e-8 --max-mf-updates 70
```
While all RBMs have nicer features, this means that they overfit more than previously, and thus overall DBM performance is slightly worse.

<p float="left">
  <img src="img/dbm_cifar2/rbm_small_0.png" width="194" />
  <img src="img/dbm_cifar2/rbm_small_2.png" width="194" /> 
  <img src="img/dbm_cifar2/rbm_small_10.png" width="194" />
  <img src="img/dbm_cifar2/rbm_small_20.png" width="194" />
</p>
<p float="left">
  <img src="img/dbm_cifar2/grbm.png" width="255" />
  <img src="img/dbm_cifar2/mrbm.png" width="255" /> 
  <img src="img/dbm_cifar2/samples.png" width="255" hspace="10" /> 
</p>
<p float="left">
  <img src="img/dbm_cifar2/W1_joint.png" width="255" />
  <img src="img/dbm_cifar2/W2_joint.png" width="255" /> 
  <img src="img/dbm_cifar2/samples.gif" width="275" /> 
</p>

The training with all pre-trainings takes quite a lot of time, but once trained, these nets can be used for other (similar) datasets/tasks.
<br>
Discriminative performance of Gaussian RBM now is very close to state of the art (having 7800 vs. 10k hidden units), and data augmentation given another 4% of test accuracy:

| <div align="center">algorithm</div> | test accuracy, % |
| :--- | :---: |
| Gaussian RBM + discriminative fine-tuning + augmentation (this example) | **68.11** |
| *Best known method using RBM (w/o data augmentation?)*: 10k hiddens + fine-tuning [**[3]**](#3) | **64.84** |
| Gaussian RBM + discriminative fine-tuning (this example) | **64.38** |
| Gaussian RBM + discriminative fine-tuning (example [#3](#3-dbm-cifar-10-naïve-script-notebook)) | **59.78** |

How to reproduce this table see [here](docs/grbm_discriminative.md).

<p float="left">
  <img src="img/dbm_cifar2/grbm.png" width="246" />
  <img src="img/dbm_cifar2/grbm_no_aug_finetuned.png" width="246" /> 
  <img src="img/dbm_cifar2/grbm_confusion_matrix.png" width="285" />
</p>

---

### How to use examples
Use **script**s for training models from scratch, for instance
```
$ python rbm_mnist.py -h

(...)

usage: rbm_mnist.py [-h] [--gpu ID] [--n-train N] [--n-val N]
                    [--data-path PATH] [--n-hidden N] [--w-init STD]
                    [--vb-init] [--hb-init HB] [--n-gibbs-steps N [N ...]]
                    [--lr LR [LR ...]] [--epochs N] [--batch-size B] [--l2 L2]
                    [--sample-v-states] [--dropout P] [--sparsity-target T]
                    [--sparsity-cost C] [--sparsity-damping D]
                    [--random-seed N] [--dtype T] [--model-dirpath DIRPATH]
                    [--mlp-no-init] [--mlp-l2 L2] [--mlp-lrm LRM [LRM ...]]
                    [--mlp-epochs N] [--mlp-val-metric S] [--mlp-batch-size N]
                    [--mlp-save-prefix PREFIX]

optional arguments:
  -h, --help            show this help message and exit
  --gpu ID              ID of the GPU to train on (or '' to train on CPU)
                        (default: 0)
  --n-train N           number of training examples (default: 55000)
  --n-val N             number of validation examples (default: 5000)
  --data-path PATH      directory for storing augmented data etc. (default:
                        ../data/)
  --n-hidden N          number of hidden units (default: 1024)
  --w-init STD          initialize weights from zero-centered Gaussian with
                        this standard deviation (default: 0.01)
  --vb-init             initialize visible biases as logit of mean values of
                        features, otherwise (if enabled) zero init (default:
                        True)
  --hb-init HB          initial hidden bias (default: 0.0)
  --n-gibbs-steps N [N ...]
                        number of Gibbs updates per weights update or sequence
                        of such (per epoch) (default: 1)
  --lr LR [LR ...]      learning rate or sequence of such (per epoch)
                        (default: 0.05)
  --epochs N            number of epochs to train (default: 120)
  --batch-size B        input batch size for training (default: 10)
  --l2 L2               L2 weight decay coefficient (default: 1e-05)
  --sample-v-states     sample visible states, otherwise use probabilities w/o
                        sampling (default: False)
  --dropout P           probability of visible units being on (default: None)
  --sparsity-target T   desired probability of hidden activation (default:
                        0.1)
  --sparsity-cost C     controls the amount of sparsity penalty (default:
                        1e-05)
  --sparsity-damping D  decay rate for hidden activations probs (default: 0.9)
  --random-seed N       random seed for model training (default: 1337)
  --dtype T             datatype precision to use (default: float32)
  --model-dirpath DIRPATH
                        directory path to save the model (default:
                        ../models/rbm_mnist/)
  --mlp-no-init         if enabled, use random initialization (default: False)
  --mlp-l2 L2           L2 weight decay coefficient (default: 1e-05)
  --mlp-lrm LRM [LRM ...]
                        learning rate multipliers of 1e-3 (default: (0.1,
                        1.0))
  --mlp-epochs N        number of epochs to train (default: 100)
  --mlp-val-metric S    metric on validation set to perform early stopping,
                        {'val_acc', 'val_loss'} (default: val_acc)
  --mlp-batch-size N    input batch size for training (default: 128)
  --mlp-save-prefix PREFIX
                        prefix to save MLP predictions and targets (default:
                        ../data/rbm_)
```
or download pretrained ones with default parameters using `models/fetch_models.sh`, 
</br>
and check **notebook**s for corresponding inference / visualizations etc.
Note that training is skipped if there is already a model in `model-dirpath`, and similarly for other experiments (you can choose different location for training another model).

---

### Memory requirements
* GPU memory: at most 2-3 GB for each model in each example, and it is always possible to decrease batch size and number of negative particles;
* RAM: at most 11GB (to run last example, features from Gaussian RBM are in `half` precision) and (much) lesser for other examples.

---

## Download models and stuff
All models from all experiments can be downloaded by running `models/fetch_models.sh` or manually from [Google Drive](https://drive.google.com/open?id=1jFsh4Jh3s41B-_hPHe_VS9apkMmIWiNy).
<br>
Also, you can download additional data (fine-tuned models' predictions, fine-tuned weights, means and standard deviations for datasets for examples [#3](#3-dbm-cifar-10-naïve-script-notebook), [#4](#4-dbm-cifar-10-script-notebook)) using `data/fetch_additional_data.sh`

## TeX notes
Check also my supplementary [notes](tex/notes.pdf) (or [dropbox](https://www.dropbox.com/s/7pk4yeixkxogcem/bm_notes.pdf?dl=0)) with some historical outlines, theory, derivations, observations etc.

## How to install
By default, the following commands install (among others) **tensorflow-gpu~=1.3.0**. If you want to install tensorflow without GPU support, replace corresponding line in [requirements.txt](requirements.txt). If you have already tensorflow installed, comment that line.
```bash
git clone https://github.com/monsta-hd/boltzmann-machines.git
cd boltzmann-machines
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

## Possible future work
* add stratification;
* add t-SNE visualization for extracted features;
* generate half MNIST digit conditioned on the other half using RBM;
* implement Centering [**[7]**](#7) for all models;
* implement classification RBMs/DBMs?;
* implement ELBO and AIS for arbitrary DBM (again, visible and topmost hidden units can be analytically summed out);
* optimize input pipeline e.g. use queues instead of `feed_dict` etc.

## Contributing
Feel free to improve existing code, documentation or implement new feature (including those listed in [Possible future work](#possible-future-work)). Please open an issue to propose your changes if they are big enough.

## References
**[1]**<a name="1"></a> R. Salakhutdinov and G. Hinton. *Deep boltzmann machines.* In: Artificial Intelligence and
Statistics, pages 448–455, 2009. [[PDF](http://proceedings.mlr.press/v5/salakhutdinov09a/salakhutdinov09a.pdf)]

**[2]**<a name="2"></a> R. Salakhutdinov, J. B. Tenenbaum, and A. Torralba. *Learning with hierarchical-deep models.* IEEE transactions on pattern analysis and machine intelligence, 35(8):1958–1971, 2013. [[PDF](https://www.cs.toronto.edu/~rsalakhu/papers/HD_PAMI.pdf)]

**[3]**<a name="3"></a> A. Krizhevsky and G. Hinton. *Learning multiple layers of features from tiny images.* 2009. [[PDF](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)]

**[4]**<a name="4"></a> G. Hinton. *A practical guide to training restricted boltzmann machines.* Momentum, 9(1):926,
2010. [[PDF](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)]

**[5]**<a name="5"></a> R. Salakhutdinov and I. Murray. *On the quantitative analysis of Deep Belief Networks.* In
A. McCallum and S. Roweis, editors, Proceedings of the 25th Annual International Conference
on Machine Learning (ICML 2008), pages 872–879. Omnipress, 2008 [[PDF](http://homepages.inf.ed.ac.uk/imurray2/pub/08dbn_ais/dbn_ais.pdf)]

**[6]**<a name="6"></a> Lin Z, Memisevic R, Konda K. *How far can we go without convolution: Improving fully-connected networks*, ICML 2016. [[arXiv](https://arxiv.org/abs/1511.02580)]

**[7]**<a name="7"></a> G. Montavon and K.-R. Müller. *Deep boltzmann machines and the centering trick.* In Neural
Networks: Tricks of the Trade, pages 621–637. Springer, 2012. [[PDF](http://gregoire.montavon.name/publications/montavon-lncs12.pdf)]
