#!/usr/bin/env bash
wget "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
wget "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
wget "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
wget "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
mkdir -p 'mnist'
mv train-images-idx3-ubyte mnist
mv train-labels-idx1-ubyte mnist
mv t10k-images-idx3-ubyte mnist
mv t10k-labels-idx1-ubyte mnist
