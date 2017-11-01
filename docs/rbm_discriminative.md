```bash
# full
python rbm_mnist.py --mlp-save-prefix '../data/rbm_full_' --mlp-lrm 0.1 1.
python rbm_mnist.py --mlp-no-init --mlp-save-prefix '../data/rbm_full_no_init_' --mlp-lrm 1. 1.

# 10k
python rbm_mnist.py --n-train 9000 --n-val 1000 --mlp-save-prefix '../data/rbm_10k_' --mlp-lrm 0.01 1.
python rbm_mnist.py --n-train 9000 --n-val 1000 --mlp-no-init --mlp-save-prefix '../data/rbm_10k_no_init_' --mlp-lrm 1. 1.

# 1k
python rbm_mnist.py --mlp-batch-size 32 --mlp-val-metric 'val_acc' --n-train 900 --n-val 100 --mlp-save-prefix '../data/rbm_900_100_' --mlp-lrm 0.01 1.
python rbm_mnist.py --mlp-batch-size 32 --mlp-val-metric 'val_acc' --n-train 900 --n-val 100 --mlp-no-init --mlp-save-prefix '../data/rbm_900_100_no_init_' --mlp-lrm 1. 1.

# 100
python rbm_mnist.py --mlp-batch-size 32 --mlp-val-metric 'val_loss' --mlp-epochs 1000 --n-train 90 --n-val 10 --mlp-save-prefix '../data/rbm_90_10_' --mlp-lrm 0.01 1.
python rbm_mnist.py --mlp-batch-size 32 --mlp-val-metric 'val_loss' --mlp-epochs 1000 --n-train 90 --n-val 10 --mlp-no-init --mlp-save-prefix '../data/rbm_90_10_no_init_' --mlp-lrm 1. 1.
```
