```bash
# full
python dbm_mnist.py --n-train 55000 --n-val 5000 --mlp-save-prefix '../data/dbm_full_'
python dbm_mnist.py  --mlp-no-init --mlp-save-prefix '../data/dbm_full_no_init_' --mlp-lrm 1.

# 10k
python dbm_mnist.py --n-train 9000 --n-val 1000 --mlp-save-prefix '../data/dbm_10k_'
python dbm_mnist.py --n-train 9000 --n-val 1000 --mlp-no-init --mlp-save-prefix '../data/dbm_10k_no_init_' --mlp-lrm 1.

# 1k
python dbm_mnist.py --mlp-batch-size 32 --n-train 900 --n-val 100 --mlp-save-prefix '../data/dbm_900_100_'
python dbm_mnist.py --mlp-batch-size 32 --n-train 900 --n-val 100 --mlp-no-init --mlp-save-prefix '../data/dbm_900_100_no_init_' --mlp-lrm 1.

# 100
python dbm_mnist.py --mlp-batch-size 32 --mlp-val-metric 'val_loss' --mlp-epochs 1000 --n-train 90 --n-val 10 --mlp-save-prefix '../data/dbm_90_10_'
python dbm_mnist.py --mlp-batch-size 32 --mlp-val-metric 'val_loss' --mlp-epochs 1000 --n-train 90 --n-val 10 --mlp-no-init --mlp-save-prefix '../data/dbm_90_10_no_init_' --mlp-lrm 1.
```
