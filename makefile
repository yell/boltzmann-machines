# need to be able to call other targets within given one
THIS_FILE := $(lastword $(MAKEFILE_LIST))

test:
	@$(MAKE) -f $(THIS_FILE) clean
	nosetests
	@$(MAKE) -f $(THIS_FILE) clean

clean:
	find . -name '*.pyc' -type f -delete
	rm -f 'random_state.json'
	rm -f 'hdm/utils/random_state.json'

data:
	./data/fetch_mnist.sh
	mv mnist data
	./data/fetch_cifar10.sh
	mv cifar-10-batches-py data

.PHONY: test clean data
