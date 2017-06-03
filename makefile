# needed to be able to call other targets within given one:
THIS_FILE := $(lastword $(MAKEFILE_LIST))

test: 
	@$(MAKE) -f $(THIS_FILE) clean
	nosetests
	@$(MAKE) -f $(THIS_FILE) clean

clean:
	find . -name '*.pyc' -type f -delete
	rm -f './random_state.json'
	rm -f 'hdp_dbm/utils/random_state.json'
	rm -rf 'test_rbm_1/'
	rm -rf 'test_rbm_2/'
	rm -rf 'test_rbm_3/'
	rm -rf 'hdp_dbm/test_rbm_1/'
	rm -rf 'hdp_dbm/test_rbm_2/'
	rm -rf 'hdp_dbm/test_rbm_3/'

LOGDIR := 'logs/'
tb:
	tensorboard --logdir=$(LOGDIR)

.PHONY: test clean tb
