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

jp:
	sudo jupyter nbextension enable --py --sys-prefix widgetsnbextension
	jupyter notebook

.PHONY: test clean jp
