test: clean
	nosetests

clean:
	find . -name '*.pyc' -type f -delete
	rm -f './random_state.json'
	rm -f 'hdp_dbm/utils/random_state.json'

.PHONY: test clean
