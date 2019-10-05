.PHONY: help clean install

.DEFAULT: help
help:
	@echo "make install"
	@echo "       install gym-quickcheck and dependencies in currently active environment"
	@echo "make clean"
	@echo "       clean all python build/compiliation files and directories"

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force {} +
	rm --force .coverage
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

install: clean
	pip install --upgrade pip
	pip install --upgrade setuptools
	pip install .
