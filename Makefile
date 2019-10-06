.PHONY: help meta install clean test coverage

TARGET ?=
ifdef TARGET
install_instruction = -e .[$(TARGET)]
else
install_instruction = -e .
endif

.DEFAULT: help
help:
	@echo "make meta"
	@echo "       update version number and meta data"
	@echo "make install"
	@echo "       install gym-quickcheck and dependencies in currently active environment"
	@echo "make clean"
	@echo "       clean all python build/compiliation files and directories"
	@echo "make test"
	@echo "       run all tests"
	@echo "make coverage"
	@echo "       run all tests and produce coverage report"

meta:
	python meta.py `git describe --tags --abbrev=0`
	scripts/embedmd -w README.md

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
	pip install $(install_instruction)

test: install
	pip install -e .[test]
	pytest --verbose --color=yes .

coverage: install
	pip install -e .[test]
	pip install pytest-cov
	pytest --cov=gym_quickcheck --cov-report term-missing
