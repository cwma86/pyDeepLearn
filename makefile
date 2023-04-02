
SHELL := /bin/bash
pkg:
	@python setup.py bdist_wheel --universal

env:
	( \
		python3 -m pip install virtualenv; \
		mkdir -p python-venv; \
		python3 -m venv python-venv; \
		source python-venv/bin/activate; \
		pip install -r requirements.txt; \
	)
	
check:
	@python3 -m unittest discover -s pyDeepLearn

clean:
	@rm -rf build dist *.egg-info ./pyDeepLearn/__pycache__ ./pyDeepLearn/tests/__pycache__