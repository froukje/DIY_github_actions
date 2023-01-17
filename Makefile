# Oneshell means I can run multiple lines in a recipe in the same shell, so I don't have to
# chain commands together with semicolon
.ONESHELL:
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_BASE=$(conda info --base)
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

  quality_checks:
  	## quality checks: isort, black, pylint
	$(CONDA_ACTIVATE) water_quality_env
	isort .
	black .
	pylint --recursive=y .

  setup:
	## create conda env from .yaml file
	conda env create --file=water_quality_env.yaml
	$(CONDA_ACTIVATE) water_quality_env
	pre-commit install

  update:
	## update existing env from .yaml file
	conda env update --prune -f water_quality_env.yaml

  test:
	## unit tests
	$(CONDA_ACTIVATE) water_quality_env
	pytest tests/model_tests.py

  train:
	$(CONDA_ACTIVATE) water_quality_env
	echo $$(which python)
	## train model
	python3 train.py

  prediction:
	## make example prediction
	$(CONDA_ACTIVATE) water_quality_env
	echo $$(which python)
	python3 make_predictions.py
