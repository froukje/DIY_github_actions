**DIY_github_actions:**

**Objective:** Learn about GitHub Actions and create examples

https://github.com/DataTalksClub/project-of-the-week/blob/main/2023-01-11-github_actions-1.md

**Content:** Simple ml example to test github actions to predict water quality.

* The data is a .csv-file and can be downloaded from [kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
* The repository contains the following files:
	* water_model.py: This is the script, where model and training is defined
		* This script saves the model and the scaler
	* make_predictions.py: makes a sample prediction
 	* water_quality_env.yaml: yaml-file with needed packages to create a conda environment
	* tests/model_tests.py: some unit tests
	* Makefile: makes easy execution of different steps possible (e.g. setup, training, tests etc.)
	* pyproject.toml: configuration file for pylint
