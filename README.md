# instaffo

Creating end-to-end  solution to run ML model for matching jobs and talents

# Instaffo Task

This project contains a solution for matching talents to jobs using a logistic regression model. The solution includes a class `Search` with methods to match a single talent to a job and multiple talents to multiple jobs.

# Project directory structure

instaffo_task/instaffo_match

--- data/
---- data.json

--- models/
---- logistic_regression_model.pkl

--- notebooks/
---- instaffo.ipynb

--- scripts/
---- init.py
---- search.py

--- utils/
---- init.py
---- search_utils.py

--- README.md
--- requirements.txt

# Project implementation steps

1) business understanding
2) data understanding
3) preprocessing
4) feature engineering
5) data analytics
6) data exploration
7) ML modeling
8) model evaluation
9) creating github environment
10) pushing model and other relevant files to the git
11) create clean code & docstring and documentation

# Directory and File Descriptions

- **data/**:
  - `data.json`: Example data file (here same data we received for task).

- **models/**:
  - `logistic_regression_model.pkl`: Pre-trained logistic regression model saved using `joblib`. 

- **notebooks/**:
  - `instaffo.ipynb`: Jupyter notebook showing implementation steps from above.

- **scripts/**:
  - `__init__.py`: An empty file to make this directory a Python package.
  - `search.py`: Contains the `Search` class which includes methods to match a single talent to a job and multiple talents to multiple jobs.
  - `run_model.py`: Script to demonstrate how to load the model and use it to make predictions.

- **utils/**:
  - `__init__.py`: An empty file to make this directory a Python package.
  - `search_utils.py`: Contains utility functions required by the `Search` class

- **README.md**:
  - This file. Provides an overview of the project, its structure, setup instructions, and usage examples.

- **requirements.txt**:
  - Lists all the Python dependencies required to run the project. Used to set up the environment quickly.

# Instructions for usage (Windows)

git clone https://<PAT>@github.com/mvuckovic70/instaffo_task.git (replace PAT with secret token provided by mail)
cd instaffo_task
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
python scripts/search.py

# Note:

Within searc.py script, there is an exampple of talents and job dicts for usage.
This is for demo purposes only.
For production usage, more suitable approach would using some api (flask or fastapi).
