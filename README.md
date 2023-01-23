# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is a project related to the Udacity's Machine Learing DevOps Engineer Nanodegree.
The objective of this project is to put into practice what we have learned during the nanodegree's first course and predict customer churn.

This project follows coding(PEP8) and engineering best practices for implementing software (modular, documented and tested).

## Files and data description
<!-- Overview of the files and data present in the root directory.  -->
### Data
This dataset was taken from <a link=https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers >Kaggle</a>
It consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc.
### Files
- `Guide.ipynb`: the Udacity's guide notebook
- `churn_notebook.ipynb`: The provided starter notebook which is messy
- `churn_library.py`: the clean script version of the starter notebook
- `constants.py`: it contains constants used in the project and for testing
- `churn_script_logging_and_tests.py`: the testing file 


## Install Dependencies
You can install the dependecies by running the following command
```
python -m pip install -r requirements.txt
```

## Running Files
After cloning the repository, you can run the project with the following command:
```
python churn_library.py
```
To run the unit tests you can run the following commad
```
python churn_script_logging_and_tests.py
```
you expect to have the logs of the testing and training under the ```./logs/``` directory.

You will find the trained models under ```./models/``` and the generated images of the EDA and models performance can be found under ```./images/``` 

