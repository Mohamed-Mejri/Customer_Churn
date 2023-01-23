"""
This module is used for logging and testing functions of the ./churn_library.py

Author: Mohamed Mejri
Date: Jan 2023
"""
import os
import logging
import pytest
import churn_library as cl
from constants import eda_files, cat_columns, all_columns, keep_cols



logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def import_data():
    return cl.import_data('./data/bank_data.csv')


@pytest.fixture
def perform_eda(import_data):
    data = import_data
    return cl.perform_eda(data)


@pytest.fixture
def prepare_df(import_data):
    data = import_data
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data


@pytest.fixture
def encoder_helper(prepare_df):
    data = prepare_df
    return cl.encoder_helper(data, cat_columns)


@pytest.fixture
def perform_feature_engineering(encoder_helper):
    data = encoder_helper
    return cl.perform_feature_engineering(data)


@pytest.fixture
def train_models(perform_feature_engineering):
    datum = perform_feature_engineering
    return cl.train_models(*datum)


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data = import_data
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        _ = perform_eda
        assert eda_files == os.listdir('./images/eda/')
        logging.info('Testing perform_eda: SUCCESS')
    except AssertionError:
        logging.error(
            'Testing perform_eda: Some expected figures were not found')
    except BaseException:
        logging.exception('Testing perform_eda: Unexpected error')


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        data = encoder_helper
        assert list(data.columns) == all_columns
        logging.info('Testing encoder_helper: SUCCESS')
    except AssertionError:
        logging.error(
            'Testing encoder_helper: Some expected columns were not found')


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        df_train, df_test, _, _ = perform_feature_engineering
        assert list(df_train.columns) == keep_cols
        assert list(df_test.columns) == keep_cols
        logging.info('Test perform_feature_engineering: SUCCESS')
    except AssertionError:
        logging.error(
            'Test perform_feature_engineering: Unexpected features selection')
    except BaseException:
        logging.exception('Test perform_feature_engineering: Unexpected error')


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        _ = train_models
        logging.info('Test train_models: SUCCESS')
    except BaseException:
        logging.exception('Test train_models: Unexpected Error')


if __name__ == "__main__":
    pytest.main(["churn_script_logging_and_tests.py"])
