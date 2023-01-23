# library doc string
"""
This module holds several functions used to analyse and predict Customer Churn

Author: Mohamed Mejri
Date: Jan 2023
"""

# import libraries
import os
import logging
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import RocCurveDisplay, classification_report
from constants import cat_columns, keep_cols

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        logging.info("file path = '%s'", pth)
        df = pd.read_csv(pth)
        logging.info("SUCCESS: Data imported successfully")
        if 'Unnamed: 0' in df.columns:
            df.drop(columns='Unnamed: 0', inplace=True)
        logging.info("%s \n shape = %s", df.head(), df.shape)
        return df
    except FileNotFoundError:
        logging.error("ERROR: File not found")
        return None


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    def hist_plot(column, figsize=(20, 10)):
        try:
            plt.figure(figsize=figsize)
            logging.info("producing hist plot for column '%s'", column)
            _ = df[column].hist()
            logging.info("SUCCESS: '%s' hist plot produced", column)
            fig_path = './images/eda/' + column + '_hist.png'
            logging.info("Saving the figure to '%s'", fig_path)
            plt.savefig(fig_path)
            plt.close()
            logging.info(
                "SUCCESS: hist_plot saved successfully to '%s'",
                fig_path)
        except BaseException:
            logging.error("Failed to save '%s' plot", column)
    try:
        assert isinstance(df, pd.DataFrame)
        logging.info(
            "Checking null values per columns \n %s",
            df.isnull().sum())
        logging.info("Columns description \n %s", df.isnull().sum())

        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        hist_plot('Churn')
        hist_plot('Customer_Age')

        plt.figure(figsize=(20, 10))
        logging.info('Producing Marital_Status count plot...')
        df.Marital_Status.value_counts('normalize').plot(kind='bar')
        logging.info('Saving Marital_Status count plot...')
        path = './images/eda/Marital_Status.png'
        plt.savefig(path)
        logging.info("SUCCESS: Marital_Status count plot saved at '%s'", path)
        plt.close()

        plt.figure(figsize=(20, 10))
        logging.info('Producing Total_Trans_Ct histplot...')
        sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
        logging.info('Saving Total_Trans_Ct histplot...')
        path = './images/eda/Total_Trans_Ct.png'
        plt.savefig(path)
        logging.info("SUCCESS: Total_Trans_Ct histplot saved at '%s'", path)
        plt.close()

        plt.figure(figsize=(20, 10))
        logging.info('Producing Correlation heatmap...')
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        logging.info('Saving Correlation heatmap...')
        path = './images/eda/Correlation_heat_map.png'
        plt.savefig(path)
        logging.info("SUCCESS: Correlation heatmap saved at '%s'", path)
        plt.close()

    except AssertionError:
        logging.error('ERROR: df type should be pd.DataFrame')


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
             used for naming variables or index y column]

    output:
            df: pandas dataframe with new encoded columns
    '''
    def encode(col):
        logging.info('Encoding column %s', col)
        col_list = []
        col_groups = df.groupby(col).mean()['Churn']

        for val in df[col]:
            col_list.append(col_groups.loc[val])
        col_name = col + '_' + response
        df[col_name] = col_list
        
    for cat_col in category_lst:
        encode(cat_col)
    return df


def perform_feature_engineering(df, response=None):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that 
              could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df['Churn']
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    logging.info('The selected features are %s', keep_cols)
    logging.info(X.head())
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    def generate_reports(
            y_test,
            y_train,
            y_test_preds,
            y_train_preds,
            model_name):
        logging.info('%s results', model_name)

        try:
            logging.info('Test results')
            test_report = classification_report(y_test, y_test_preds)
            logging.info(test_report)

            logging.info('Train results')
            train_report = classification_report(y_train, y_train_preds)
            logging.info(train_report)
        except Exception as err:
            logging.exception(
                'ERROR: failed to generate classification report')
        try:
            logging.info('Generating report images for %s', model_name)
            plt.rc('figure', figsize=(6, 5))
            plt.text(0.01, 1.25, str(model_name + 'Test'),
                     {'fontsize': 10}, fontproperties='monospace')
            # approach improved by OP -> monospace!
            plt.text(
                0.01, 0.05, str(test_report), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(0.01, 0.6, str(model_name + 'Test'),
                     {'fontsize': 10}, fontproperties='monospace')
            # approach improved by OP -> monospace!
            plt.text(
                0.01, 0.7, str(train_report), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.axis('off')
            image_path = './images/results/' + model_name + 'report.png'
            plt.savefig(image_path)
            plt.close()
            logging.info('SUCCESS: report image saved at %s', image_path)
        except BaseException:
            logging.exception(
                'ERROR: Failed to generate report as images for %s',
                model_name)

    # generating reports for Random Forest's results
    generate_reports(
        y_test,
        y_train,
        y_test_preds_rf,
        y_train_preds_rf,
        'Random_Forest')

    # generating reports for Logistic Regression's results
    generate_reports(
        y_test,
        y_train,
        y_test_preds_lr,
        y_train_preds_lr,
        'Logistic_Regression')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    logging.info('Intializing shap explainer')
    explainer = shap.TreeExplainer(model)
    logging.info('Producing shap values')
    shap_values = explainer.shap_values(X_data)
    logging.info('Producing shap summary plot')
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    logging.info("Saving the shap plot under %s", output_pth)
    plt.savefig(output_pth + 'shap.png')
    plt.close()

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    def roc_plots(fig_name='roc_plot'):
        path = './images/results/'
        plt.figure(figsize=(15, 8))
        plt.cla()
        ax = plt.gca()
        logging.info('Producing roc for Random Forest...')
        _ = RocCurveDisplay.from_estimator(
            cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
        logging.info('Producing roc for Logistic Regression...')
        _ = RocCurveDisplay.from_estimator(
            lrc, X_test, y_test, ax=ax, alpha=0.8)
        logging.info('Saving figure...')
        plt.savefig(path + fig_name + '.png')
        plt.close()

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train, y_test, y_train_preds_lr, y_train_preds_rf,
        y_test_preds_lr, y_test_preds_rf)

    # Saving roc_plots
    try:
        logging.info('Generating roc plots')
        roc_plots()
        logging.info(
            "SUCCESS: Roc plots generated and saved under './images/results/'")
    except BaseException:
        logging.exception('ERROR: Failed to generate and save roc plots')

    # save best model
    try:
        logging.info("Saving models under './models/")
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        logging.info('SUCCESS: Models saved successfully')
    except BaseException:
        logging.exception("ERROR: Couldn't save the models")

    try:
        logging.info('Generating feature importance plots')
        feature_importance_plot(
            cv_rfc.best_estimator_,
            X_test,
            './images/results/')
        logging.info(
            "SUCCESS: Feature importance plots saved successfully under './images/results/'")
    except BaseException:
        logging.exception(
            'ERROR: Failed to plot and save feature importance plots')


if __name__ == "__main__":
    print("It works!")
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    df = encoder_helper(df, cat_columns)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)
