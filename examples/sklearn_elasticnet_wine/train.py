# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    f1 = f1_score(actual, pred, average="macro")
    precision = precision_score(actual, pred, average="macro")
    recall = recall_score(actual, pred, average="macro")
    return f1, precision, recall


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s",
            e,
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = LogisticRegression(
            max_iter=max_iter,
            l1_ratio=l1_ratio,
            random_state=42,
            penalty="elasticnet",
            multi_class="multinomial",
            solver="saga",
        )
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (f1, precision, recall) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (max_iter=%f, l1_ratio=%.2f):" % (max_iter, l1_ratio))
        print("F1: %s" % f1)
        print("Precision: %s" % precision)
        print("Recall: %s" % recall)

        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(lr, "model")
