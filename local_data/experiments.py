"""
Script to retrieve transferability experiments setting
(i.e. dataframe path, target classes, and task type)
"""

from local_data.constants import *


def get_experiment_setting(experiment):

    # Transferability for classification
    if experiment == "CFI":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "CFI.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "elevated blood pressure": 1, "stage 1 hypertension": 2, "stage 2 hypertension": 3}}

    else:
        setting = None
        print("Experiment not prepared...")

    return setting
