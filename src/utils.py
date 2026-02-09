import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}
        best_models = {}
        # Iterate through models dict items to get name and estimator
        for model_name, model in models.items():
            # get parameter grid for this model (fallback to empty dict)
            para = param.get(model_name, {})
            # run grid search if param grid provided, otherwise use model as-is
            if para:
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                # no hyperparameter tuning for this model
                best_model = model
                best_model.fit(X_train, y_train)

            # evaluate
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            best_models[model_name] = best_model

        # return both the performance report and the dict of trained estimators
        return report, best_models
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)