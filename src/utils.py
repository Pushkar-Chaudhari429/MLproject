import os
import sys
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
import dill
from sklearn.model_selection import GridSearchCV
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}

        for i in range(len(models)):
            model=list(models.values())[i]
            parameters=param[list(models.keys())[i]]
            grid=GridSearchCV(model,parameters,cv=3)
            grid.fit(X_train,y_train)
            model.set_params(**grid.best_params_)
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            score=r2_score(y_test,y_pred)
            report[list(models.keys())[i]]=score

            return report
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)