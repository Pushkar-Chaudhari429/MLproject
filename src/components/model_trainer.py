from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import sys
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
            try:
                logging.info("Splitting training and testing input data")
                X_train,y_train,X_test,y_test=(
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
                )
                models={
                    "Random Forest":RandomForestRegressor(),
                    "Decision Tree":DecisionTreeRegressor(),
                    "Gradient Boosting":GradientBoostingRegressor(),
                    "Linear Regression":LinearRegression(),
                    "XGBRegressor":XGBRegressor(),
                    "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor":AdaBoostRegressor(),
                    "KNeighbors Regressor":KNeighborsRegressor()
                }

                params={
                     "Decision Tree":{
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        'splitter':['best','random'],
                        'max_features':['sqrt','log2']
                    },
                    "Random Forest":{
                        'n_estimators':[8,16,32,64,128,256]
                    },  
                    "Gradient Boosting":{
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        'n_estimators':[8,16,32,64,128,256]
                    },
                    "Linear Regression":{
                         'fit_intercept':[True,False],
                         'copy_X':[True,False],
                    },
                    "XGBRegressor":{
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators':[8,16,32,64,128,256]
                    },
                    "CatBoosting Regressor":{
                        'depth':[6,8,10],
                        'learning_rate':[.1,.01,.05,.001],
                        'iterations':[30,50,100]
                    },
                    "AdaBoost Regressor":{
                        'learning_rate':[.1,.01,0.5,.001],
                        'n_estimators':[8,16,32,64,128,256]
                    },
                    "KNeighbors Regressor":{
                        'n_neighbors':[5,7,9,11],
                        'weights':['uniform','distance'],
                        'algorithm':['ball_tree','kd_tree','brute']
                    }
                }

                model_report=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

                best_model_score=max(sorted(model_report.values()))
                best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
                best_model=models[best_model_name]
                if best_model_score<0.6:
                    raise CustomException("No best model found")
                logging.info("Best model found on both training and testing dataset")

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
                prediction=best_model.predict(X_test)
                final_score=r2_score(y_test,prediction)
                final=[final_score,best_model_name]
                return final
            except Exception as e:
                raise CustomException(e,sys)

