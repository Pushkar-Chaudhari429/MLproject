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

                model_report=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

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

