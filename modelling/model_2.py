import xgboost
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd


class PriceHouses_model_xgboost():
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

    def fit_model_randomized_grid_searchCV(self):
        regressor = xgboost.XGBRegressor()    
        
        hyperparameters = {"n_estimators": [100,500,800,1200],
                      "learning_rate": [0.05,0.1, 0.18, 0.25, 0.3],
                      "booster": ["gbtree", "gblinear"],
                      "base_score" : [0.25,0.5, 0.6, 0.85, 1]
                      }

        random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameters,
            cv=5, n_iter=10,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
        predicts = random_cv.fit(self.X_train, self.y_train).predict(self.X_test)
        preds = pd.DataFrame(predicts)
        preds.to_csv(".\outputs\\predsxgboost.csv")
        return predicts
 
