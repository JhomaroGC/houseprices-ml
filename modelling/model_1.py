from pandas.io.parsers import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class PriceHouses_model():
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
    
    def fit_model_grid_searchCV(self):
        parameters = {"n_estimators": [30,50,100],
                      "criterion": ["mse", "mae"],
                      "max_depth": [5, 10, 15]}
        rfr = RandomForestRegressor()
        clf = GridSearchCV(rfr, parameters)
        clf.fit(self.X_train, self.y_train)
        # print(f"Mejores parámetros del modelo después de la búsqueda: {clf.best_params_}")
        predicts = clf.predict(self.X_test)
        preds = pd.DataFrame(predicts)
        preds.to_csv(".\outputs\\preds.csv")
        return predicts
    
def save_submission():
    test = pd.read_csv("data\\test.csv")
    preds = pd.read_csv("outputs\\preds.csv")
    submission = pd.concat([test['Id'], preds], axis = 1)
    sub = pd.DataFrame(submission.values, columns = ['Id', 'Id_', 'SalePrice'])
    sub = sub.set_index("Id")
    sub[['SalePrice']].to_csv("outputs\\my_submission.csv")
    print("Predicción guardada exitosamente, lista para envío: \n")
    return sub[["SalePrice"]]