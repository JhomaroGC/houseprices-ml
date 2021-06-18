from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class PriceHouses_model():
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
    
    def fit_model_grid_searchCV(self):
        parameters = {"n_estimators": [1,2],
                      "criterion": ["mse", "mae"],
                      "max_depth": [1, 2, 3]}
        rfr = RandomForestRegressor()
        clf = GridSearchCV(rfr, parameters)
        clf.fit(self.X_train, self.y_train)
        print(f"Mejores parámetros del modelo después de la búsqueda: {clf.best_params_}")
        predicts = clf.predict(self.X_test)
        preds = pd.Dataframe(predicts)
        preds.to_csv(".\outputs\\preds.txt")
        print("Predicciones del modelo: \n", predicts)
        return predicts
    
