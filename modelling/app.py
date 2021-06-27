from data import PriceHousesData
from model_1 import PriceHouses_model, save_submission, save_submission_2
from model_2 import PriceHouses_model_xgboost


#Get sets for train an test, and df_ for save submissiion
X_train, X_test, y_train, df_submission = PriceHousesData("data\house-prices-advanced-regression-techniques.zip", "train.csv", "test.csv").preprocessing_all_data()

# Fit model_1 with gridsearchCV
# predictions = PriceHouses_model(X_train, y_train, X_test).fit_model_grid_searchCV()

#fit the model_2
pred = PriceHouses_model_xgboost(X_train, y_train, X_test).fit_model_randomized_grid_searchCV()
#save submission
submission = save_submission_2()
print(submission)