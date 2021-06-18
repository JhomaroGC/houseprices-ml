from data import PriceHousesData

X, y = PriceHousesData("data\house-prices-advanced-regression-techniques.zip", "train.csv", "test.csv").preprocessing_all_data()

print(X, y)