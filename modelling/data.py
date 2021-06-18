import pandas as pd
import shutil
import seaborn as sns
import matplotlib.pyplot as plt

from zipfile import ZipFile, Path
from sklearn.model_selection import train_test_split

class PriceHousesData():
    """
    Built specially for kaggle´s dataset competition

    Read an load data
    -Preprocessing train dataset
    -Preprocessing test dataset
    Paramaters:
    -----------
    path: string--> directory for train or test dataset
    file: string--> file name to unpack 
    """
    def __init__(self, path, train, test):
        self.path = path
        self.train = train
        self.test = test
    
    #processing joined train-test datasets, return two train-test dataset 
    def preprocessing_all_data(self):
        df,df_ = _load_read_data(self.path, self.train, self.test)
        #Concatenate datasets for preprocessing
        data = pd.concat([df,df_])
        #Delete ID column 
        del data['Id']
        #Extracting missing values an save in a text file for watching
        df_isnull = pd.DataFrame(data.isnull().sum().sort_values(ascending = False).head(20))
        df_isnull.to_csv(".\outputs\\dataset_missing_values(before).txt")
        data = _fill_missing_values_train_df(data)
        data = _fill_missing_values_test_df(data)
        df_isnull = pd.DataFrame(data.isnull().sum().sort_values(ascending = False).head(20))
        df_isnull.to_csv(".\outputs\\dataset_missing_values(after).txt")
        data = _change_typologies(data)        
        data = _encoding(data)
        X = data.drop(['SalePrice'], axis = 1)
        y = data['SalePrice']
        X_train = X[:1460]
        X_test =X[1460:] 
        y_train = y[:1460]
        return X_train, X_test, y_train

    #Processing train dataset, return a new dataset
    def preprocessing_train_df(self):
        data = _load_read_data(self.path, self.file)
        df_isnull = pd.DataFrame(data.isnull().sum().sort_values(ascending = False).head(20))
        df_isnull.to_csv(".\outputs\\train_dataset_missing_values(before).txt")
        data = _fill_missing_values_train_df(data)
        df_isnull = pd.DataFrame(data.isnull().sum().sort_values(ascending = False).head(20))
        df_isnull.to_csv(".\outputs\\train_dataset_missing_values(after).txt")
        data = _change_typologies(data)        
        data = _encoding(data)
        return data
    
    #preprocessing test dataset, return a new dataset
    def preprocessing_test_df(self):
        data = _load_read_data(self.path, self.file)
        df_isnull = pd.DataFrame(data.isnull().sum().sort_values(ascending = False).head(20))
        df_isnull.to_csv(".\outputs\\test_dataset_missing_values(before).txt")
        data = _fill_missing_values_test_df(data)
        df_isnull = pd.DataFrame(data.isnull().sum().sort_values(ascending = False).head(20))
        df_isnull.to_csv(".\outputs\\test_dataset_missing_values(after).txt")
        data = _change_typologies(data)
        data = _encoding(data)
        return data
    
def _load_read_data(path, train, test): 
    shutil.unpack_archive(path)
    with ZipFile(path) as myzip:
        df = pd.read_csv(myzip.open(train))
        df_ = pd.read_csv(myzip.open(test))
    # data = pd.concat([df,df_], axis = 1)
    return df, df_

def _fill_missing_values_train_df(data):
    df = data.copy()
    df['PoolQC'] = df['PoolQC'].fillna("No Pool")
    df['MiscFeature'] = df['MiscFeature'].fillna("Other")
    df['Alley'] = df['Alley'].fillna("No Alley")
    df['Fence'] = df['Fence'].fillna("No Fence")
    df['FireplaceQu'] = df['FireplaceQu'].fillna("No Fireplace")
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
    df['GarageCond'] = df['GarageCond'].fillna("No Garage")
    df['GarageType'] = df['GarageType'].fillna("No Garage")
    df['GarageFinish'] = df['GarageFinish'].fillna("No Garage")
    df['GarageQual'] = df['GarageQual'].fillna("No Garage")
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna("No Basement")
    df['BsmtExposure'] = df['BsmtExposure'].fillna("No Basement")
    df['BsmtQual'] = df['BsmtQual'].fillna("No Basement")
    df['BsmtCond'] = df['BsmtCond'].fillna("No Basement")
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna("No Basement")
    df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
    df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0])
    return df

def _fill_missing_values_test_df(data):
    df = data.copy()
    df = _fill_missing_values_train_df(df)
    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
    df['BsmtFullBath'] = df['BsmtFullBath'].fillna(df['BsmtFullBath'].mode()[0])
    df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(df['BsmtHalfBath'].mode()[0])
    df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
    df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
    df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].mode()[0])
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
    df['GarageCars'] = df['GarageCars'].fillna(df['GarageCars'].mode()[0])
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mode()[0])
    df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].mode()[0])
    df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(df['BsmtFinSF2'].mode()[0])
    df['GarageArea'] = df['GarageArea'].fillna(df['GarageArea'].mode()[0])
    return df

def _change_typologies(data):
    df = data.copy()
    df['MSSubClass'] = df['MSSubClass'].astype('string')
    df['OverallQual'] = df['OverallQual'].astype('string')
    df['OverallCond'] = df['OverallCond'].astype('string')
    df['YrSold'] = df['YrSold'].astype('string')
    df['MoSold'] = df['MoSold'].astype('string')
    df['YearBuilt'] = df['YearBuilt'].astype('string')
    df['YearRemodAdd'] = df['YearRemodAdd'].astype('string')
    df['GarageYrBlt'] = df['GarageYrBlt'].astype('string')
    return df

def _encoding(data):
    df = data.copy()
    categorical_ = df.select_dtypes(exclude = ['float', 'int64'])
    for c in categorical_:
        dummie_ = pd.get_dummies(df[c], prefix = f"{c}")
        del df[c]
        df = pd.concat([df, dummie_], axis = 1)
    return df
