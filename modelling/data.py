import pandas as pd
import shutil
from zipfile import ZipFile, Path
import seaborn as sns
import matplotlib.pyplot as plt

class PriceHousesData():
    def __init__(self, path):
        self.path = path
    
    def exploration(self):
        data = _load_read_data(self.path)
        df_isnull = pd.DataFrame(data.isnull().sum().sort_values(ascending = False).head(20))
        df_isnull.to_csv(".\outputs\\before_nulls.txt")
        data = _fill_missing_values(data)
        df_isnull = pd.DataFrame(data.isnull().sum().sort_values(ascending = False).head(20))
        df_isnull.to_csv(".\outputs\\after_nulls.txt")
        return data

def _load_read_data(path): 
    shutil.unpack_archive(path)
    with ZipFile(path) as myzip:
        df = pd.read_csv(myzip.open("train.csv"))
    return df

def _fill_missing_values(data):
    df = data.copy()
    del df['Id']
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
