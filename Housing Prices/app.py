#%%
import pandas as pd
from matplotlib import pyplot as plt



#%%
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")



#%%
df_train.info()



#%%
df_test.info()



#%%
col_to_drop = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
df_train.drop(columns=col_to_drop, axis=1, inplace=True)
df_test.drop(columns=col_to_drop, axis=1, inplace=True)



#%%
df_train.shape



#%%
df_test.shape



#%%
# substiuindo valores nulos pela média
df_train['LotFrontage'].fillna(int(df_train['LotFrontage'].mean()), inplace=True)
df_train['MasVnrArea'].fillna(int(df_train['MasVnrArea'].mean()), inplace=True)
df_train['GarageYrBlt'].fillna(int(df_train['GarageYrBlt'].mean()), inplace=True)

df_test['LotFrontage'].fillna(int(df_test['LotFrontage'].mean()), inplace=True)
df_test['MasVnrArea'].fillna(int(df_test['MasVnrArea'].mean()), inplace=True)
df_test['GarageYrBlt'].fillna(int(df_test['GarageYrBlt'].mean()), inplace=True)
df_test['BsmtFinSF1'].fillna(int(df_test['GarageYrBlt'].mean()), inplace=True)
df_test['BsmtFinSF2'].fillna(int(df_test['GarageYrBlt'].mean()), inplace=True)
df_test['BsmtUnfSF'].fillna(int(df_test['GarageYrBlt'].mean()), inplace=True)
df_test['TotalBsmtSF'].fillna(int(df_test['GarageYrBlt'].mean()), inplace=True)
df_test['GarageArea'].fillna(int(df_test['GarageArea'].mean()), inplace=True)



#%%
# substituindo valores faltantes pelo valo mais frequente
df_train.MasVnrType.fillna('None', inplace=True)
df_train.BsmtQual.fillna('TA', inplace=True)
df_train.BsmtCond.fillna('TA', inplace=True)
df_train.BsmtExposure.fillna('NO', inplace=True)
df_train.BsmtFinType1.fillna('Unf', inplace=True)
df_train.BsmtFinType2.fillna('Unf', inplace=True)
df_train.Electrical.fillna('SBrkr', inplace=True)
df_train.GarageType.fillna('Attchd', inplace=True)
df_train.GarageFinish.fillna('Unf', inplace=True)
df_train.GarageQual.fillna('TA', inplace=True)
df_train.GarageCond.fillna('TA', inplace=True)

df_test.MasVnrType.fillna('None', inplace=True)
df_test.BsmtQual.fillna('TA', inplace=True)
df_test.BsmtCond.fillna('TA', inplace=True)
df_test.BsmtExposure.fillna('NO', inplace=True)
df_test.BsmtFinType1.fillna('Unf', inplace=True)
df_test.BsmtFinType2.fillna('Unf', inplace=True)
df_test.Electrical.fillna('SBrkr', inplace=True)
df_test.GarageType.fillna('Attchd', inplace=True)
df_test.GarageFinish.fillna('Unf', inplace=True)
df_test.GarageQual.fillna('TA', inplace=True)
df_test.GarageCond.fillna('TA', inplace=True)
df_test.Utilities.fillna('AllPub', inplace=True)
df_test.Exterior1st.fillna('VinylSd', inplace=True)
df_test.Exterior2nd.fillna('VinylSd', inplace=True)
df_test.BsmtFullBath.fillna(float(0), inplace=True)
df_test.BsmtHalfBath.fillna(float(0), inplace=True)
df_test.KitchenQual.fillna('TA', inplace=True)
df_test.Functional.fillna('Typ', inplace=True)
df_test.GarageCars.fillna(float(2), inplace=True)
df_test.SaleType.fillna('WD', inplace=True)



#%%
# substituindo variáveis categóricas por dummy variables
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)



#%%
# colunas na base de dados de teste que, após gerar as variáveis dummy, 
# percebi que não continham os mesmos valores que a base de teste
for col_train in df_train.columns:
    if col_train not in df_test.columns:
        # print(col_train)
        # # SalePrice
        # # Utilities_NoSeWa
        # # Condition2_RRAe
        # # Condition2_RRAn
        # # Condition2_RRNn
        # # HouseStyle_2.5Fin
        # # RoofMatl_ClyTile
        # # RoofMatl_Membran
        # # RoofMatl_Metal
        # # RoofMatl_Roll
        # # Exterior1st_ImStucc
        # # Exterior1st_Stone
        # # Exterior2nd_Other
        # # Heating_Floor
        # # Heating_OthW
        # # Electrical_Mix
        # # GarageQual_Ex
        df_test[col_train] = 0



#%%
print("Shape treinamento {}".format(df_train.shape))
print("Shape teste {}".format(df_test.shape))



#%%
# separando variáveis de treino e teste
X_train = df_train.drop('SalePrice', axis=1).values
y_train = df_train['SalePrice'].values
X_test = df_test.values



#%%
# treinando modelo
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
score = linear_regression.score(X_train, y_train)
print("SCORE: {}".format(score))



#%%
df_submission = pd.DataFrame()
df_submission['Id'] = df_test['Id']
df_submission['SalePrice'] = linear_regression.predict(X_test)
csv = df_submission[['Id', 'SalePrice']]



#%%
csv.to_csv('submission.csv', index=False)