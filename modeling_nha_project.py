#!/usr/bin/env python
# coding: utf-8

# ## Librarires

# In[11]:


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from datetime import date
import joblib


# In[ ]:


df = pd.read_csv("NHA_training_data.csv")


# In[ ]:


df.drop(columns=["Unnamed: 0"],axis=1,inplace=True)
df.fillna(df.median(numeric_only=True),inplace=True)


# In[ ]:


df.describe().T


# ## Outlier replacing with median

# In[ ]:


selected_features = ["BASICSHIM","LHSSHIM","RHSSHIM"]

for col in selected_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = df[col].median()
    
    df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), median, df[col])


df


# ## Feature selection 

# In[ ]:


X = df[["LHSSHIM","RHSSHIM","TYPE"]].copy()
X['TYPE'] = X['TYPE'].astype('category')

y = df["BASICSHIM"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)


# ## Testing dataset loading 

# In[ ]:


tdf = pd.read_csv("NHA_test_data.csv")


# In[ ]:


X1 = tdf[["LHSSHIM","RHSSHIM","TYPE"]].copy()
X1['TYPE'] = X1['TYPE'].astype('category')
y1 = tdf['BASICSHIM']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=143)


# ## CatBoostingRegressor Modeling 

# In[ ]:


# cmodel = CatBoostRegressor(iterations = 200,
#                            max_depth = 8,
#                            learning_rate = 0.05,
#                            subsample = 1.0,
#                            verbose=0,
#                            loss_function="RMSE",
#                            random_state=42)

# cmodel.fit(X_train, y_train, cat_features=['TYPE'])
# y_pred = cmodel.predict(X_test)

# mse = mean_squared_error(y_test,y_pred)
# rmse = np.sqrt(mse)
# r2_scores = r2_score(y_test,y_pred)

# print(f"Mean Squared Error -> {mse*100:.2f}%")
# print(f"Root Mean Square Error -> {rmse:.2f}%")
# print(f"R2 Score -> {r2_scores*100:.2f}%")


# ## LightBoostRegressor Modeling

# In[ ]:


final_lmodel = LGBMRegressor(
    subsample=1.0,
    max_depth=6,
    n_estimators=200,
    learning_rate=0.05,
    verbose=-1,
    n_jobs=-1,
    colsample_bytree=1.0,
    random_state=123
)
final_lmodel.fit(X_train, y_train)
y_predictions = final_lmodel.predict(X_test)


mse = mean_squared_error(y_test, y_predictions)
rmse = np.sqrt(mse)
r2_scores = r2_score(y_test, y_predictions)

print(f"Mean Squared Error -> {mse:.4f}mm²")
print(f"Root Mean Square Error -> {rmse:.4f}mm²")
print(f"R2 Score -> {r2_scores*100:.2f}%")


# ## Testing The Training Data

# In[ ]:


y1_train_pred = final_lmodel.predict(X1_train)

r2_train = r2_score(y1_train, y1_train_pred)
mse_train = mean_squared_error(y1_train, y1_train_pred)
print(f"Training R²: {r2_train*100:.2f}% | MSE: {mse_train:.4f}mm^2")


# ## Testing The Test Data

# In[ ]:


y1_test_pred = final_lmodel.predict(X1_test)

r2_test = r2_score(y1_test, y1_test_pred)
mse_test = mean_squared_error(y1_test, y1_test_pred)
print(f"Test R²: {r2_test*100:.2f}% | MSE: {mse_test:.4f}")


# In[ ]:


# joblib.dump(final_lmodel,"lgbm.pkl")


# ## Real time value prediction

# In[ ]:


from datetime import date
import pandas as pd


sample = [[1.5, 1.59, "250+8=020"]]
sample_df = pd.DataFrame(sample, columns=["LHSSHIM", "RHSSHIM", "TYPE"])

sample_df["TYPE"] = sample_df["TYPE"].astype("category")  

actual_value = 1.440000057
pred = final_lmodel.predict(sample_df)

dic = {
    "date": date.today(),
    "Actual":actual_value,
    "Predicted": pred[0]
}

result_df = pd.DataFrame([dic])
result_df


# In[ ]:


Mean_Absolute_Error = abs(actual_value - pred)
print(f"MAE -> {Mean_Absolute_Error}")


# In[ ]:




