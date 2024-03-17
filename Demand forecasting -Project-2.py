import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as sc
from sklearn.model_selection import cross_validate as cv
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint
from sklearn.decomposition import PCA




file= r"C:\Users\Madhujit\Desktop\train.csv"
df=pd.read_csv(file)
file2=r"C:\Users\Madhujit\Downloads\meal_info.csv"
file3=r"C:\Users\Madhujit\Downloads\fulfilment_center_info.csv"
meal=pd.read_csv(file2)
fulfilment_center_info=pd.read_csv(file3)

final_df=pd.merge(df,fulfilment_center_info,how="left",on="center_id")


final_df1=pd.merge(final_df,meal,how="left",on="meal_id")
final_df1.drop(["id"],axis=1,inplace=True)

final_df1.columns

final_df1["cd-md"]=final_df1["center_id"].astype(str)+"_"+final_df1["meal_id"].astype(str)

final_df1["cc-rc"]=final_df1["city_code"].astype(str)+"_"+final_df1["region_code"].astype(str)


final_df1["w-c-m"]=final_df1["week"].astype(str)+"_"+final_df1["center_id"].astype(str)+"_"+final_df1["meal_id"].astype(str)

center_id_statistics=final_df1.groupby("cd-md")["num_orders"].agg({"mean","std","median"}).reset_index()


center_id_statistics.columns=["cd-md","mean_c","std_c","median_c"]
center_id_statistics


final_df1=pd.merge(final_df1,center_id_statistics,how="left",on="cd-md")




city_region_statistics=final_df1.groupby("cc-rc")["num_orders"].agg({"mean","std","median"}).reset_index()


city_region_statistics.columns=["cc-rc","mean_cr","std_cr","median_cr"]

final_df1=pd.merge(final_df1,city_region_statistics,how="left",on="cc-rc")





week_statistics=final_df1.groupby("w-c-m")["num_orders"].agg({"mean","std"}).reset_index()


week_statistics.columns=["w-c-m","mean_wcm","std_wcm"]

final_df1=pd.merge(final_df1,week_statistics,how="left",on="w-c-m")


final_df1["cd-md"]=final_df1["cd-md"].astype("int64")

final_df1["cc-rc"]=final_df1["cc-rc"].astype("int64")
final_df1["w-c-m"]=final_df1["w-c-m"].astype("int64")

col=['center_type', 'category', 'cuisine']

one_hot_encoding=pd.get_dummies(final_df1[col])

final_df1.drop(columns=(col),axis=1,inplace=True)

final_df1=pd.concat([final_df1,one_hot_encoding],axis=1)

col2=['center_type_TYPE_A', 'center_type_TYPE_B', 'center_type_TYPE_C',
       'category_Beverages', 'category_Biryani', 'category_Desert',
       'category_Extras', 'category_Fish', 'category_Other Snacks',
       'category_Pasta', 'category_Pizza', 'category_Rice Bowl',
       'category_Salad', 'category_Sandwich', 'category_Seafood',
       'category_Soup', 'category_Starters', 'cuisine_Continental',
       'cuisine_Indian', 'cuisine_Italian', 'cuisine_Thai']


final_df1[col2]=final_df1[col2].astype("int64")

final_df1.drop(["std_wcm"],axis=1,inplace=True)

final_df1["median_c"]=final_df1["median_c"].fillna(0)

x=final_df1.drop(["num_orders"],axis=1)
y=final_df1["num_orders"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

scale=StandardScaler()

x_train_scale=scale.fit_transform(x_train)
x_test_scale=scale.transform(x_test)

reg=RandomForestRegressor()

model=reg.fit(x_train_scale,y_train)
y_pred=model.predict(x_test_scale)

r2_score(y_test,y_pred)

mean_absolute_error(y_test,y_pred)

model.feature_importances_


feature_score=dict(zip(model.feature_importances_,x.columns))

feature_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# Assuming X is your feature matrix and y is your target variable
# X and y should be numpy arrays or pandas DataFrames/Series

# Define your model
model = RandomForestRegressor()  # Example model, replace with your own

# Define the number of folds for cross-validation
num_folds = 5  # Example: 5-fold cross-validation

# Define the cross-validation method (e.g., KFold)
cross_val_method = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model, x_train_scale,y_train, cv=cross_val_method, scoring='r2')

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)

# Print the mean and standard deviation of the cross-validation scores
print("Mean CV Score:", cv_scores.mean())
print("Standard Deviation of CV Scores:", cv_scores.std())
