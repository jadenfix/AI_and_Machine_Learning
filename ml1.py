import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

name = ['jpfix','mshafe01']
#data = pd.read_csv("http://ashafran.github.io/boston.csv")
#data = pd.read_csv("boston.csv")

#data = pd.read_csv("/Users/jadenfix/Desktop/Graduate School Materials/Computing and Machine Learning/boston.csv")
print(data)
X = data.drop(columns=['MEDV'])
y = data['MEDV']

#model 1 
model1 = LinearRegression()
model1.fit(X, y)
y_pred1 = model1.predict(X)
mse1 = mean_squared_error(y, y_pred1)

#model 2 
poly2 = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X2 = poly2.fit_transform(X)
model2 = LinearRegression()
model2.fit(X2, y)
y_pred2 = model2.predict(X2)
mse2 = mean_squared_error(y, y_pred2)

#model 3
poly3 = PolynomialFeatures(degree=2, include_bias=False)
X3 = poly3.fit_transform(X)
model3 = LinearRegression()
model3.fit(X3, y)
y_pred3 = model3.predict(X3)
mse3 = mean_squared_error(y, y_pred3)

#model 4 
log_y = np.log(y)
model4 = LinearRegression()
model4.fit(X, log_y)
log_y_pred4 = model4.predict(X)
mse4 = mean_squared_error(np.exp(log_y), np.exp(log_y_pred4))

#model 5
X2_log = poly2.fit_transform(X)
model5 = LinearRegression()
model5.fit(X2_log, log_y)
log_y_pred5 = model5.predict(X2_log)
mse5 = mean_squared_error(np.exp(log_y), np.exp(log_y_pred5))

#model 6 
X3_log = poly3.fit_transform(X)
model6 = LinearRegression()
model6.fit(X3_log, log_y)
log_y_pred6 = model6.predict(X3_log)
mse6 = mean_squared_error(np.exp(log_y), np.exp(log_y_pred6))

#(BIC =n*log(MSE) + log(n)*p ) in order (4-6 ar logged)
bic1 = len(data) * np.log(mse1) + (X.shape[1]+1) * np.log(len(data))
bic2 = len(data) * np.log(mse2) + (X2.shape[1]+1) * np.log(len(data))
bic3 = len(data) * np.log(mse3) + (X3.shape[1]+1) * np.log(len(data))
bic4 = len(data) * np.log(mse4) + (X.shape[1]+1) * np.log(len(data))
bic5 = len(data) * np.log(mse5) + (X2.shape[1]+1) * np.log(len(data))
bic6 = len(data) * np.log(mse6) + (X3.shape[1]+1) * np.log(len(data))
bic = [bic1, bic2, bic3, bic4, bic5, bic6]
print(bic)

loo = LeaveOneOut()
loo_pred1 = cross_val_predict(model1, X, y, cv=loo)
loo_mse1 = mean_squared_error(y, loo_pred1)

loo_pred2 = cross_val_predict(model2, X2, y, cv=loo)
loo_mse2 = mean_squared_error(y, loo_pred2)

loo_pred3 = cross_val_predict(model3, X3, y, cv=loo)
loo_mse3 = mean_squared_error(y, loo_pred3)

loo_pred4 = cross_val_predict(model4, X, log_y, cv=loo)
loo_mse4 = mean_squared_error(np.exp(log_y), np.exp(loo_pred4))

loo_pred5 = cross_val_predict(model5, X2_log, log_y, cv=loo)
loo_mse5 = mean_squared_error(np.exp(log_y), np.exp(loo_pred5))

loo_pred6 = cross_val_predict(model6, X3_log, log_y, cv=loo)
loo_mse6 = mean_squared_error(np.exp(log_y), np.exp(loo_pred6))

loo = [loo_mse1, loo_mse2, loo_mse3, loo_mse4, loo_mse5, loo_mse6]
#kfold 6 (MSE) test (maybe train) in order (k=10) # all + 
kf = KFold(n_splits=10, random_state=0, shuffle=True)
kf_scores1 = cross_val_score(model1, X, y, cv=kf, scoring="neg_mean_squared_error")
kf_mse1 = -kf_scores1.mean()

kf_scores2 = cross_val_score(model2, X2, y, cv=kf, scoring="neg_mean_squared_error")
kf_mse2 = -kf_scores2.mean()

kf_scores3 = cross_val_score(model3, X3, y, cv=kf, scoring="neg_mean_squared_error")
kf_mse3 = -kf_scores3.mean()

kf_pred4 = cross_val_predict(model4, X, log_y, cv=kf)
kf_mse4 = mean_squared_error(np.exp(log_y), np.exp(kf_pred4))

kf_pred5 = cross_val_predict(model5, X2_log, log_y, cv=kf)
kf_mse5 = mean_squared_error(np.exp(log_y), np.exp(kf_pred5))

kf_pred6 = cross_val_predict(model6, X3_log, log_y, cv=kf)
kf_mse6 = mean_squared_error(np.exp(log_y), np.exp(kf_pred6))




kfoldregression = [kf_mse1, kf_mse2, kf_mse3, kf_mse4, kf_mse5, kf_mse6]
print(kfoldregression)
#best one
best_bic = np.argmin(bic) + 1
best_loo = np.argmin(loo) + 1
best_kfoldregression = np.argmin(kfoldregression) + 1

print("BIC values:", bic)
print("Best model via BIC:", best_bic)

print("LOO MSE values:", loo)
print("Best model via LOO MSE:", best_loo)

print("K-Fold MSE values:", kfoldregression)
print("Best model via K-Fold MSE:", best_kfoldregression)

###############################################################
###############################################################
###############################################################
#machine learning part 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
#models

model7=DecisionTreeRegressor(criterion="squared_error",max_depth=3,random_state=0)
model7.fit(X,y)
t7=tree.export_text(model7)
print(t7)
tree.plot_tree(model7,feature_names=list(X.columns), filled=True,fontsize=7)
#y_pred7 = model7.predict(X)
kfold = KFold(n_splits=10, random_state=0, shuffle=True)
cv_scores7 = cross_val_score(model7, X, y, cv=kfold, scoring="neg_mean_squared_error")
mse7 = -cv_scores7.mean()

model8=DecisionTreeRegressor(criterion="squared_error",max_depth=3,min_samples_leaf=10,random_state=0)
model8.fit(X,y)
t8=tree.export_text(model8)
print(t8)
tree.plot_tree(model8,feature_names=list(X.columns), filled=True,fontsize=7)
#y_pred8 = model8.predict(X)
cv_scores8 = cross_val_score(model8, X, y, cv=kfold, scoring="neg_mean_squared_error")
mse8 = -cv_scores8.mean()

model9 = RandomForestRegressor(random_state=0,max_features=None)
model9.fit(X,y)
feature_importances9 = model9.feature_importances_
#y_pred9 = model9.predict(X)
cv_scores9 = cross_val_score(model9, X, y, cv=kfold, scoring="neg_mean_squared_error")
mse9 = -cv_scores9.mean()

model10 = RandomForestRegressor(random_state=0,max_features=1/3)
model10.fit(X,y)
feature_importances10 = model10.feature_importances_
#y_pred10 = model10.predict(X)
cv_scores10 = cross_val_score(model10, X, y, cv=kfold, scoring="neg_mean_squared_error")
mse10 = -cv_scores10.mean()

kfold_tree_mse = [mse7, mse8, mse9, mse10]
best_tree = np.argmin(kfold_tree_mse) + 7
print("Best Tree:",best_tree)

kfoldoverall = kfoldregression + kfold_tree_mse
best_overall = np.argmin(kfoldoverall)+1
print("Best Overall(Accordig to MSE):",best_overall)

#all_models = [mse1, mse2, msel3, mse4, mse5, mse6, mse7, mse8, mse9, mse10]
#best_model = np.argmin(all_models)+1
#yhat = best_model.predict(X)
#print("This is Y-Hat:",yhat)
#yhat=np.argmin[yhats]
all_models = [None, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
# Generate Predictions from the Best Overall Model
# Map model indices to their corresponding feature transformations
transformed_X = {
    1: X,  # mod 1
    2: poly2.fit_transform(X),  # mod 2: interaction-only polynomial features
    3: poly3.fit_transform(X),  # mod 3: Full quadratic polynomial features
    4: X,  # mod 4
    5: poly2.fit_transform(X),  # interaction features
    6: poly3.fit_transform(X),  # quadratic features
    7: X,  # mod 7
    8: X,  # mod 8
    9: X,  # mod 9
    10: X,  # mod 10
}

# best model and corresponding features
best_model = all_models[best_overall]
best_X = transformed_X[best_overall]  # Apply appropriate transformation for the model
#predictions
yhat = best_model.predict(best_X)
print(f"Predictions from Best Model (Model {best_overall}):\n", yhat)
