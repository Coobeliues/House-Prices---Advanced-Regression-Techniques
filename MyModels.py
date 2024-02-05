from lightgbm import LGBMRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# LGBMRegressor
def tune_lgbm(X_train, y):
    lgbm_params = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 500, 1000], 'max_depth': [3, 5, 7], 'num_leaves': [2, 10, 31, 50]}
    grid_lgbm = GridSearchCV(LGBMRegressor(random_state=42, verbose=-1), param_grid=lgbm_params)
    grid_lgbm.fit(X_train, y)

    return grid_lgbm.best_params_


# LassoCV
def tune_lasso(X_train, y):
    param_grid_lasso = {'eps': [1e-4, 1e-3, 1e-2], 'n_alphas': [100, 500, 1000], 'max_iter': [500, 1000]}
    grid_lasso = GridSearchCV(LassoCV(), param_grid=param_grid_lasso)
    grid_lasso.fit(X_train, y)

    return grid_lasso.best_params_


# ElasticNetCV
def tune_enet(X_train, y):
    param_grid_enet = {'l1_ratio': [0.1, 0.5, 0.7, 0.9], 'eps': [1e-4, 1e-3, 1e-2], 'n_alphas': [100, 500, 1000], 'max_iter': [500, 1000]}
    grid_enet = GridSearchCV(ElasticNetCV(), param_grid=param_grid_enet)
    grid_enet.fit(X_train, y)

    return grid_enet.best_params_


# XGBRegressor
def tune_xgb(X_train, y):
    xgb_params = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 500, 1000], 'max_depth': [3, 5, 7]}
    grid_xgb = GridSearchCV(XGBRegressor(random_state=42), param_grid=xgb_params)
    grid_xgb.fit(X_train, y)

    return grid_xgb.best_params_


# AdaBoostRegressor
def tune_ada(X_train, y):
    adaboost_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    grid_adaboost = GridSearchCV(AdaBoostRegressor(random_state=42), param_grid=adaboost_params)
    grid_adaboost.fit(X_train, y)

    return grid_adaboost.best_params_


# Ridge
def tune_ridge(X_train, y):
    param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 20, 50]}
    grid_ridge = GridSearchCV(Ridge(), param_grid=param_grid_ridge)
    grid_ridge.fit(X_train, y)

    return grid_ridge.best_params_['alpha']


# RandomForestRegressor
def tune_rf(X_train, y):
    param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20, 50], 'min_samples_split': [2, 5]}
    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid=param_grid_rf)
    grid_rf.fit(X_train, y)

    return grid_rf.best_params_


# GradientBoostingRegressor
def tune_gb(X_train, y):
    param_grid_gb = {'n_estimators': [200, 300, 500], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 4]}
    grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid=param_grid_gb)
    grid_gb.fit(X_train, y)

    return grid_gb.best_params_


# KNeighborsRegressor
def tune_knn(X_train, y):
    k_range = list(range(1, 21))
    weight_options = ['distance']
    param_grid = dict(n_neighbors=k_range, weights=weight_options)

    grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid=param_grid)
    grid_knn.fit(X_train, y)

    return grid_knn.best_params_


# SVR
def tune_svr(X_train, y):
    param_grid_svr = {'C': [0.1], 'kernel': ['linear', 'poly', 'rbf'], 'degree': [3,5], 'coef0': [0.01, 1, 5]}
    grid_svr = GridSearchCV(SVR(), param_grid=param_grid_svr)
    grid_svr.fit(X_train, y)

    return grid_svr.best_params_


# def create_tuned_models(X_train, y):
#     best_params_lgbm = tune_lgbm(X_train, y)
#     best_params_lasso = tune_lasso(X_train, y)
#     best_params_enet = tune_enet(X_train, y)
#     best_params_xgb = tune_xgb(X_train, y)
#     best_params_ada = tune_ada(X_train, y)
#     best_params_ridge = tune_ridge(X_train, y)
#     best_params_rf = tune_rf(X_train, y)
#     best_params_gb = tune_gb(X_train, y)
#     best_params_knn = tune_knn(X_train, y)
#     best_params_svr = tune_svr(X_train, y)


#     models_tuned = {'Ridge': Ridge(alpha=best_params_ridge),
#                     'Random Forest': RandomForestRegressor(**best_params_rf, random_state=42),
#                     'Gradient Boosting': GradientBoostingRegressor(**best_params_gb, random_state=42),
#                     'AdaBoost': AdaBoostRegressor(**best_params_ada, random_state=42),
#                     'XGB': XGBRegressor(**best_params_xgb, random_state=42),
#                     'LassoCV': LassoCV(**best_params_lasso),
#                     'ElasticNetCV': ElasticNetCV(**best_params_enet),
#                     'LGBMRegressor': LGBMRegressor(**best_params_lgbm, random_state=42, verbose=-1),
#                     'KNN': KNeighborsRegressor(**best_params_knn),
#                     'SVR': SVR(**best_params_svr)
#                 }

#     return models_tuned


# def evaluate_models(models_tuned, X_train, y):
#     for model_name, model in models_tuned.items():
#         score = rmse_cv(model).mean()
#         print(f"{model_name} RMSE: {score}")


# def trainEvaluate_models(models_tuned, X_train, y):
#     for model_name, model in models_tuned.items():
#         score = rmse_cv(model).mean()
#         print(f"{model_name} RMSE: {score}")

#         model.fit(X_train, y)


# models_tuned = create_tuned_models(X_train, y)
# evaluate_models(models_tuned, X_train, y)
# trainEvaluate_models(models_tuned, X_train, y)
