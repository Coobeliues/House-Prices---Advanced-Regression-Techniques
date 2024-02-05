import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

import MyModels


# rmse cross validationz
def rmse_cv(model, X_train, y):
    rmse = np.sqrt(
        -cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5)
    )

    return rmse


def create_tuned_models(X_train, y):
    best_params_lgbm = MyModels.tune_lgbm(X_train, y)
    best_params_lasso = MyModels.tune_lasso(X_train, y)
    best_params_enet = MyModels.tune_enet(X_train, y)
    best_params_xgb = MyModels.tune_xgb(X_train, y)
    best_params_ada = MyModels.tune_ada(X_train, y)
    best_params_ridge = MyModels.tune_ridge(X_train, y)
    best_params_rf = MyModels.tune_rf(X_train, y)
    best_params_gb = MyModels.tune_gb(X_train, y)
    best_params_knn = MyModels.tune_knn(X_train, y)
    best_params_svr = MyModels.tune_svr(X_train, y)

    # Выводим best_params для каждой модели
    # print(f"LGBMRegressor Best Params: {best_params_lgbm}")
    # print(f"LassoCV Best Params: {best_params_lasso}")
    # print(f"ElasticNetCV Best Params: {best_params_enet}")
    # print(f"XGBRegressor Best Params: {best_params_xgb}")
    # print(f"AdaBoostRegressor Best Params: {best_params_ada}")
    # print(f"Ridge Best Params: {best_params_ridge}")
    # print(f"Random Forest Best Params: {best_params_rf}")
    # print(f"Gradient Boosting Best Params: {best_params_gb}")
    # print(f"KNeighborsRegressor Best Params: {best_params_knn}")
    # print(f"SVR Best Params: {best_params_svr}")

    models_tuned = {
        "Ridge": Ridge(alpha=best_params_ridge),
        "Random Forest": RandomForestRegressor(**best_params_rf, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(**best_params_gb, random_state=42),
        "AdaBoost": AdaBoostRegressor(**best_params_ada, random_state=42),
        "XGB": XGBRegressor(**best_params_xgb, random_state=42),
        "LassoCV": LassoCV(**best_params_lasso),
        "ElasticNetCV": ElasticNetCV(**best_params_enet),
        "LGBMRegressor": LGBMRegressor(**best_params_lgbm, random_state=42, verbose=-1),
        "KNN": KNeighborsRegressor(**best_params_knn),
        "SVR": SVR(**best_params_svr),
    }

    return models_tuned


# def trainEvaluate_models(models_tuned, X_train, y, flag=None):
#     for model_name, model in models_tuned.items():
#         score = rmse_cv(model, X_train, y).mean()
#         print(f"{model_name} RMSE: {score}")

#         if flag is not None:
#             if flag == True:
#                 model.fit(X_train, y)


def savePred(models_tuned, test, X_test):
    for model_name, model in models_tuned.items():
        predictions = pd.DataFrame(
            {"Id": test["Id"], "SalePrice": np.expm1(model.predict(X_test))}
        )
        # # Model Information
        # model_params = str(model.get_params())  # Convert model parameters to string
        # preproc_methods = str(X_train.columns)  # Just an example, modify as per your preprocessing method
        # score_rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5)).mean()

        # model_info = pd.DataFrame({
        #     "ModelName": [model_name],
        #     "ModelParams": [model_params],
        #     "PreprocessingMethods": [preproc_methods],
        #     "ScoreRMSE": [score_rmse],
        # })

        # # Append model information to the results list
        # results.append(model_info)
        predictions.to_csv(
            f"/home/ernar/Desktop/house_price/predictions/predictions_{model_name}.csv",
            index=False,
        )
