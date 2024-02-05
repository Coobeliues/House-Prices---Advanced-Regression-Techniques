from sklearn.utils.validation import check_is_fitted
import pandas as pd
import prepprocessing, MyTrain       # MyModels, MySave,
import warnings
warnings.filterwarnings('ignore')


train, test, all_data = prepprocessing.load_data("/home/ernar/Desktop/house_price/dataset/train.csv",
                                                 "/home/ernar/Desktop/house_price/dataset/test.csv")

# col = ['MSSubClass', 'OverallCond', 'BsmtHalfBath', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'] 
# all_data.drop(columns=col, axis=1, inplace=True) 
# col = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
# all_data.drop(columns=col, axis=1, inplace=True)

train, all_data, info_trans = prepprocessing.Transform(train, all_data, transform='log')

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

preprocessor_mean_minmax_mode_train = prepprocessing.preprocessingPipeline(X_train, imputer_strategy='mean', scaler='MinMaxScaler', cat_imputer_strategy='mode')
X_train_processed = preprocessor_mean_minmax_mode_train.fit_transform(X_train)


# Check if the transformers are fitted
check_is_fitted(preprocessor_mean_minmax_mode_train)


# Preprocess the test data
X_test_processed = preprocessor_mean_minmax_mode_train.transform(X_test)


# Check if the transformers are fitted
check_is_fitted(preprocessor_mean_minmax_mode_train)  # check for the transformers fitted on training data
check_is_fitted(preprocessor_mean_minmax_mode_train.named_transformers_['num'])  # check for the numeric transformer fitted on training data
check_is_fitted(preprocessor_mean_minmax_mode_train.named_transformers_['cat'])  # check for the categorical transformer fitted on training data


models_tuned = MyTrain.create_tuned_models(X_train_processed, y)


MyTrain.savePred(models_tuned, test, X_test_processed)


# Save preprocessing information to CSV
numImputer = str(repr(preprocessor_mean_minmax_mode_train.named_transformers_['num']['imputer']))
scaler = str(repr(preprocessor_mean_minmax_mode_train.named_transformers_['num']['scaler']))

# Save model information to CSV
model_info = []
for model_name, model in models_tuned.items():
    model_params = str(model.get_params())
    score_rmse = MyTrain.rmse_cv(model, X_train_processed, y).mean()
    model.fit(X_train_processed, y)
    model_info.append({
        "ModelName": model_name,
        "\tModelParams": model_params,
        "\tImputerStrategy": numImputer,
        "\tScaler": scaler,
        "\tScoreRMSE": score_rmse,
    })

model_info_df = pd.DataFrame(model_info)
model_info_df.to_csv("/home/ernar/Desktop/house_price/model_infos/model_info.csv", index=False)

