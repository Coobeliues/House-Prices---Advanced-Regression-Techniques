import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    train.drop('Id', axis=1, inplace=True)  # Этот столбец не имеет никакого отношения, так что мы ее дропнем
    test = pd.read_csv(test_path)
    all_data = pd.concat([train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']])

    return train, test, all_data


def Transform(train, all_data, transform=None):
    transformation_info = ''

    if transform is not None:
        if transform == 'log':
            train["SalePrice"] = np.log1p(train["SalePrice"])

            # get numeric features
            numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

            skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))

            # фильтрация скошенности  > 0.75
            skewed_feats = skewed_feats[skewed_feats > 0.75].index

            for feat in skewed_feats:
                all_data[feat] = np.log1p(all_data[feat])

            transformation_info = 'log'
            # transformation_info['skewed_features'] = list(skewed_feats)
            # return train, all_data, transformation_info

        elif transform == 'box':
            train["SalePrice"] = boxcox1p(train["SalePrice"], 0.05)

            # get numeric features
            numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

            # compute skewness for each numeric feature
            skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))

            # filter features with skewness > 0.75
            skewed_feats = skewed_feats[skewed_feats > 0.75].index

            # apply Box-Cox transformation to reduce skewness
            for feat in skewed_feats:
                all_data[feat] = boxcox1p(all_data[feat], 0.05)

            transformation_info = 'boxcox'
            # transformation_info['skewed_features'] = list(skewed_feats)
            # return train, all_data, transformation_info

        else:
            raise ValueError("Invalid transformation. Use 'log' or 'boxcox'.")

    return train, all_data, transformation_info


def preprocessingPipeline(X_train, imputer_strategy=None, scaler=None, cat_imputer_strategy=None):

    # Numeric and categorical features
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Numeric transformer
    numeric_transformers = []
    if imputer_strategy is not None:
        if imputer_strategy == 'mean':
            numeric_transformers.append(('imputer', SimpleImputer(strategy='mean')))
        elif imputer_strategy == 'median':
            numeric_transformers.append(('imputer', SimpleImputer(strategy='median')))
        elif imputer_strategy == 'mode':
            numeric_transformers.append(('imputer', SimpleImputer(strategy='most_frequent')))
        else:
            raise ValueError("Invalid imputer strategy for numeric features. Use 'mean', 'median', or 'mode'.")
    
    if scaler is not None:
        if scaler == 'MinMaxScaler':
            numeric_transformers.append(('scaler', MinMaxScaler()))
        elif scaler == 'StandardScaler':
            numeric_transformers.append(('scaler', StandardScaler()))
        else:
            raise ValueError("Invalid scaler. Use 'MinMaxScaler' or 'StandardScaler'.")
    
    numeric_transformer = Pipeline(steps=numeric_transformers)

    # Categorical transformer
    categorical_transformers = []
    if cat_imputer_strategy is not None:
        if cat_imputer_strategy == 'mode':
            categorical_transformers.append(('imputer', SimpleImputer(strategy='most_frequent')))
        elif cat_imputer_strategy == 'constant':
            categorical_transformers.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
        else:
            raise ValueError("Invalid imputer strategy for categorical features. Use 'mode' or 'constant'.")
    
    categorical_transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
    categorical_transformer = Pipeline(steps=categorical_transformers)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor
