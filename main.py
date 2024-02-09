# Импортируем основные библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from preprocessing import Preprocessor

# import shap
import streamlit as st

import joblib  # Библиотека для сохранения пайплайнов/моделей


import warnings

warnings.filterwarnings("ignore")

# Важная настройка для корректной настройки pipeline!
import sklearn

sklearn.set_config(transform_output="pandas")

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OrdinalEncoder,
    TargetEncoder,
)
from category_encoders.cat_boost import CatBoostEncoder

from sklearn.model_selection import GridSearchCV, KFold

# for model learning
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)

# models
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import accuracy_score


# tunning hyperparamters model
# import optuna

from tabulate import tabulate
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import accuracy_score


# tunning hyperparamters model
# import optuna

from tabulate import tabulate


# preprocessor = Preprocessor()
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")


st.write(
    """
# Предсказание цен на недвижимость.

Загрузите csv файл.
"""
)

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл", type=["csv"])

if uploaded_file is not None:
    # Прочитать данные из файла
    data = pd.read_csv(uploaded_file)
    transformed_data = preprocessor.transform(data)

    # Предсказание по всем записям
    predictions = np.expm1(model.predict(transformed_data))

    formatted_predictions = np.char.mod("$%.0f", predictions)

    output_df = pd.DataFrame({"Price predictions": formatted_predictions})
    # output_df = output_df.reset_index(drop=True)
    output_df.index += 1

    # Вывод предсказаний
    st.write("Предсказания модели:", output_df)
