from category_encoders.cat_boost import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import NotFittedError, Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler


class Preprocessor:
    def __init__(self):
        pass

    def get_imputer(self):
        # Cтроим imputer для заполнения NaN значений
        NAImputerValues = [
            "Alley",
            "MasVnrType",
            "BsmtQual",
            "Functional",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
            "FireplaceQu",
            "GarageType",
            "GarageFinish",
            "GarageQual",
            "GarageCond",
            "PoolQC",
            "Fence",
            "MiscFeature",
        ]
        MostFrequentImputerValues = [
            "Utilities",
            "MSZoning",
            "Exterior1st",
            "Exterior2nd",
            "Electrical",
            "KitchenQual",
            "SaleType",
        ]
        ZeroValuesImputerValues = [
            "LotFrontage",
            "MasVnrArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "BsmtFullBath",
            "BsmtHalfBath",
            "GarageCars",
            "GarageArea",
        ]
        MedianImputerValues = ["GarageYrBlt"]

        self.imputer = ColumnTransformer(
            transformers=[
                (
                    "MostFrequentImputer",
                    SimpleImputer(strategy="most_frequent"),
                    MostFrequentImputerValues,
                ),
                (
                    "NAImputer",
                    SimpleImputer(strategy="constant", fill_value="NA"),
                    NAImputerValues,
                ),
                (
                    "ZeroValuesImputer",
                    SimpleImputer(strategy="constant", fill_value=0),
                    ZeroValuesImputerValues,
                ),
                (
                    "MedianImputer",
                    SimpleImputer(strategy="median"),
                    MedianImputerValues,
                ),
            ],
            verbose_feature_names_out=False,
            remainder="passthrough",
        )

    def get_encoder(self):
        # Столбец, который планируем кодировать с помощью OneHotEncoder
        one_hot_encoding_columns = [
            "Street",
            "Alley",
            "Utilities",
            "RoofStyle",
            "RoofMatl",
            "MasVnrType",
            "Foundation",
            "CentralAir",
            "GarageFinish",
            "PavedDrive",
        ]

        # Столбец, который планируем кодировать порядково, с помощью OrdinalEncoder
        ordinal_encoding_columns = [
            "LotShape",
            "LandContour",
            "LotConfig",
            "LandSlope",
            "ExterQual",
            "ExterCond",
            "BsmtQual",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
            "HeatingQC",
            "Electrical",
            "KitchenQual",
            "FireplaceQu",
            "GarageQual",
            "GarageCond",
            "PoolQC",
            "Fence",
        ]

        # Столбец, который планируем кодировать с помощью CatboostEncoder
        catboost_encoding_columns = [
            "MSZoning",
            "Neighborhood",
            "Condition1",
            "Condition2",
            "BldgType",
            "HouseStyle",
            "Exterior1st",
            "Exterior2nd",
            "Heating",
            "Functional",
            "GarageType",
            "MiscFeature",
            "SaleType",
            "SaleCondition",
        ]
        # Оборачиваем Encoder в ColumnTransformer
        self.encoder = ColumnTransformer(
            [
                (
                    "one_hot_encoding",
                    OneHotEncoder(sparse_output=False),
                    one_hot_encoding_columns,
                ),
                ("ordinal_encoding", OrdinalEncoder(), ordinal_encoding_columns),
                ("catboost_encoding", CatBoostEncoder(), catboost_encoding_columns),
            ],
            verbose_feature_names_out=False,
            remainder="passthrough",
        )

    def get_scaler(self):
        # Числовые столбцы, которые необходимо пронормировать
        standard_scaler_columns = [
            "MSZoning",
            "Neighborhood",
            "Condition1",
            "Condition2",
            "BldgType",
            "HouseStyle",
            "Exterior1st",
            "Exterior2nd",
            "Heating",
            "Functional",
            "GarageType",
            "MiscFeature",
            "SaleType",
            "SaleCondition",
            "MSSubClass",
            "LotFrontage",
            "LotArea",
            "OverallQual",
            "OverallCond",
            "YearBuilt",
            "YearRemodAdd",
            "MasVnrArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "LowQualFinSF",
            "GrLivArea",
            "BsmtFullBath",
            "BsmtHalfBath",
            "FullBath",
            "HalfBath",
            "BedroomAbvGr",
            "KitchenAbvGr",
            "TotRmsAbvGrd",
            "Fireplaces",
            "GarageYrBlt",
            "GarageCars",
            "GarageArea",
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "PoolArea",
            "MiscVal",
            "MoSold",
            "YrSold",
        ]

        # Оборачиваем Scaler в ColumnTransformer
        self.scaler = ColumnTransformer(
            [("scaling_num_columns", MinMaxScaler(), standard_scaler_columns)],
            verbose_feature_names_out=False,
            remainder="passthrough",
        )

    def fit(self, data, y):
        self.get_imputer()
        self.get_encoder()
        self.get_scaler()
        self.pipeline = Pipeline(
            [
                ("imputer", self.imputer),
                ("encoder", self.encoder),
                ("scaler", self.scaler),
            ]
        )
        data = data.iloc[:1460, :]

        self.pipeline.fit(data, y)

    def transform(self, data):
        try:
            return self.pipeline.transform(data)
        except NotFittedError():
            raise NotFittedError("Fit data first")

    def fit_transform(self, data, y):
        self.fit(data, y)
        return self.transform(data)
