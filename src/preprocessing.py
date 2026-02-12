"""
Preprocessing utilities
- import pre-cleaned data
- impute null values
- scale/normalize data
- encode categorical data
- train_test split
"""

# ---------------------------------------------------------------------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.config import  CLEANED_DATA_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from typing import List, Tuple

#----------------------------------------------------------------------------------------------------------------------------------------
# LOAD PRE_CLEANED DATA
# ----------------------------------------------------------------------------------------------------------------------------------------

def load_data(data_path: Path = CLEANED_DATA_PATH):

    """
    This will load pre-cleaned data.
    -Removing leading or lagging white space from column name.
    -Return clean data.

    """
    df = pd.read_csv(data_path)

    # Normalizing column names

    df.columns = [str(c).strip() for c in df.columns.to_list()]
    return df


#----------------------------------------------------------------------------------------------------------------------------------------
# EXTRACTING CATEGORICAL AND NUMERICAL COLUMNS
#----------------------------------------------------------------------------------------------------------------------------------------

def _extract_cat_cols_num_cols(df:pd.DataFrame):

    """
    Extracting the categorical and numeric columns seperately
    """

    cat_cols = df.select_dtypes(include = ["object", "category", "bool"]).columns.to_list()
    num_cols = df.select_dtypes(include = ["int64", "float64", "number"]).columns.to_list()

    return cat_cols, num_cols


#----------------------------------------------------------------------------------------------------------------------------------------
# BUILDING PREPROCESSOR
#----------------------------------------------------------------------------------------------------------------------------------------

def preprocessor(df_or_X: pd.DataFrame):

    #creating a backup

    df = df_or_X.copy()

    # If target column is present, drop the column

    if TARGET_COLUMN in df.columns:
        df = df.drop(columns= TARGET_COLUMN)

    else:
        df = df
    
    # extracting the categorical columns and numeric columns
    cat_cols, num_cols = _extract_cat_cols_num_cols(df)

    #Numeric Pipeline: Impute --> Scale
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy= "median")),
        ("scaler", StandardScaler())
    ])

    # Categorical Pipeline: Impute --> Encode
    cat_transformer = Pipeline(steps = [
        ("imputer", SimpleImputer(strategy= "most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown= "ignore", sparse_output= False))
    ])

    #combining the pipelines
    transformers = []

    if cat_cols:
        transformers.append(("cat", cat_transformer, cat_cols))
    
    if num_cols:
        transformers.append(("num", num_transformer, num_cols))

    prepro = ColumnTransformer(transformers= transformers, remainder= "drop", verbose_feature_names_out= False)

    return prepro

#----------------------------------------------------------------------------------------------------------------------------------------
# TRAIN- TEST SPLIT
# ----------------------------------------------------------------------------------------------------------------------------------------

def split_data(df: pd.DataFrame):

    df = df.copy()

    #if target column is missing
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target Column {TARGET_COLUMN} is not found in DataFrame columns: {df.columns.to_list}")
    
    X = df.drop(columns= [TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    y = y.map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= TEST_SIZE, random_state= RANDOM_STATE, stratify= y)

    return X_train, X_test, y_train, y_test

print ("Preprocessing is executed")
    
    





