"""
Basic configurations and Path
"""

from pathlib import Path

#----------------------------------------------------------------------------------------------------------
# Root directory
#----------------------------------------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parents[1]


#---------------------------------------------------------------------------------------------------------
# Data and Model Path
#---------------------------------------------------------------------------------------------------------

DATA_DIR: Path = Path(BASE_DIR) / "data"
DATA_PATH: Path = Path(DATA_DIR) / "Telco_Customer_Churn.csv"
CLEANED_DATA_PATH: Path = Path(DATA_DIR) / "cleaned_data.csv"


MODEL_DIR: Path = Path(BASE_DIR) / "models"
BEST_MODEL_PATH: Path = Path(MODEL_DIR) / "best_model.joblib"
FEATURES_PATH: Path = Path(MODEL_DIR) / "feature_columns.jason"


TARGET_COLUMN: str = "Churn"

#----------------------------------------------------------------------------------------------------------
# TRAIN, TEST & CV
# ---------------------------------------------------------------------------------------------------------

TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
CV_FOLD: int = 5
N_JOBS: int = -1
SCORING: str = "roc_auc"

print ("configuarations are successfully applied")
