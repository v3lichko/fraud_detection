from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
METRICS_TABLE_PATH = PROJECT_ROOT / "reports" / "metrics.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "Class"
