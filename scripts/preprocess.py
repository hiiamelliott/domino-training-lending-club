"""
preprocess.py
-------------
Feature engineering and cleaning pipeline for the LendingClub loan dataset.

Steps:
    1. Load raw CSV from Domino Dataset or S3 data source
    2. Drop high-nullity columns (>40% missing)
    3. Filter to fully resolved loan statuses only
    4. Create binary target: is_default
    5. Engineer new features: debt_to_income_ratio, credit_utilization, loan_to_income
    6. Encode categorical features
    7. Drop remaining nulls and redundant columns
    8. Write cleaned dataset to Domino Dataset

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --input /path/to/raw.csv --output /path/to/cleaned.csv
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (override via CLI args or environment variables)
# ---------------------------------------------------------------------------
PROJECT_NAME = os.environ.get("DOMINO_PROJECT_NAME", "LendingClubProject")
DEFAULT_INPUT = f"/mnt/data/{PROJECT_NAME}/lending_raw.csv"
DEFAULT_OUTPUT = f"/mnt/data/{PROJECT_NAME}/lending_clean.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NULL_THRESHOLD = 0.40          # Drop columns with > 40% missing values

# Loan statuses that represent a fully resolved outcome
RESOLVED_STATUSES = [
    "Fully Paid",
    "Charged Off",
    "Default",
]

# Target: these statuses map to is_default = 1
DEFAULT_STATUSES = ["Charged Off", "Default"]

# Categorical columns to one-hot encode
CATEGORICAL_COLS = [
    "grade",
    "home_ownership",
    "purpose",
    "term",
    "verification_status",
]

# Columns to drop after feature engineering (leakage or redundant)
COLS_TO_DROP = [
    "loan_status",       # replaced by is_default
    "sub_grade",         # granular version of grade — keep grade only
    "emp_title",         # high cardinality free text
    "title",             # high cardinality free text
    "zip_code",          # partially redacted, low signal
    "addr_state",        # too granular for this workshop
    "earliest_cr_line",  # replaced by derived credit_age_years
    "issue_d",           # date — not used in scoring
    "url",
    "desc",
    "id",
    "member_id",
    "policy_code",
]

# Numeric columns expected for feature engineering
REQUIRED_NUMERIC = [
    "loan_amnt",
    "annual_inc",
    "revol_bal",
    "revol_util",
    "dti",
    "int_rate",
    "installment",
    "open_acc",
    "total_acc",
    "pub_rec",
    "delinq_2yrs",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    log.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def drop_high_null_cols(df: pd.DataFrame, threshold: float = NULL_THRESHOLD) -> pd.DataFrame:
    null_frac = df.isnull().mean()
    cols_to_drop = null_frac[null_frac > threshold].index.tolist()
    log.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100:.0f}% nulls: {cols_to_drop}")
    return df.drop(columns=cols_to_drop, errors="ignore")


def filter_resolved_loans(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["loan_status"].isin(RESOLVED_STATUSES)].copy()
    log.info(f"Filtered to resolved loans: {before:,} → {len(df):,} rows")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df["is_default"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    default_rate = df["is_default"].mean()
    log.info(f"Target created — default rate: {default_rate:.2%}")
    return df


def clean_percent_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip % signs from int_rate and revol_util if stored as strings."""
    for col in ["int_rate", "revol_util"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace("%", "", regex=False).astype(float)
    return df


def clean_term_col(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise term to '36' or '60' string values."""
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.strip().str.replace(" months", "", regex=False)
    return df


def clean_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """Convert emp_length strings ('3 years', '10+ years', '< 1 year') to ordinal int."""
    if "emp_length" not in df.columns:
        return df
    mapping = {
        "< 1 year": 0,
        "1 year":   1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years":  5, "6 years": 6, "7 years": 7, "8 years": 8,
        "9 years":  9, "10+ years": 10,
    }
    df["emp_length"] = df["emp_length"].map(mapping)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering new features...")

    # Ratio of loan amount to annual income
    df["loan_to_income"] = np.where(
        df["annual_inc"] > 0,
        df["loan_amnt"] / df["annual_inc"],
        np.nan
    )

    # Credit utilisation (revol_util already a % — normalise to 0–1)
    df["credit_utilization"] = df["revol_util"].clip(0, 100) / 100

    # Monthly payment burden as % of monthly income
    df["payment_to_income"] = np.where(
        df["annual_inc"] > 0,
        (df["installment"] * 12) / df["annual_inc"],
        np.nan
    )

    # Derogatory mark flag
    df["has_derog"] = ((df["pub_rec"] > 0) | (df["delinq_2yrs"] > 0)).astype(int)

    # Credit breadth ratio
    df["credit_breadth"] = np.where(
        df["total_acc"] > 0,
        df["open_acc"] / df["total_acc"],
        np.nan
    )

    # Credit age in years from earliest_cr_line
    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        df["credit_age_years"] = (
            pd.Timestamp("today") - df["earliest_cr_line"]
        ).dt.days / 365.25
    
    log.info("Feature engineering complete")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in CATEGORICAL_COLS if c in df.columns]
    log.info(f"One-hot encoding: {present}")
    df = pd.get_dummies(df, columns=present, drop_first=True, dtype=int)
    return df


def drop_redundant_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLS_TO_DROP if c in df.columns]
    log.info(f"Dropping redundant columns: {cols}")
    return df.drop(columns=cols, errors="ignore")


def drop_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()
    log.info(f"Dropped rows with remaining nulls: {before:,} → {len(df):,}")
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Cleaned dataset saved to: {path}  ({len(df):,} rows, {df.shape[1]} columns)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_data(input_path)
    df = drop_high_null_cols(df)
    df = filter_resolved_loans(df)
    df = create_target(df)
    df = clean_percent_cols(df)
    df = clean_term_col(df)
    df = clean_emp_length(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    df = drop_redundant_cols(df)
    df = drop_remaining_nulls(df)
    save_data(df, output_path)

    # Summary stats
    log.info("=== Preprocessing Summary ===")
    log.info(f"Final shape     : {df.shape}")
    log.info(f"Default rate    : {df['is_default'].mean():.2%}")
    log.info(f"Columns         : {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LendingClub loan data")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Path to raw CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path for cleaned CSV output")
    args = parser.parse_args()

    run_pipeline(args.input, args.output)
