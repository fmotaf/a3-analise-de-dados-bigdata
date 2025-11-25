from functools import lru_cache

import pandas as pd


def _to_number(value: str) -> float:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # remove currency, percent, commas and whitespace
    for ch in ["₹", ",", "%"]:
        value = str(value).replace(ch, "")
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def load_raw_data(csv_path: str = "data/amazon.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_clean_data(csv_path: str = "data/amazon.csv") -> pd.DataFrame:
    df = load_raw_data(csv_path).copy()

    # numeric conversions
    df["discounted_price_num"] = df["discounted_price"].apply(_to_number)
    df["actual_price_num"] = df["actual_price"].apply(_to_number)
    df["discount_percentage_num"] = df["discount_percentage"].apply(_to_number)
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count_num"] = (
        df["rating_count"].astype(str).str.replace(",", "", regex=False)
    )
    df["rating_count_num"] = pd.to_numeric(df["rating_count_num"], errors="coerce")

    # categoria principal (primeiro nível antes de |)
    df["main_category"] = df["category"].astype(str).str.split("|").str[0]

    # garantir que texto de review seja string
    df["review_title"] = df["review_title"].astype(str)
    df["review_content"] = df["review_content"].astype(str)

    return df
