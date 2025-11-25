from typing import Tuple

import numpy as np
import pandas as pd


def resumo_geral(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "discounted_price_num",
        "actual_price_num",
        "discount_percentage_num",
        "rating_num",
        "rating_count_num",
    ]
    summary = df[cols].describe().T
    summary["missing"] = df[cols].isna().sum()
    return summary


def distribuicao_precos(df: pd.DataFrame, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    series = df["discounted_price_num"].dropna()
    return np.histogram(series, bins=bins)


def distribuicao_ratings(df: pd.DataFrame) -> pd.Series:
    return df["rating_num"].value_counts().sort_index()


def media_desconto_por_categoria(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("main_category")["discount_percentage_num"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="avg_discount")
    )


def top_produtos_por_rating(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    ordered = df.sort_values([
        "rating_num",
        "rating_count_num",
    ], ascending=[False, False])
    return ordered[[
        "product_id",
        "product_name",
        "main_category",
        "discounted_price_num",
        "rating_num",
        "rating_count_num",
    ]].head(n)


def correlacoes(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "discounted_price_num",
        "actual_price_num",
        "discount_percentage_num",
        "rating_num",
        "rating_count_num",
    ]
    return df[cols].corr()
