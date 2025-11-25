from typing import Tuple

import numpy as np
import pandas as pd


def resumo_geral(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "preco_descontado",
        "preco_original",
        "desconto_percentual",
        "nota_media",
        "qtde_avaliacoes",
    ]
    summary = df[cols].describe().T
    summary["missing"] = df[cols].isna().sum()
    return summary


def distribuicao_precos(df: pd.DataFrame, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    series = df["preco_descontado"].dropna()
    return np.histogram(series, bins=bins)


def distribuicao_ratings(df: pd.DataFrame) -> pd.Series:
    return df["nota_media"].value_counts().sort_index()


def media_desconto_por_categoria(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("main_category")["desconto_percentual"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="avg_discount")
    )


def top_produtos_por_rating(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    ordered = df.sort_values([
        "nota_media",
        "qtde_avaliacoes",
    ], ascending=[False, False])
    return ordered[[
        "product_id",
        "product_name",
        "main_category",
        "preco_descontado",
        "nota_media",
        "qtde_avaliacoes",
    ]].head(n)


def correlacoes(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "preco_descontado",
        "preco_original",
        "desconto_percentual",
        "nota_media",
        "qtde_avaliacoes",
    ]
    return df[cols].corr()
