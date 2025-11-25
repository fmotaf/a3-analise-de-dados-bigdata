import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from analysis import (
    correlacoes,
    distribuicao_ratings,
    media_desconto_por_categoria,
    resumo_geral,
    top_produtos_por_rating,
)
from data_loader import load_clean_data
from models import prever_rating, treinar_modelo_rating
from nlp_utils import termos_mais_frequentes

st.set_page_config(page_title="Análise de Produtos Amazon", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_clean_data()


@st.cache_resource
def get_model():
    df = get_data()
    return treinar_modelo_rating(df)


def pagina_visao_geral(df: pd.DataFrame):
    st.header("Visão Geral")
    st.write("Resumo estatístico das principais variáveis numéricas.")

    resumo = resumo_geral(df)
    st.dataframe(resumo.round(2))

    col1, col2, col3 = st.columns(3)
    col1.metric("Nº de produtos", len(df))
    col2.metric("Rating médio", f"{df['nota_media'].mean():.2f}")
    col3.metric("Desconto médio (%)", f"{df['desconto_percentual'].mean():.1f}")


def pagina_precos_descontos(df: pd.DataFrame):
    st.header("Preços e Descontos")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribuição de preços (com desconto)")
        fig, ax = plt.subplots()
        sns.histplot(df["preco_descontado"].dropna(), bins=20, ax=ax)
        ax.set_xlabel("Preço com desconto (₹)")
        st.pyplot(fig)

    with col2:
        st.subheader("Distribuição de descontos (%)")
        fig, ax = plt.subplots()
        sns.histplot(df["desconto_percentual"].dropna(), bins=20, ax=ax)
        ax.set_xlabel("Desconto (%)")
        st.pyplot(fig)

    st.subheader("Desconto médio por categoria")
    cat = media_desconto_por_categoria(df)
    st.bar_chart(cat.set_index("main_category"))


def pagina_avaliacoes(df: pd.DataFrame):
    st.header("Avaliações (Ratings)")

    dist = distribuicao_ratings(df)
    st.subheader("Distribuição de ratings")
    st.bar_chart(dist)

    st.subheader("Top produtos por rating")
    top_n = st.slider("Quantidade de produtos", 5, 20, 10)
    top = top_produtos_por_rating(df, n=top_n)
    st.dataframe(top)


def pagina_reviews_texto(df: pd.DataFrame):
    st.header("Análise de Texto das Reviews")

    qtd = st.slider("Quantidade máxima de reviews para analisar", 100, 2000, 500)
    textos = df["review_content"].dropna().astype(str).head(qtd)
    freq = termos_mais_frequentes(textos, top_n=30)

    if not freq:
        st.info("Não há termos suficientes para análise.")
        return

    termos, contagens = zip(*freq)
    freq_df = pd.DataFrame({"termo": termos, "frequencia": contagens})
    st.bar_chart(freq_df.set_index("termo"))
    st.dataframe(freq_df)


def pagina_modelo(df: pd.DataFrame):
    st.header("Modelo Simples para Prever Rating")

    model, metrics = get_model()

    st.write("Desempenho no conjunto de teste:")
    col1, col2 = st.columns(2)
    col1.metric("R²", f"{metrics['r2']:.3f}")
    col2.metric("RMSE", f"{metrics['rmse']:.3f}")

    st.subheader("Testar previsão de rating")

    c1, c2 = st.columns(2)
    with c1:
        discounted_price = st.number_input("Preço com desconto (₹)", min_value=0.0, value=200.0)
        actual_price = st.number_input("Preço original (₹)", min_value=0.0, value=500.0)
    with c2:
        discount_percentage = st.number_input("Desconto (%)", min_value=0.0, max_value=100.0, value=50.0)
        rating_count = st.number_input("Quantidade de ratings", min_value=0.0, value=100.0)

    if st.button("Prever rating"):
        features = [discounted_price, actual_price, discount_percentage, rating_count]
        pred = prever_rating(model, features)
        st.success(f"Rating previsto: {pred:.2f}")


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("Filtros")

    categorias = ["(todas)"] + sorted(df["main_category"].dropna().unique().tolist())
    cat_sel = st.sidebar.selectbox("Categoria principal", categorias)

    rating_min = st.sidebar.slider("Rating mínimo", 0.0, 5.0, 0.0, 0.5)

    preco_min, preco_max = float(df["preco_descontado"].min()), float(
        df["preco_descontado"].max()
    )
    faixa_preco = st.sidebar.slider(
        "Faixa de preço (com desconto)", preco_min, preco_max, (preco_min, preco_max)
    )

    df_f = df.copy()
    if cat_sel != "(todas)":
        df_f = df_f[df_f["main_category"] == cat_sel]

    df_f = df_f[df_f["nota_media"] >= rating_min]
    df_f = df_f[
        (df_f["preco_descontado"] >= faixa_preco[0])
        & (df_f["preco_descontado"] <= faixa_preco[1])
    ]

    return df_f


def main():
    st.title("Análise de Dados de Produtos Amazon")

    df = get_data()
    df_filtrado = aplicar_filtros(df)

    st.sidebar.markdown("---")
    pagina = st.sidebar.radio(
        "Selecione a página",
        (
            "Visão geral",
            "Preços e descontos",
            "Avaliações",
            "Reviews (texto)",
            "Modelo de previsão",
        ),
    )

    st.sidebar.markdown(f"**Total de registros filtrados:** {len(df_filtrado)}")

    if pagina == "Visão geral":
        pagina_visao_geral(df_filtrado)
    elif pagina == "Preços e descontos":
        pagina_precos_descontos(df_filtrado)
    elif pagina == "Avaliações":
        pagina_avaliacoes(df_filtrado)
    elif pagina == "Reviews (texto)":
        pagina_reviews_texto(df_filtrado)
    elif pagina == "Modelo de previsão":
        pagina_modelo(df_filtrado)


if __name__ == "__main__":
    main()
