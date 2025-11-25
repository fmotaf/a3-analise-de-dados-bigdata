# Projeto de Análise de Dados Amazon

Este projeto realiza análise exploratória e modelagem simples utilizando um dataset de produtos da Amazon, com visualização interativa via Streamlit.

## Integrantes:
- Bruno Melo
- Daniel Menezes
- Fernando Mota
- Lucas Albino

## Estrutura

- `data/amazon.csv`: dataset original.
- `app/data_loader.py`: leitura e limpeza dos dados.
- `app/analysis.py`: funções de análise estatística e agregações.
- `app/nlp_utils.py`: funções simples para análise de texto das reviews.
- `app/models.py`: modelo de regressão linear para prever rating.
- `app/streamlit_app.py`: aplicação Streamlit.

## Como rodar

Dentro da pasta do projeto (`a3-unifacs`):

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Certifique-se de que o arquivo `data/amazon.csv` está no caminho correto.
