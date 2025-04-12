# 🔍 Classificador de Texto com Machine Learning usando Dados do Reddit

Este projeto demonstra a aplicação de **aprendizado supervisionado** em dados reais da web, com o objetivo de **classificar postagens do Reddit em categorias temáticas específicas**. Utilizando técnicas modernas de **Processamento de Linguagem Natural (NLP)** e **modelos de classificação**, ele constrói um pipeline completo, desde a coleta dos dados até a avaliação dos resultados.

As categorias-alvo deste projeto são os subreddits:

`datascience`, `machinelearning`, `physics`, `deeplearning` e `dataengineering`.

---

## 📦 Tecnologias e Bibliotecas

- **Python 3.10+**
- **PRAW** – para extração de dados da API do Reddit
- **Scikit-learn** – para pré-processamento e construção dos modelos
- **TF-IDF** – vetorização textual
- **SVD (Truncated SVD)** – redução de dimensionalidade
- Modelos utilizados:
  - **K-Nearest Neighbors (KNN)**
  - **Random Forest**
  - **Regressão Logística com Validação Cruzada**
- **Matplotlib e Seaborn** – para visualizações gráficas

---

## 🚀 Como Executar o Projeto

### 1. Clone este repositório

```bash
git clone https://github.com/seu-usuario/reddit-text-classifier.git
cd reddit-text-classifier

```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure suas credenciais da API do Reddit

Crie um arquivo `config.py` (que **já está ignorado pelo `.gitignore`**) com o seguinte conteúdo:

```python
client_id = "SEU_CLIENT_ID"
client_secret = "SEU_CLIENT_SECRET"
password = "SUA_SENHA"
user_agent = "NOME_DO_SEU_APP"
username = "SEU_USUARIO"
```

### 4. Execute o script principal

```bash
python main.py
```

---

## 📊 Resultados

- Geração de gráficos com a **distribuição de posts por categoria**
- **Matrizes de confusão** com porcentagem de acertos por modelo
- **Relatórios de classificação (precision, recall, f1-score)**

---

## 📄 Licença

Este projeto está licenciado sob os termos da licença MIT – consulte o arquivo `LICENSE` para detalhes.
