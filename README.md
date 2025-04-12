# Classificador de Texto com Dados do Reddit

Este projeto utiliza **Machine Learning supervisionado** para classificar postagens coletadas via API do Reddit em diferentes categorias:  
`datascience`, `machinelearning`, `physics`, `deeplearning` e `dataengineering`.

---

## 📌 Tecnologias Utilizadas

- **Python 3.10+**
- **PRAW** (API Reddit)
- **Scikit-learn**
- **TF-IDF + SVD**
- **KNN**, **Random Forest**, **Regressão Logística**
- **Matplotlib e Seaborn** para gráficos

---

## 🚀 Como Executar

### 1. Clone o repositório

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
