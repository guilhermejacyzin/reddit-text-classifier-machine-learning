# Projeto 1 - Classificação de Texto com Aprendizagem Supervisionada

# Pacotes
import re # regular expression para filtrar, limpar e configurar o texto
from typing import Any

import praw # autenticação com reddit
import config # autenticação com reddit
import numpy as np
from sklearn.model_selection import train_test_split # divisão de dados de treino e teste
from sklearn.feature_extraction.text import TfidfVectorizer # preparar matriz com dados de texto
from sklearn.decomposition import TruncatedSVD # redução de dimensionalidade

## 3 modelos para avaliar qual deles tem a melhor 'performance'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import classification_report # para imprimir as métricas de avaliação do modelo
from sklearn.pipeline import Pipeline # criação do pipeline // sequência de atividades para treinar o modelo
from sklearn.metrics import confusion_matrix # matriz de confusão para imprimir o resultado final do modelo
import matplotlib.pyplot as plt # criação de gráficos
import seaborn as sns  # criação de gráficos

## Carregando lista de dados

# Lista de temas utilizados para buscas no Reddit.
# Classes - variável target // dados de saída
assuntos = ['datascience', 'machinelearning', 'physics', 'deeplearning', 'dataengineering']

# Função - Carregamento de dados
def carrega_dados():

    from config import client_id, client_secret, password, user_agent, username

    api_reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        password=password,
        user_agent=user_agent,
        username=username
    )

    # Número de caracteres usando expressões regulares // Verifica se existe caracter e dígito - Caso tenha
    # Para cada post extraído, filtra caracteres alfabéticos e dígitos e busca o comprimento para ver o total de caracteres em cada post
    char_count = lambda post: len(re.sub(r'\W|\d', '', post.selftext))

    # Condição para filtrar os posts (retorna posts com número de caracteres >= 100)
    mask = lambda post: char_count(post) >= 100

    # Listas para os resultados
    data = []
    labels = []

    # Loop
    for i, assunto in enumerate(assuntos):

        # Extrai os posts do assunto do Loop
        subreddit_data = api_reddit.subreddit(assunto).new(limit = 1000)

        # Filtra os posts que não satisfazem a condição
        # Para cada post retornado do filtro aplicado ao post, retorna o próprio post
        posts = [post.selftext for post in filter(mask, subreddit_data)]

        # Adiciona posts e labels às linhas
        data.extend(posts)
        labels.extend([i] * len(posts))

        # Print
        print(f"Número de posts do assunto r/{assunto}: {len(posts)}",
              f"\nUm dos posts extraídos: {posts[0][:600]}...\n",
              "_" * 80 + '\n')

    return data, labels


## Divisão em Dados de Treino e Teste

# Variáveis de Controle
TEST_SIZE = .2 # 20% dos dados para teste
RANDOM_STATE = 0 # random_state = 0 para reproduzir os mesmos resultados nesta versão // mesmo padrão de aleatoriedade

# Função para Split dos Dados
def split_data():

    print(f'Split {100 * TEST_SIZE}% dos dados para teste e avaliação do modelo...')

    # Split de dados
    X_treino, X_teste, y_treino, y_teste = train_test_split(data,
                                                            labels,
                                                            test_size= TEST_SIZE,
                                                            random_state= RANDOM_STATE)

    print(f'{len(y_teste)} amostras de teste.')

    return X_treino, X_teste, y_treino, y_teste


## Pré Processamento de dados e Extração dos Atributos

# Remover símbolos, números e strings semelhantes a url com pré-processador personalizado
# Vetorizar texto usando o termo frequência inversa de frequência do documento
# Reduzir para valores principais usando decomposição de valor singular
# Particionar dados e rótulos em conjuntos de treinamento // validação

# Vars de controle
MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER = 30

# Função para o pipeline de pré-processamento

def processing_pipeline():

    # Remover símbolos, números e strings semelhantes a url com pré-processador personalizado
    pattern = r'\W|\d|http.*\s+|www.*\s+'

    # Aplicando padrão e removendo o texto
    preprocessor = lambda text: re.sub(pattern, ' ', text)

    # Vetorização de texto usando o termo frequência inversa de frequência do documento TF-IDF
    vectorizer = TfidfVectorizer(preprocessor = preprocessor, stop_words= 'english', min_df = MIN_DOC_FREQ)
    # cada palavra da sentença (frase/termo) terá um peso diferente, fará um count da freq e será normalizado

    # Reduzindo a dimensionalidade da matriz TF-IDF para não prejudicar o algoritmo de ML
    decomposition = TruncatedSVD(n_components= N_COMPONENTS, n_iter= N_ITER)
    # SVD = singular value decomposition
    # PCA = não utilizar - matriz ficará esparsa (com muitos valores iguais a zero). Usei SVD que retornará uma matriz densa


    # Pipeline para aplicar o tfidf e o svd
    pipeline = [('tfidf', vectorizer), ('svd', decomposition)]

    return pipeline

## Seleção do Modelo

# Variáveis de Controle = N_NEIGHBORS e CV (Cross Validation)
N_NEIGHBORS = 5 # número de "vizinhos mais próximos" como params
CV = 3 # três validações cruzadas

# Função criar modelos
def cria_modelos():

    modelo_1 = KNeighborsClassifier(n_neighbors= N_NEIGHBORS)
    modelo_2 = RandomForestClassifier(random_state= RANDOM_STATE)
    modelo_3 = LogisticRegressionCV(cv = CV, random_state= RANDOM_STATE)

    modelos = [("KNN", modelo_1), ("RandomForest", modelo_2), ("LogReg", modelo_3)]

    return modelos


## Treinamento e Avaliação do Model

# Função para o treinamento e avaliação dos modelos

def treina_avalia(modelos, pipeline, X_treino, X_teste, y_treino, y_teste):

    # Result em uma lista
    resultados = []

    # Loop
    for name, modelo in modelos:

        # Pipeline
        pipe = Pipeline(pipeline + [(name, modelo)])

        # Treinamento
        print(f'Treinando o modelo {name} com dados de treino...')
        pipe.fit(X_treino, y_treino)

        # Previsões com dados de teste
        y_pred = pipe.predict(X_teste)

        # Calcula métricas
        report = classification_report(y_teste, y_pred)
        print("Relatório de classificação\n", report)

        resultados.append([modelo, {'modelo': name, 'previsões': y_pred, 'report': report,}])

    return resultados

## Executando o Pipelina para todos os Modelos

# Pipeline de ML
if __name__=="__main__":

    # Carrega os dados
    data, labels = carrega_dados()

    # Divisão dos dados
    X_treino, X_teste, y_treino, y_teste = split_data()

    # Pipeline de pré-processamento
    pipeline = processing_pipeline()

    # Cria os modelos
    all_models = cria_modelos()

    # Treina e avalia os modelos
    resultados: list[list[dict[str, str | dict | Any] | Any]] = treina_avalia(all_models, pipeline, X_treino, X_teste, y_treino, y_teste)

print("Concluído com sucesso")



# DataViz

def plot_distribution():
    _, counts = np.unique(labels, return_counts = True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15,6), dpi=120)
    plt.title("Número de Posts por assunto")
    sns.barplot(x = assuntos, y = counts)
    plt.legend([' '.join([f.title(), f'- {c} posts']) for f,c in zip(assuntos, counts)])
    plt.show()


def plot_confusion(result):
    print("Relatório de Classificação\n", result[-1]['report'])
    y_pred = result[-1]['previsões']
    conf_matrix = confusion_matrix(y_teste, y_pred)
    _, test_counts = np.unique(y_teste, return_counts= True)
    conf_matrix_percent = conf_matrix / test_counts.transpose() * 100
    plt.figure(figsize= (9,8), dpi=120)
    #plt.title(result[-1]['model'].upper() + " Resultados")
    plt.title(result[-1]['modelo'].upper() + " Resultados")

    plt.xlabel("Valor Real")
    plt.ylabel("Previsão do Modelo")
    ticklabels = [f'r/{sub}' for sub in assuntos]
    sns.heatmap(data = conf_matrix_percent, xticklabels=ticklabels, yticklabels=ticklabels, annot=True, fmt = '.2f')

# Grafico de avaliação
plot_distribution()

# Resultado do KNN
plot_confusion(resultados[0])

# Resultado do RandomForest
plot_confusion(resultados[1])

# Resultado da RegLog
plot_confusion(resultados[2])

