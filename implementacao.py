
import zipfile
import requests
import os
import pandas as pd
from datetime import datetime
import re
import warnings
import subprocess
warnings.filterwarnings("ignore")
from collections import OrderedDict

from elasticsearch import Elasticsearch

from sentence_transformers import SentenceTransformer

import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import sklearn

nltk.download('punkt_tab')
nltk.download('stopwords')

import spacy
import unidecode


subprocess.call(["docker", "compose", "up", "-d"])
# Se esse comando falhar ou retornar 1, execute-o diretamente no terminal para identificar o erro.
# PS: O docker deve estar instalado e rodando



# Base de dados de notícias da Lupa
url = "https://docs.google.com/uc?export=download&confirm=t&id=1W067Md2EbvVzW1ufzFg17Hf7Y9cCZxxr"
filename = "articles_lupa_lab_elasticsearch.zip"
data_path = "data"
zip_file_path = f"{data_path}/{filename}"

os.makedirs(data_path, exist_ok=True)

# Baixa o zip
with open(zip_file_path, "wb") as f:
    f.write(requests.get(url, allow_redirects=True).content)

# Extrai o csv do zip
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(data_path)
    
    output_file = f"{data_path}/articles_lupa.csv"
assert os.path.exists(output_file)



#PREPROCESSAMENTO

# Implementações de pré-processamentos de texto. Modifiquem, adicionem, removam conforme necessário.
class Preprocessors:
    STOPWORDS = set(nltk.corpus.stopwords.words('portuguese'))

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.spacy_nlp = spacy.load("pt_core_news_sm")  # Utiliza para lematização

    # Remove stopwords do português
    def remove_stopwords(self, text):
        # Tokeniza as palavras
        tokens = word_tokenize(text)
        # Remove as stop words
        tokens = [word for word in tokens if word not in self.STOPWORDS]

        return ' '.join(tokens)

    # Realiza a lematização
    def lemma(self, text):
        return " ".join([token.lemma_ for token in self.spacy_nlp(text)])

    # Realiza a stemização
    def porter_stemmer(self, text):
        # Tokeniza as palavras
        tokens = word_tokenize(text)

        for index in range(len(tokens)):
            # Realiza a stemização
            stem_word = self.stemmer.stem(tokens[index])
            tokens[index] = stem_word

        return ' '.join(tokens)

    # Transforma o texto em lower case
    def lower_case(self, str):
        return str.lower()

    # Remove urls com regex
    def remove_urls(self, text):
        url_pattern = r'https?://\S+|www\.\S+'
        without_urls = re.sub(pattern=url_pattern, repl=' ', string=text)
        return without_urls

    # Remove números com regex
    def remove_numbers(self, text):
        number_pattern = r'\d+'
        without_number = re.sub(pattern=number_pattern,
                                repl=" ", string=text)
        return without_number

    # Converte caracteres acentuados para sua versão ASCII
    def accented_to_ascii(self, text):
        text = unidecode.unidecode(text)
        return text


# Carregar o modelo gerador de embeddings
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Caminho para salvar o dataframe de notícias
data_df_path = "data/data_df.pkl"

# Selecione diferentes pré-processamentos
# Exemplo:
"""
preprocessor = Preprocessors()
preprocessing_steps = [
    preprocessor.remove_urls,
    preprocessor.remove_stopwords,
]
"""

preprocessor = Preprocessors()

preprocessing_steps = [
    # Adicione os pré-processamentos aqui
    preprocessor.lower_case,
    preprocessor.remove_stopwords,
    preprocessor.remove_urls
]

RECREATE_DF = True

# Cria o data frame se ele já existir ou se a variável RECREATE_INDEX for verdadeira
# Ou (exclusivo) carrega o dataframe salvo
if not os.path.exists(data_df_path) or RECREATE_DF:
    df = pd.read_csv(output_file, sep=";")[["Título", "Texto", "Data de Publicação"]]
    df["Data de Publicação"] = df["Data de Publicação"].apply(
        lambda str_date: datetime.strptime(str_date.split(" - ")[0], "%d.%m.%Y"))
    df.sort_values("Data de Publicação", inplace=True, ascending=False)
    df["Embeddings"] = [None] * len(df)
    df["doc_id"] = df.reset_index(drop=True).index

    for i, row in df.iterrows():
        texto_completo = row["Texto"].strip() + "\n" + row["Título"].strip()

        df.at[i, "Texto completo"] = texto_completo
        texto_processado = texto_completo
        for preprocessing_step in preprocessing_steps:
            texto_processado = preprocessing_step(texto_processado)

        df.at[i, "Texto processado"] = texto_processado
        embeddings = model.encode(texto_completo).tolist()
        df.at[i, "Embeddings"] = embeddings

    print("Geração de embeddings finalizada.")

    with open(data_df_path, "wb") as f:
        df.to_pickle(f)
else:
    with open(data_df_path, "rb") as f:
        df = pd.read_pickle(f)
    print("Dataframe carregado de arquivo.")


es = Elasticsearch(
    hosts = [{'host': "localhost", 'port': 9200, "scheme": "https"}],
    basic_auth=("elastic" ,"elastic"),
    verify_certs = False,
)

RECREATE_INDEX = True

index_name = "verificacoes_lupa"

# Se a flag for True e se o índice existir, ele é deletado
if RECREATE_INDEX and es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"Índice '{index_name}' deletado.")

# Cria o índice e popula com os dados
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, mappings={
        "properties": {
            "doc_id": {"type": "integer"},
            "full_text": {"type": "text"},
            "processed_text": {"type": "text"},
            "embeddings": {"type": "dense_vector", "dims": 384}
        }
    })
    print(f"Índice '{index_name}' criado.")

    for i, row in df.iterrows():
        es.index(index=index_name, id=row["doc_id"], body={
            "doc_id": row["doc_id"],
            "full_text": row["Texto completo"],
            "processed_text": row["Texto processado"],
            "embeddings": row["Embeddings"]
        })
    print("Índice preenchido.")

print("Indexação finalizada.")

# Estas serão as queries QF1 e QF2
with open("data/queries_fixadas.txt", "r") as f:
    queries_fixadas = [line.strip() for line in f.readlines()]
    assert len(queries_fixadas) == 2
    QF1 = queries_fixadas[0]
    QF2 = queries_fixadas[1]

# Preencha aqui as queries do grupo
QP1 = "Candidatos politicos estão ligados com o crime organizado."
QP2 = "indicados ao Oscar que ganharam premiação na cerimônia."

queries = OrderedDict()
queries["QF1"] = QF1
queries["QF2"] = QF2
queries["QP1"] = QP1
queries["QP2"] = QP2


# Implementação de busca esparsa (léxica) com BM25
def lexical_search(queries: dict[str, str]):
    lexical_results = {}
    for query_id, query in queries.items():

        # Pré-processa os dados
        for preprocessing_step in preprocessing_steps:
            query = preprocessing_step(query)

        search_query = {
            "query": {
                "match": {
                    "processed_text": query
                }
            }
        }

        # Realiza a busca
        response = es.search(index=index_name, body=search_query)

        hits_results = []
        # Recupera os resultados
        for hit in response["hits"]["hits"]:
            hits_results.append((hit["_source"]["doc_id"], hit["_score"]))
        lexical_results[query_id] = hits_results

    return lexical_results


# Realiza busca semântica (densa) com KNN exato
def semantic_search(queries: dict[str, str]):
    semantic_results = {}

    for query in queries:
        # Aplica todos os pré-processamentos aos dados
        for preprocessing_step in preprocessing_steps:
            query = preprocessing_step(query)

        query_vector = model.encode(query).tolist()

        search_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }

        # Realiza a busca
        response = es.search(index=index_name, body=search_query)

        hits_results = []
        # Recupera top 10 resultados
        for hit in response["hits"]["hits"]:
            hits_results.append((hit["_source"]["doc_id"], hit["_score"]))

        semantic_results[query] = hits_results


    return semantic_results



# Busca híbrida ou RRF. Implemente sua solução aqui. Você pode realizar as duas buscas anteriores (léxica e semântica) como base para formar a busca híbrida.
def hybrid_search(queries: dict[str, str]):
    ## TODO: Implementar busca híbrida
    pass

# Implemente sua própria estratégia de busca, podendo ela ser esparsa, densa ou híbrida. Implemente algo como "more_like_this", "BM35", "fuzzy" etc.
def creative_search(queries: dict[str, str]):
    ## TODO: Implementar busca híbrida
    result = {}
    for query_id, query in queries.items():
        for preprocessing_step in preprocessing_steps:
            query = preprocessing_step(query)
        """
        search_query = {
            "query": {
                "match": {
                    "processed_text": query
                }
            },
            "aggs": {
                "unique_docs": {
                    "terms": {
                        "field": "doc_id",
                    }
                }
            }
        }
        """

        # Consulta BM25
        bm25_query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "processed_text": {
                                    "query": query,
                                    "boost": 1.0  # Ajuste o boost conforme necessário
                                }
                            }
                        }
                    ]
                }
            }
        }

        # Consulta Fuzzy
        fuzzy_query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "fuzzy": {
                                "processed_text": {
                                    "value": query,
                                    "fuzziness": "AUTO",  # Ou ajuste conforme necessário
                                    "boost": 1.0  # Ajuste o boost conforme necessário
                                }
                            }
                        }
                    ]
                }
            }
        }

        # Combinando as duas consultas (BM25 e Fuzzy) em uma consulta complexa
        combined_query = {
            "query": {
                "bool": {
                    "should": [
                        bm25_query["query"]["bool"]["should"][0],  # BM25
                        fuzzy_query["query"]["bool"]["should"][0]  # Fuzzy
                    ],
                    "minimum_should_match": 1  # Para garantir que pelo menos uma das consultas seja atendida
                }
            }
        }

        response = es.search(
            index=index_name,
            body=combined_query
        )

        hits_results = []
        # Recupera os resultados
        for hit in response["hits"]["hits"]:
            hits_results.append((hit["_source"]["doc_id"], hit["_score"]))

        result[query_id] = hits_results
    return result

search_functions = [
    ("lexical", lexical_search),
    ("semantic", semantic_search),
    ("hybrid", hybrid_search),
    ("creative", creative_search)
]

def run_all_searches(queries: dict[str, str]):
    all_search_results = {}
    for search_name, search_function in search_functions:
        results = search_function(queries)
        all_search_results[search_name] = results
    return all_search_results


all_search_results = run_all_searches(queries)
search_results_df = pd.DataFrame(all_search_results)
search_results_df


def generate_exploded_df(search_results_df):
    exploded_search_results_df = pd.concat([search_results_df[col].explode() for col in search_results_df.columns],
                                           axis=1)
    exploded_search_results_df = exploded_search_results_df.apply(lambda l: [doc_id for doc_id, _ in l])
    return exploded_search_results_df


def generate_found_docs_text_df(exploded_search_results_df, all_docs_df):
    # Recupera os ids únicos dos documentos
    documents_ids = set(exploded_search_results_df.to_numpy().flatten().tolist())

    # Salva os textos e os ids dos documetnos que foram encontrados ems usas buscas
    documents_df = all_docs_df[all_docs_df["doc_id"].isin(documents_ids)][["Texto processado", "doc_id"]]
    return documents_df


exploded_df = generate_exploded_df(search_results_df)
found_docs_text_df = generate_found_docs_text_df(exploded_df, all_docs_df=df)


def save_results_to_file(exploded_df: pd.DataFrame,
                         found_docs_text_df: pd.DataFrame,
                         exploded_df_save_filepath: str = "data/search_results.csv",
                         found_docs_text_save_filepath: str = "data/documents.csv"):
    exploded_df.to_csv(exploded_df_save_filepath)
    found_docs_text_df.to_csv(found_docs_text_save_filepath)
    print("Resultados das buscas salvos em 'data/search_results.csv'.")
    print("Documentos de interesse salvos em 'data/documents.csv'.")


save_results_to_file(exploded_df, found_docs_text_df)
exploded_df
