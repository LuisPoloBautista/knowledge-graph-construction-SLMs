import pandas as pd

df_filtro_mx = pd.read_csv('C:/Users/LUIS VILCHES/Desktop/nz_earthquake.csv', encoding="latin9")
df_filtro_mx.info()


import itertools
import re

df_filtro_mx['text']=df_filtro_mx['text'].apply(str)


def clean_text(texto):
    # Eliminar caracteres HTML dentro de etiquetas <>
    texto = re.sub(r'<[^>]*?>', '', texto)
    
    # Eliminar texto que parece ser HTML o URLs, incluyendo patrones como "httpstcovtRWA6HTh3"
    texto = re.sub(r'\b(?:https?|ftp|file|www)\S*\b', '', texto)
    
    # Separar palabras pegadas (de minúscula a mayúscula)
    texto = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', texto)
    
    # Eliminar puntuación
    texto = re.sub(r"[^\w\s]", "", texto)
    
    # Eliminar guiones bajos y otros caracteres específicos
    texto = re.sub("_", "", texto)
    texto = re.sub(r'\+', '', texto)
    texto = re.sub(r':', '', texto)
    
    # Eliminar cualquier texto que contenga una estructura similar a URLs o pseudo-HTML
    texto = re.sub(r'\b\w{4,5}tco\w+\b', '', texto)
    
    return texto



df_filtro_mx['text'] = df_filtro_mx['text'].apply(clean_text)
df_filtro_mx['text']


df_filtro_mx.to_csv('nz_corpus.csv', encoding="latin9", index=False)