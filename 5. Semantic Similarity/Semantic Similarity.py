import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter

filename = "C:/.....json"

# Cargar los datos JSON
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Cargar el modelo de embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# Obtener todos los "head_type"
head_types = [entry.get('tail') for entry in data if isinstance(entry, dict) and entry.get('tail')]

# Vectorizar los "head_type"
head_type_embeddings = model.encode(head_types)

# Función para visualizar embeddings
def visualizar_embeddings(embeddings, labels, title):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    plt.figure(figsize=(14, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='orange')
    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=6)
    plt.title(title)
    plt.show()

# Visualización antes de la unificación
visualizar_embeddings(head_type_embeddings, head_types, 'Embeddings de head_type (Antes de Unificar)')

# Métricas antes de la unificación
unique_head_types_before = set(head_types)
print(f"Cantidad de tail únicos antes de la unificación: {len(unique_head_types_before)}")

# Calcular la matriz de similitud coseno
similarity_matrix = cosine_similarity(head_type_embeddings)

# Obtener estadísticas de similitud coseno
similarities = similarity_matrix[np.triu_indices(len(head_types), k=1)]  # Excluir diagonal
print(f"Media de similitudes coseno antes de la unificación: {np.mean(similarities):.4f}")
print(f"Mediana de similitudes coseno antes de la unificación: {np.median(similarities):.4f}")

# Unificación de palabras semánticamente similares con umbral 0.9
similarity_threshold = 0.9
unified_map = {}

for i, head_type in enumerate(head_types):
    if head_type not in unified_map:
        similarities = similarity_matrix[i]
        similar_indices = [j for j in range(len(similarities)) if similarities[j] >= similarity_threshold]
        similar_words = [head_types[j] for j in similar_indices]
        representative = min(similar_words, key=lambda x: head_types.index(x))  # Elegir el primero que aparece
        for word in similar_words:
            unified_map[word] = representative

# Aplicar la unificación en el JSON
data_unified = data.copy()
for entry in data_unified:
    if isinstance(entry, dict) and entry.get('tail'):
        entry['tail'] = unified_map[entry['tail']]

# Guardar el JSON modificado
with open('C:/...json', 'w', encoding='utf-8') as f:
    json.dump(data_unified, f, ensure_ascii=False, indent=4)

# Métricas después de la unificación
unified_head_types = set(unified_map.values())
print(f"Cantidad de head_type únicos después de la unificación: {len(unified_head_types)}")
print(f"Reducción porcentual en la cantidad de head_type: {100 * (1 - len(unified_head_types) / len(unique_head_types_before)):.2f}%")

# Calcular nueva matriz de similitud coseno tras la unificación
unified_head_type_embeddings = model.encode(list(unified_head_types))
unified_similarity_matrix = cosine_similarity(unified_head_type_embeddings)

# Obtener estadísticas de similitud coseno después de la unificación
unified_similarities = unified_similarity_matrix[np.triu_indices(len(unified_head_types), k=1)]
print(f"Media de similitudes coseno después de la unificación: {np.mean(unified_similarities):.4f}")
print(f"Mediana de similitudes coseno después de la unificación: {np.median(unified_similarities):.4f}")

# Visualización después de la unificación
visualizar_embeddings(unified_head_type_embeddings, list(unified_head_types), 'Embeddings de head_type (Después de Unificar)')

# Mostrar las palabras que se unificaron
print("Palabras unificadas:")
for original, unified in unified_map.items():
    if original != unified:
        print(f"{original} --> {unified}")
