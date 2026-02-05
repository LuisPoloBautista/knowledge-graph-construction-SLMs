import json
import requests

# Contadores de métricas
contador_encontradas = 0
contador_creadas = 0

# Función para buscar en Wikidata
def buscar_wikidata(entidad):
    global contador_encontradas
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={entidad}&language=es&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "search" in data and data["search"]:
            for result in data["search"]:
                if "id" in result:
                    contador_encontradas += 1  # Contar como encontrada
                    return f"https://www.wikidata.org/entity/{result['id']}"
    return None

# Función para generar URI automáticamente
def generar_uri(entidad):
    global contador_creadas
    contador_creadas += 1  # Contar cada entidad creada automáticamente
    return f"https://example.org/entity/{entidad.replace(' ', '_')}"

# Función para procesar un chunk
def procesar_chunk(chunk):
    for elemento in chunk:
        for clave in ["head", "relation", "tail", "head_type", "tail_type"]:
            valor = elemento.get(clave, "")
            if valor:
                uri_clave = f"{clave}_uri"
                uri = buscar_wikidata(valor) or generar_uri(valor)
                elemento[uri_clave] = uri
    return chunk

# Cargar JSON en chunks
def cargar_json_en_chunks(filename, chunk_size=100):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

# Guardar JSON actualizado
def guardar_json(data, output_file="C:/Users/LUIS VILCHES/Desktop/wikidata_gemma.json"):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Flujo principal
if __name__ == "__main__":
    filename = "C:/Users/LUIS VILCHES/Desktop/KG LLM y SML2/h_Embeddings/tripletas_gemma_lemma.json"
    json_actualizado = []
    
    for chunk in cargar_json_en_chunks(filename):
        json_actualizado.extend(procesar_chunk(chunk))
    
    guardar_json(json_actualizado)
    
    # Mostrar métricas finales
    print(f"URIs encontradas en Wikidata gpt: {contador_encontradas}")
    print(f"URIs generadas automáticamente gpt: {contador_creadas}")
