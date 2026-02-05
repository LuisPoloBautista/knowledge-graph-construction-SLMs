import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class JSONComparator:
    def __init__(self, archivos):
        """
        Inicializa el comparador con nombres de archivos personalizados
        
        Args:
            archivos (dict): Diccionario con nombres de modelos como clave y rutas de archivos como valor
        """
        self.archivos = archivos
        self.datos_json = {}
        self.nombres_modelos = list(archivos.keys())
        
    def cargar_json(self):
        """Carga los archivos JSON"""
        for modelo, archivo in self.archivos.items():
            with open(archivo, 'r', encoding='utf-8') as f:
                self.datos_json[modelo] = json.load(f)
    
    def obtener_elementos(self):
        """
        Obtiene elementos de cada archivo con un método de comparación más sofisticado
        
        Returns:
            dict: Conjunto de elementos para cada modelo
        """
        elementos_sets = {}
        for modelo, data in self.datos_json.items():
            elementos = set()
            for i, item in enumerate(data):
                # Log missing values
                for field in ['head', 'tail', 'relation']:
                    if not item.get(field):
                        print(f"Warning: Missing {field} in {modelo}, record {i}")
    
                elementos.add((
                    (item.get('head', '') or '').lower() if isinstance(item.get('head', ''), str) else '',
                    item.get('head_type', '') or '',
                    (item.get('relation', '') or '').lower() if isinstance(item.get('relation', ''), str) else '',
                    (item.get('tail', '') or '').lower() if isinstance(item.get('tail', ''), str) else '',
                    item.get('tail_type', '') or ''
                ))
            elementos_sets[modelo] = elementos
        return elementos_sets
    
    def medir_solapamiento(self, set1, set2):
        """
        Calcula el solapamiento con un método más preciso
        
        Args:
            set1 (set): Primer conjunto de elementos
            set2 (set): Segundo conjunto de elementos
        
        Returns:
            float: Porcentaje de solapamiento
        """
        # Usar Jaccard Index con normalización adicional
        interseccion = set1.intersection(set2)
        union = set1.union(set2)
        return (len(interseccion) / len(union)) * 100 if len(union) > 0 else 0
    
    def comparar_archivos(self):
        """
        Compara archivos y genera resultados detallados
        
        Returns:
            tuple: DataFrames de resultados y matrices de solapamiento
        """
        self.cargar_json()
        elementos_sets = self.obtener_elementos()
        
        # Preparar listas para resultados
        resultados = []
        matriz_solapamiento = []
        matriz_diferentes = []
        
        # Comparación completa, omitiendo comparaciones consigo mismo
        for i, modelo1 in enumerate(self.nombres_modelos):
            fila_solapamiento = []
            fila_diferentes = []
            for j, modelo2 in enumerate(self.nombres_modelos):
                if i != j:
                    solapamiento = self.medir_solapamiento(
                        elementos_sets[modelo1], 
                        elementos_sets[modelo2]
                    )
                    elementos_diferentes = len(
                        elementos_sets[modelo1].symmetric_difference(elementos_sets[modelo2])
                    )
                    total_elementos = len(
                        elementos_sets[modelo1].union(elementos_sets[modelo2])
                    )
                    porcentaje_diferentes = (elementos_diferentes / total_elementos) * 100
                    
                    resultados.append({
                        'Modelo 1': modelo1,
                        'Modelo 2': modelo2,
                        'Porcentaje Solapamiento': solapamiento,
                        'Porcentaje Diferentes': porcentaje_diferentes
                    })
                    
                    fila_solapamiento.append(solapamiento)
                    fila_diferentes.append(porcentaje_diferentes)
                else:
                    # Para la diagonal, ponemos 0
                    fila_solapamiento.append(0)
                    fila_diferentes.append(0)
            
            matriz_solapamiento.append(fila_solapamiento)
            matriz_diferentes.append(fila_diferentes)
        
        # Crear DataFrame de resultados
        df_resultados = pd.DataFrame(resultados)
        
        return df_resultados, np.array(matriz_solapamiento), np.array(matriz_diferentes)
    
    def visualizar_heatmap(self, matriz_solapamiento, matriz_diferentes):
        """
        Genera heatmaps de solapamiento y elementos diferentes
        
        Args:
            matriz_solapamiento (np.array): Matriz de solapamiento
            matriz_diferentes (np.array): Matriz de elementos diferentes
        """
        plt.figure(figsize=(16, 6))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(matriz_solapamiento, 
                    annot=True, 
                    cmap='YlGnBu', 
                    xticklabels=self.nombres_modelos, 
                    yticklabels=self.nombres_modelos,
                    fmt='.2f')
        plt.title('Porcentaje de Solapamiento')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(matriz_diferentes, 
                    annot=True, 
                    cmap='YlOrRd', 
                    xticklabels=self.nombres_modelos, 
                    yticklabels=self.nombres_modelos,
                    fmt='.2f')
        plt.title('Porcentaje de Elementos Diferentes')
        
        plt.tight_layout()
        plt.show()
    
    def ejecutar_analisis(self):
        """Método principal para ejecutar todo el análisis"""
        df_resultados, matriz_solapamiento, matriz_diferentes = self.comparar_archivos()
        
        # Imprimir tabla de resultados
        print("Resultados de Comparación:")
        print(df_resultados.to_string(index=False))
        
        # Generar visualización
        self.visualizar_heatmap(matriz_solapamiento, matriz_diferentes)

# Ejemplo de uso
if __name__ == "__main__":
    # Definir el directorio base donde están los archivos JSON
    base_dir = r"C:/Users/LUIS VILCHES/Desktop/"
    
    archivos = {
        "Gemma 2": os.path.join(base_dir, "tripletas_gemma_lemma.json"),
        "Llama 3.1": os.path.join(base_dir, "tripletas_llama_lemma.json"),
        "OLMO": os.path.join(base_dir, "tripletas_olmo_lemma.json"),
        "GPT-4o": os.path.join(base_dir, "tripletas_gpt_resumidas.json"),
        "Mistral:7b": os.path.join(base_dir, "tripletas_mistral_vec_tail.json"),
        "Mixtral:8x22b": os.path.join(base_dir, "tripletas_mixtral_vec_tail.json")
    }
    
    # Verificar que los archivos existen
    for modelo, ruta in archivos.items():
        if not os.path.exists(ruta):
            print(f"Advertencia: No se encuentra el archivo para {modelo}: {ruta}")
    
    comparador = JSONComparator(archivos)
    comparador.ejecutar_analisis()