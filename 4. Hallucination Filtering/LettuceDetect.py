import pandas as pd
import gc
import json
from tqdm import tqdm
import time
from lettucedetect.models.inference import HallucinationDetector
from transformers import BertModel, BertConfig  # Changed from ModernBert to Bert

# Initializing a BERT style configuration
configuration = BertConfig()

# Initializing a model from the bert-base style configuration
model = BertModel(configuration)

# Accessing the model configuration
configuration = model.config

# Initialize the hallucination detector globally
detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
)

# Función para convertir una tripleta en una oración o afirmación legible
def convertir_tripleta_a_oracion(tripleta):
    try:
        head = str(tripleta.get('head', ''))
        relation = str(tripleta.get('relation', ''))
        tail = str(tripleta.get('tail', ''))

        # Si alguna de las claves esenciales no está presente, omitir la tripleta
        if not head or not relation or not tail:
            raise KeyError("Tripleta incompleta")
        
        return f"{head} {relation} {tail}."
    except KeyError as e:
        # Imprimir un mensaje de depuración y continuar con la siguiente tripleta
        print(f"Tripleta ignorada: {e}")
        return ""
    except Exception as e:
        print(f"Error al procesar la tripleta: {e}")
        return ""

# Función para verificar si una afirmación está respaldada por la noticia
def verificar_alucinacion(noticia, oracion):
    try:
        # Usar LettuceDect para detectar alucinaciones
        predictions = detector.predict(
            context=[noticia],
            question="¿Qué información contiene el texto?",
            answer=oracion,
            output_format="spans"
        )
        
        # Si hay predicciones y la confianza es alta (>0.5), consideramos que es una alucinación
        if predictions and any(pred['confidence'] > 0.5 for pred in predictions):
            return 1  # Es una alucinación
        return 0  # No es una alucinación
    except Exception as e:
        print(f"Error al verificar la afirmación: {e}")
        return 0

# Función para procesar cada fila y calcular el porcentaje de alucinación por tripleta
def procesar_fila(fila):
    noticia = fila['text']  # Asumiendo que el documento está en una columna llamada 'texto_completo'
    json_afirmacion = str(fila['TripletasLlama'])  # Aseguramos que el valor sea tratado como string

    try:
        afirmaciones_json = json.loads(json_afirmacion)
    except json.JSONDecodeError:
        print("JSON inválido, omitiendo fila.")
        return pd.Series([0, json.dumps([], ensure_ascii=False), 0])  # Si no se puede cargar el JSON, retornamos 0 y una lista vacía

    total_tripletas = len(afirmaciones_json)
    if total_tripletas == 0:
        return pd.Series([0, json.dumps([], ensure_ascii=False), 0])  # No hay tripletas para procesar

    alucinaciones_tripletas = 0
    detalles_tripletas = []  # Para almacenar los resultados de cada tripleta

    for tripleta in afirmaciones_json:
        oracion = convertir_tripleta_a_oracion(tripleta)
        if oracion:  # Verificamos solo si la oración no está vacía
            try:
                resultado_alucinacion = verificar_alucinacion(noticia, oracion)
                detalles_tripletas.append({'oracion': oracion, 'resultado': resultado_alucinacion})
                alucinaciones_tripletas += resultado_alucinacion
            except Exception as e:
                print(f"Error al procesar la oración '{oracion}': {e}")
                continue  # Pasar a la siguiente tripleta en caso de error

    # Calculamos el porcentaje de alucinación por tripleta
    porcentaje_alucinacion = (alucinaciones_tripletas / total_tripletas) * 100
    return pd.Series([porcentaje_alucinacion, json.dumps(detalles_tripletas, ensure_ascii=False), total_tripletas])  # Retornamos los detalles y el número de tripletas

# Cargar archivo localmente (modificar el nombre de archivo si es necesario)
filename = 'C:/.....csv'
data = pd.read_csv(filename, encoding="latin9")

# Seleccionar un subconjunto de datos para pruebas si es necesario
#df = data.iloc[0:942]

# Usar tqdm para mostrar el progreso del procesamiento
start_time = time.time()
tqdm.pandas(desc="Procesando filas")

# Aplicar la función con progreso
data[['porcentaje_alucinacion', 'detalles_tripletas', 'total_tripletas']] = data.progress_apply(procesar_fila, axis=1)

# Calcular métricas generales
promedio_porcentaje_alucinacion = data['porcentaje_alucinacion'].mean()
total_tripletas_evaluadas = data['total_tripletas'].sum()
total_documentos = len(data)

# Imprimir métricas finales
print(f"\nMétricas generales:")
print(f"Porcentaje promedio de alucinación en todos los documentos: {promedio_porcentaje_alucinacion:.2f}%")
print(f"Total de tripletas evaluadas: {total_tripletas_evaluadas}")
print(f"Total de documentos procesados: {total_documentos}")

# Guardar el nuevo CSV con la columna de porcentaje de alucinación y detalles de tripletas
data.to_csv('C:/.....csv', index=False, encoding="latin9")

end_time = time.time()
elapsed_time = (end_time - start_time) / 60  # Tiempo en minutos
print(f"\nArchivo procesado y guardado. Tiempo estimado: {elapsed_time:.2f} minutos")