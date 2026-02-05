from transformers import MarianMTModel, MarianTokenizer
import torch
import json

# Carga el modelo en GPU si está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# Claves que serán traducidas
TRANSLATABLE_KEYS = ['head', 'relation', 'tail']

# Diccionario para contar las traducciones realizadas por clave
translation_counts = {key: 0 for key in TRANSLATABLE_KEYS}

# Función para traducir un texto solo si está en inglés
def translate_text_if_english(text):
    if text:  # Verifica que el texto no esté vacío
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
            translated = model.generate(**inputs)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error de traducción para '{text}': {e}")
    return text  # Devuelve el texto original si hay error

# Nombre del archivo JSON de entrada y salida
filename = "C:.....json"
output_filename = "C:......json"

# Carga el archivo JSON de entrada
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Itera sobre cada entrada de la lista
for entry in data:
    if isinstance(entry, dict):
        for key in TRANSLATABLE_KEYS:
            if key in entry:
                original_text = entry[key]
                translated_text = translate_text_if_english(original_text)
                
                if translated_text != original_text:
                    translation_counts[key] += 1  # Cuenta la traducción
                
                entry[key] = translated_text

# Guarda el archivo JSON modificado
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

# Imprime estadísticas de traducción
print(f"Archivo '{output_filename}' actualizado correctamente con las traducciones.")
print("Estadísticas de traducción:")
for key, count in translation_counts.items():
    print(f" - {key}: {count} traducciones realizadas")

# Muestra el total de traducciones por clave
total_traducciones = sum(translation_counts.values())
print(f"\nTotal de traducciones realizadas: {total_traducciones}")
print(f"Desglose por clave:")
for key in TRANSLATABLE_KEYS:
    print(f"{key}: {translation_counts[key]}")
