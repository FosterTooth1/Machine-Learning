import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def leer_archivo_texto(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        return archivo.readlines()  # Cambiado a readlines para obtener una lista de líneas (tweets)
    
def normalizar_texto(texto):
    return texto.lower()

def menciones(texto):
    return re.sub(r'@\w+', '', texto)

def urls(texto):
    return re.sub(r'http\S+|www.\S+', '', texto)

def caracteres_especiales(texto):
    return re.sub(r'[^a-zA-Z0-9\s]', '', texto)

def eliminar_id_fecha_hora(texto):
    texto = re.sub(r'\d{18}|\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', texto)
    return texto

def tokenizar(texto):
    return word_tokenize(texto)

def stopwoords(tokens):
    stpw = set(stopwords.words('spanish'))
    return [token for token in tokens if token not in stpw]

def stemming(tokens):
    stm = PorterStemmer()
    return [stm.stem(token) for token in tokens]

def tokens_a_texto(tokens, separador=' '):
    return separador.join(tokens)

def procesar_texto(texto):
    texto = eliminar_id_fecha_hora(texto)
    texto = menciones(texto)
    texto = urls(texto)
    texto = caracteres_especiales(texto)
    texto = normalizar_texto(texto)
    tokens = tokenizar(texto)
    tokens = stopwoords(tokens)
    tokens = stemming(tokens)
    return tokens_a_texto(tokens)

def guardar_texto_en_archivo(texto, ruta_archivo):
    with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
        archivo.write(texto)

def main():
    tweets = leer_archivo_texto('tweets_emo_negativas.txt')
    tweets_procesados = [procesar_texto(tweet) for tweet in tweets]
    texto_final = '\n'.join(tweets_procesados)  # Une todos los tweets procesados con un salto de línea
    print('Este es el texto preprocesado:')
    print(texto_final)  # Muestra el texto final conservando el formato original

    # Guarda el texto procesado en un archivo
    guardar_texto_en_archivo(texto_final, 'Tweets_processed.txt')
   
main()
