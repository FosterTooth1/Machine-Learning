import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support

# Leer el archivo de tweets preprocesados
with open('Tweets_processed.txt', 'r', encoding='utf-8') as file:
    tweets = file.readlines()

# Función para etiquetar automáticamente los tweets basada en palabras clave
def label_tweet(tweet):
    if 'ira' in tweet:
        return 'Ira'
    elif 'miedo' in tweet:
        return 'Miedo'
    elif 'tristeza' in tweet:
        return 'Tristeza'
    else:
        return 'Otro'  # Para tweets que no contengan ninguna de las palabras clave

# Etiquetar los tweets
labels = [label_tweet(tweet) for tweet in tweets]

# Eliminar las palabras clave de los tweets
clean_tweets = [re.sub(r'\b(ira|miedo|tristeza)\b', '', tweet) for tweet in tweets]

# Crear los vectores TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
X = vectorizer.fit_transform(tweets)
feature_names = vectorizer.get_feature_names_out()

# Mostrar cómo cada tweet se representa en términos de TF-IDF
for i, tweet in enumerate(tweets):
    print(f"Tweet {i+1}: {tweet.strip()}")
    vector = [f"{feature_names[j]}, {X[i, j]:.3f}" for j in X[i].nonzero()[1]]
    print(f"Vector: {vector}\n")

# Convertir etiquetas a un DataFrame
y = pd.Series(labels)

# Función para imprimir métricas
def print_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

# Clasificador Naive Bayes
nb_model = MultinomialNB()
y_pred_nb = cross_val_predict(nb_model, X, y, cv=10)
print("Naive Bayes Classifier Metrics:")
print_metrics(y, y_pred_nb)

# Clasificador SVM
svm_model = SVC(kernel='linear')
y_pred_svm = cross_val_predict(svm_model, X, y, cv=10)
print("Support Vector Machine Classifier Metrics:")
print_metrics(y, y_pred_svm)
