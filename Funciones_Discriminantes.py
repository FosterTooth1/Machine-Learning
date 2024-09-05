#Flores Lara Alberto 5BV1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generar datos sintéticos
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un pipeline con transformación polinomial y regresión logística
model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('logistic_regression', LogisticRegression())
])

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir y evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Función para graficar la frontera de decisión
def graficar(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, colors=['#ffff66', '#6666ff'])
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.Paired, edgecolor='k')
    plt.title("Frontera de decisión")
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.show()

# Imprimir el valor de Accuracy
print("Accuracy:", accuracy)

# Graficar la frontera de decisión
graficar(model, X, y)
