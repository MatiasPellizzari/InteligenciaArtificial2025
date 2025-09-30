import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Función para entrenar y evaluar un modelo KNN con un valor de k
def evaluar_knn(k, X_train, X_test, y_train, y_test, X, y):
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== k = {k} ===")
    print("Accuracy:", acc)
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

    # Graficar frontera de decisión
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6,4))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.title(f"Frontera de decisión con k={k}")
    plt.show()


def main():
    # 1) Generar dataset sintético
    X, y = make_moons(n_samples=300, noise=0.35, random_state=42)

    # Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 2) Probar con distintos valores de k
    for k in [1, 3, 10, 30, 100]:
        evaluar_knn(k, X_train, X_test, y_train, y_test, X, y)


if __name__ == "__main__":
    main()
