import numpy as np
import matplotlib.pyplot as plt

# Datos
X = np.array([[0,0],
              [0,1],
              [1,0],
              [2,2],
              [2,3],
              [3,2]])

y = np.array([0,0,0,1,1,1])

# Añadimos columna de 1's para el sesgo
X_bias = np.c_[np.ones(X.shape[0]), X]

# Parámetros
w = np.random.rand(X_bias.shape[1]) - 0.5   # pesos iniciales
eta = 0.1
epochs = 20

# Entrenamiento
for _ in range(epochs):
    for i in range(len(X_bias)):
        z = np.dot(w, X_bias[i])
        y_pred = 1 if z >= 0 else 0
        error = y[i] - y_pred
        w = w + eta * error * X_bias[i]

print("Pesos finales:", w)

# --------- GRAFICAR ------------
for i, point in enumerate(X):
    if y[i] == 0:
        plt.scatter(point[0], point[1], color="red", marker="o", label="Clase 0" if i==0 else "")
    else:
        plt.scatter(point[0], point[1], color="blue", marker="x", label="Clase 1" if i==3 else "")

# Recta de decisión: w0 + w1*x1 + w2*x2 = 0
x_vals = np.linspace(-1,4,100)
y_vals = -(w[0] + w[1]*x_vals) / w[2]
plt.plot(x_vals, y_vals, 'k--', label="Frontera")

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()

# --------- PRUEBAS NUEVAS ---------
test_points = np.array([[1,1], [3,3], [0,2]])
X_test = np.c_[np.ones(test_points.shape[0]), test_points]

for i, pt in enumerate(test_points):
    pred = 1 if np.dot(w, X_test[i]) >= 0 else 0
    print(f"Punto {pt} => clase predicha: {pred}")
