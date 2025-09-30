import numpy as np
import matplotlib.pyplot as plt
# Datos: (IBU, RMS) y clase
X = np.array([
[15, 20], # Lager
[12, 15], #...
[28, 39],
[21, 30],
[45, 20], # Stout
[40, 61],
[42, 70] #...
])
# 0 = Lager, 1 = Stout
y = np.array([0,0,0,0,1,1,1])
# Funcion sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Modelo

def modelo(X, Y, learning_rate, iterations):
    X = X.T #trasponemos para realizar la multiplicación de matrices
    n = X.shape[0] #cantidad de características
    m = X.shape[1] #cantidad de casos
    W = np.zeros((n,1)) #vector de pesos para cada característica
    B = 0
    for i in range(iterations):
        Z = np.dot(W.T, X) + B #mult pesos por casos
        A = sigmoid(Z) #tenemos el vector de resultados de cada caso
        # Función de costo
        costo = -(1/m)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A))
        # Aplicación de la técnica Gradient Descent
        dW = (1/n)*np.dot(A-Y, X.T)
        dB = (1/n)*np.sum(A - Y)
        # Ajuste de pesos
        W = W - learning_rate * dW.T
        B = B - learning_rate * dB
        if(i%(iterations/10) == 0):
            print("costo luego de iteración", i, "es : ", costo)
    print(W,B) 
    return W, B

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Entrenar modelo
    W, B = modelo(X, y, learning_rate=0.01, iterations=1000)
    print("Pesos finales:", W.flatten(), "Bias:", B)

    # Predicción en nuevos puntos
    def predecir(punto):
        z = np.dot(W.T, punto.T) + B
        return 1 if sigmoid(z) >= 0.5 else 0

    test_points = np.array([[18,22], [35,50], [25,25]])
    for pt in test_points:
        print(f"Punto {pt} -> clase predicha: {predecir(pt)}")

    # Graficar datos y frontera
    for i in range(len(y.flatten())):
        if y.flatten()[i] == 0:
            plt.scatter(X[i,0], X[i,1], color="red", marker="o", label="Lager" if i==0 else "")
        else:
            plt.scatter(X[i,0], X[i,1], color="blue", marker="x", label="Stout" if i==4 else "")

    # Frontera de decisión (W1*x1 + W2*x2 + B = 0.5)
    x_vals = np.linspace(10,50,100)
    y_vals = -(W[0]*x_vals + B - np.log(0.5/(1-0.5)))/W[1]
    plt.plot(x_vals, y_vals.flatten(), 'k--', label="Frontera")

    plt.xlabel("IBU")
    plt.ylabel("RMS")
    plt.legend()
    plt.grid(True)
    plt.show()