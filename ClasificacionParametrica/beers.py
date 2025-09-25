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
print(“costo luego de iteración", i, "es : ", costo)
return W, B