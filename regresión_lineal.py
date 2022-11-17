import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

!gdown '1YB1vIna6NmitE36EmqO1ZwfE3ViQopeb' -O dataset.csv

datos = pd.read_csv("dataset.csv")

x = np.array(datos["x"])
y = np.array(datos["y"])
m = np.size(x)
media = np.mean(x)
sigma = np.std(x)
x = (x - media) / sigma

plt.title("Regresion lineal")
plt.plot(x,y, 'ok', color = "blue")
plt.xlabel("x: area")
plt.ylabel("y: costo")

a0 = 1000000
a1 = 0
beta = 0.9
iter = 1
iterMax = 100
h = a0 + a1 * x 

plt.plot(x, h, 'r')

J = (1/(2*m)) * sum(np.power((h-y), 2))

convergencia = np.zeros(iterMax)
convergencia[0] = J
while iter < iterMax:
  a0 = a0 -beta * (1/m) * sum(h-y)
  a1 = a1 -beta * (1/m) * sum((h-y)*x)
  h = a0 + a1 * x
  J = (1/(2*m)) * sum(np.power((h-y), 2))
  convergencia[iter] = J
  iter += 1

plt.plot(x, h, 'g')

plt.figure(2)
plt.plot(convergencia)
plt.title("Gráfico de convergencia")
plt.xlabel("Iteraciones")
plt.ylabel("Función de costo J (error)")
print("a0 = ", a0, "a1 = ", a1)
