import random
import math
import matplotlib.pyplot as plt

# Generar datos sintéticos para dos clases normales
def generate_data(n, mean, cov):
    data = []
    for _ in range(n):
        x = random.gauss(mean[0], cov[0])
        y = random.gauss(mean[1], cov[1])
        data.append([x, y])
    return data

class0 = generate_data(100, [0, 0], [1, 1])
class1 = generate_data(100, [3, 3], [1, 1])
X_train = class0 + class1
y_train = [0]*100 + [1]*100

# Estimador no paramétrico: k-Nearest Neighbors
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

def knn_classifier(point, data, labels, k=5):
    distances = [(euclidean_distance(point, data[i]), labels[i]) for i in range(len(data))]
    distances.sort(key=lambda x: x[0])
    votes = [label for _, label in distances[:k]]
    return max(set(votes), key=votes.count)

# Visualización de la frontera de decisión
x_min, x_max = -4, 7
y_min, y_max = -4, 7
step = 0.3
xx, yy = [], []
zz = []

x = x_min
while x <= x_max:
    y = y_min
    while y <= y_max:
        xx.append(x)
        yy.append(y)
        zz.append(knn_classifier([x, y], X_train, y_train, k=5))
        y += step
    x += step

# Gráfico
plt.figure(figsize=(8, 6))
plt.scatter([p[0] for p in class0], [p[1] for p in class0], color='blue', label='Clase 0', alpha=0.6)
plt.scatter([p[0] for p in class1], [p[1] for p in class1], color='red', label='Clase 1', alpha=0.6)
plt.scatter(xx, yy, c=zz, cmap='coolwarm', alpha=0.15, s=10)
plt.title('Clasificador k-NN (k=5)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
