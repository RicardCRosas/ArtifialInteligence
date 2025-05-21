import numpy as np
import matplotlib.pyplot as plt
import math
import random

# --------------------------
# 1. Generar datos sintéticos
# --------------------------
def generate_data(n, mean, std_dev):
    return [[random.gauss(mean[0], std_dev[0]), random.gauss(mean[1], std_dev[1])] for _ in range(n)]

class0 = generate_data(100, [0, 0], [1, 1])
class1 = generate_data(100, [3, 3], [1, 1])

X_train = class0 + class1
y_train = [0]*100 + [1]*100

# --------------------------
# 2. Kernel Gaussiano para Parzen
# --------------------------
def gaussian_kernel(x, xi, h):
    diff = np.array(x) - np.array(xi)
    norm_sq = np.dot(diff, diff)
    return math.exp(-norm_sq / (2 * h**2))

# --------------------------
# 3. Clasificador Parzen
# --------------------------
def parzen_classifier(x, data, labels, h=1.0):
    total_0 = sum(gaussian_kernel(x, data[i], h) for i in range(len(data)) if labels[i] == 0)
    total_1 = sum(gaussian_kernel(x, data[i], h) for i in range(len(data)) if labels[i] == 1)
    return 0 if total_0 > total_1 else 1

# --------------------------
# 4. Visualización de frontera
# --------------------------
x_min, x_max = -4, 7
y_min, y_max = -4, 7
step = 0.3

xx, yy, zz = [], [], []

x = x_min
while x <= x_max:
    y = y_min
    while y <= y_max:
        point = [x, y]
        label = parzen_classifier(point, X_train, y_train, h=0.8)
        xx.append(x)
        yy.append(y)
        zz.append(label)
        y += step
    x += step

# --------------------------
# 5. Gráfico
# --------------------------
plt.figure(figsize=(8, 6))
plt.scatter([p[0] for p in class0], [p[1] for p in class0], color='blue', label='Clase 0', alpha=0.6)
plt.scatter([p[0] for p in class1], [p[1] for p in class1], color='red', label='Clase 1', alpha=0.6)
plt.scatter(xx, yy, c=zz, cmap='coolwarm', alpha=0.15, s=10)
plt.title('Clasificador Parzen Windows (h=0.8)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
