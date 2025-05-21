import random
import math
import matplotlib.pyplot as plt

# Generar datos sintéticos para dos clases normales 2D
def generate_class_data(n, mean, cov):
    data = []
    for _ in range(n):
        x = random.gauss(mean[0], cov[0])
        y = random.gauss(mean[1], cov[1])
        data.append([x, y])
    return data

# Datos de entrenamiento
class0 = generate_class_data(100, [0, 0], [1, 1])
class1 = generate_class_data(100, [3, 3], [1, 1])

X_train = class0 + class1
y_train = [0]*100 + [1]*100

# Función para estimar media y varianza
def estimate_parameters(data):
    n = len(data)
    mean_x = sum(d[0] for d in data) / n
    mean_y = sum(d[1] for d in data) / n
    var_x = sum((d[0] - mean_x)**2 for d in data) / n
    var_y = sum((d[1] - mean_y)**2 for d in data) / n
    return [mean_x, mean_y], [var_x, var_y]

# PDF normal univariada (independiente)
def gaussian_pdf(x, mean, var):
    return (1 / math.sqrt(2 * math.pi * var)) * math.exp(-((x - mean)**2) / (2 * var))

# Clasificador Bayesiano
def bayes_classifier(x, params0, params1, prior0, prior1):
    mean0, var0 = params0
    mean1, var1 = params1
    px0 = gaussian_pdf(x[0], mean0[0], var0[0]) * gaussian_pdf(x[1], mean0[1], var0[1])
    px1 = gaussian_pdf(x[0], mean1[0], var1[0]) * gaussian_pdf(x[1], mean1[1], var1[1])
    return 0 if px0 * prior0 > px1 * prior1 else 1

# Entrenamiento
params0 = estimate_parameters(class0)
params1 = estimate_parameters(class1)
prior0 = len(class0) / (len(class0) + len(class1))
prior1 = 1 - prior0

# Visualización con frontera de decisión
x_min, x_max = -4, 7
y_min, y_max = -4, 7
step = 0.2
xx, yy = [], []
zz = []

x = x_min
while x <= x_max:
    y = y_min
    while y <= y_max:
        xx.append(x)
        yy.append(y)
        zz.append(bayes_classifier([x, y], params0, params1, prior0, prior1))
        y += step
    x += step

# Gráfico
plt.figure(figsize=(8, 6))
plt.scatter([p[0] for p in class0], [p[1] for p in class0], color='blue', label='Clase 0', alpha=0.6)
plt.scatter([p[0] for p in class1], [p[1] for p in class1], color='red', label='Clase 1', alpha=0.6)
plt.scatter(xx, yy, c=zz, cmap='coolwarm', alpha=0.15, s=10)
plt.title('Clasificador Bayesiano (Simplificado)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
