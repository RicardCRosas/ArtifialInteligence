import random
import math

# -------------------------------
# Simulación OCR en 1D con 3 clases
# -------------------------------
def generate_class_data(n, mean, std_dev):
    return [random.gauss(mean, std_dev) for _ in range(n)]

# Generar datos
class_1 = generate_class_data(50, 2, 1.2)
class_2 = generate_class_data(50, 5, 2.2)
class_3 = generate_class_data(50, 8, 3.2)

# Etiquetas
X = class_1 + class_2 + class_3
y = [1]*50 + [2]*50 + [3]*50

# -------------------------------
# Estimaciones ML de media y varianza
# -------------------------------
def ml_mean(data):
    return sum(data) / len(data)

def ml_variance(data, mean):
    return sum((x - mean)**2 for x in data) / len(data)

mean_1 = ml_mean(class_1)
mean_2 = ml_mean(class_2)
mean_3 = ml_mean(class_3)

var_1 = ml_variance(class_1, mean_1)
var_2 = ml_variance(class_2, mean_2)
var_3 = ml_variance(class_3, mean_3)

# -------------------------------
# Funciones discriminantes de Bayes
# -------------------------------
def gi(x, mean, var, prior):
    return -((x - mean) ** 2) / (2 * var) + math.log(prior)

# Clasificador Bayes
def bayes_classifier(x):
    prior = 1/3
    g1 = gi(x, mean_1, var_1, prior)
    g2 = gi(x, mean_2, var_2, prior)
    g3 = gi(x, mean_3, var_3, prior)
    g_values = {1: g1, 2: g2, 3: g3}
    return max(g_values, key=g_values.get)

# -------------------------------
# Probar el clasificador con conjunto test
# -------------------------------
test_set = generate_class_data(10, 2, 1.2) + generate_class_data(10, 5, 2.2) + generate_class_data(10, 8, 3.2)
true_labels = [1]*10 + [2]*10 + [3]*10
pred_labels = [bayes_classifier(x) for x in test_set]

# Precisión
correct = sum(t == p for t, p in zip(true_labels, pred_labels))
accuracy = correct / len(true_labels)

# -------------------------------
# Estimación de parámetros de otras distribuciones
# -------------------------------

# Ejemplo: Distribución logística
def logistic_ml_params(data):
    mean = ml_mean(data)
    mad = sum(abs(x - mean) for x in data) / len(data)
    scale = mad * math.sqrt(3) / math.pi
    return mean, scale

logistic_mean1, logistic_scale1 = logistic_ml_params(class_1)

# Ejemplo: Fisher-Tippett (Gumbel) aproximado
def fisher_tippett_ml_params(data):
    mean = ml_mean(data)
    std_dev = math.sqrt(ml_variance(data, mean))
    gamma = 0.5772
    beta = std_dev * math.sqrt(6) / math.pi
    mu = mean - gamma * beta
    return mu, beta

fisher_mu1, fisher_beta1 = fisher_tippett_ml_params(class_1)

# Resultados resumidos
output = {
    "ML estimates (Gaussian)": {
        "Clase 1": {"Media": round(mean_1, 3), "Varianza": round(var_1, 3)},
        "Clase 2": {"Media": round(mean_2, 3), "Varianza": round(var_2, 3)},
        "Clase 3": {"Media": round(mean_3, 3), "Varianza": round(var_3, 3)}
    },
    "Accuracy on test set": round(accuracy, 3),
    "Logistic parameters (Clase 1)": {"Media": round(logistic_mean1, 3), "Scale": round(logistic_scale1, 3)},
    "Fisher-Tippett parameters (Clase 1)": {"Mu": round(fisher_mu1, 3), "Beta": round(fisher_beta1, 3)}
}

print(output)