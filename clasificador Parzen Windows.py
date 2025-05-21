import random
import math

# -------------------------------
# 1. Generación de datos sintéticos
# -------------------------------
def generate_data(n, mean, std_dev):
    return [[random.gauss(mean[0], std_dev[0]), random.gauss(mean[1], std_dev[1])] for _ in range(n)]

class0 = generate_data(100, [0, 0], [1, 1])
class1 = generate_data(100, [3, 3], [1, 1])

X = class0 + class1
y = [0]*100 + [1]*100

# -------------------------------
# 2. Kernel Gaussiano para Parzen
# -------------------------------
def gaussian_kernel(x, xi, h):
    diff_x = x[0] - xi[0]
    diff_y = x[1] - xi[1]
    norm_sq = diff_x**2 + diff_y**2
    return math.exp(-norm_sq / (2 * h**2))

# -------------------------------
# 3. Clasificador Parzen Windows
# -------------------------------
def parzen_predict(x, X_train, y_train, h=0.8):
    sum_0 = 0
    sum_1 = 0
    for i in range(len(X_train)):
        weight = gaussian_kernel(x, X_train[i], h)
        if y_train[i] == 0:
            sum_0 += weight
        else:
            sum_1 += weight
    return 0 if sum_0 > sum_1 else 1

# -------------------------------
# 4. Métricas de evaluación
# -------------------------------
def precision_score(y_true, y_pred):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    return tp / (tp + fp) if tp + fp > 0 else 0.0

def recall_score(y_true, y_pred):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    return tp / (tp + fn) if tp + fn > 0 else 0.0

def accuracy_score(y_true, y_pred):
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return correct / len(y_true)

def f1_score(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

# -------------------------------
# 5. Validación cruzada manual (5-fold)
# -------------------------------
folds = 5
fold_size = len(X) // folds
results_parzen = []

for i in range(folds):
    start, end = i * fold_size, (i + 1) * fold_size
    X_test = X[start:end]
    y_test = y[start:end]
    X_train = X[:start] + X[end:]
    y_train = y[:start] + y[end:]

    y_pred = [parzen_predict(x, X_train, y_train, h=0.8) for x in X_test]

    results_parzen.append({
        "Fold": i + 1,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1 Score": round(f1_score(y_test, y_pred), 3)
    })

# -------------------------------
# 6. Mostrar resultados en tabla
# -------------------------------
headers = ["Fold", "Accuracy", "Precision", "Recall", "F1 Score"]
row_format = "{:<8}" * len(headers)
print(row_format.format(*headers))
print("-" * 8 * len(headers))
for row in results_parzen:
    print(row_format.format(row["Fold"], row["Accuracy"], row["Precision"], row["Recall"], row["F1 Score"]))
