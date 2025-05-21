import random
import math

# Código proporcionado con evaluación de k-NN
def generate_data(n, mean, std_dev):
    return [[random.gauss(mean[0], std_dev[0]), random.gauss(mean[1], std_dev[1])] for _ in range(n)]

# Generar dos clases
class0 = generate_data(100, [0, 0], [1, 1])
class1 = generate_data(100, [3, 3], [1, 1])

X = class0 + class1
y = [0]*100 + [1]*100

# Clasificador k-NN
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

def knn_predict(x, X_train, y_train, k=5):
    distances = sorted([(euclidean_distance(x, X_train[i]), y_train[i]) for i in range(len(X_train))])
    top_k = [label for _, label in distances[:k]]
    return max(set(top_k), key=top_k.count)

# Métricas
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

# Validación cruzada manual (5-fold)
folds = 5
fold_size = len(X) // folds
results = []

for i in range(folds):
    start, end = i * fold_size, (i + 1) * fold_size
    X_test = X[start:end]
    y_test = y[start:end]
    X_train = X[:start] + X[end:]
    y_train = y[:start] + y[end:]

    y_pred = [knn_predict(x, X_train, y_train, k=5) for x in X_test]

    results.append({
        "Fold": i + 1,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1 Score": round(f1_score(y_test, y_pred), 3)
    })

# Mostrar resultados en tabla simple
headers = ["Fold", "Accuracy", "Precision", "Recall", "F1 Score"]
row_format = "{:<8}" * len(headers)
print(row_format.format(*headers))
print("-" * 8 * len(headers))
for row in results:
    print(row_format.format(row["Fold"], row["Accuracy"], row["Precision"], row["Recall"], row["F1 Score"]))
