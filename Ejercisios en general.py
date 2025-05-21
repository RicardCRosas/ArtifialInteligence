import numpy as np
import matplotlib.pyplot as plt


# ========================================================
# 1. DISTRIBUCIONES GAUSSIANAS: Univariada y Multivariada
# ========================================================
def ejemplo_gaussiana():
    # --- Univariada ---
    mu = 0
    sigma = 1
    x = np.linspace(-5, 5, 400)
    pdf = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    plt.figure()
    plt.plot(x, pdf, 'b-', lw=2)
    plt.title("Gaussiana Univariada (mu=0, sigma=1)")
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.grid(True)
    plt.show()

    # --- Multivariada (2D) ---
    mu_multi = np.array([0, 0])
    sigma_multi = np.array([[1, 0.5], [0.5, 1]])
    sigma_inv = np.linalg.inv(sigma_multi)
    det_sigma = np.linalg.det(sigma_multi)
    x_vals = np.linspace(-3, 3, 100)
    y_vals = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    pos = np.dstack((X, Y))
    d = 2
    factor = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_sigma))
    diff = pos - mu_multi
    exponent = -0.5 * np.einsum('...i,ij,...j->...', diff, sigma_inv, diff)
    pdf_multi = factor * np.exp(exponent)

    plt.figure()
    cp = plt.contourf(X, Y, pdf_multi, cmap="viridis")
    plt.title("Gaussiana Multivariada (2D)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(cp)
    plt.show()


# ========================================================
# 2. ESTIMACIÓN PARAMÉTRICA
# 2.1 Normal (ya visto)
# ========================================================
def ejemplo_estimacion_parametrica_normal():
    np.random.seed(0)
    n = 500
    data = np.random.normal(2, 1.5, size=n)
    sample_mean = np.sum(data) / n
    sample_var = np.sum((data - sample_mean) ** 2) / n
    print("Normal: Media =", sample_mean, "Varianza =", sample_var)

    x = np.linspace(min(data) - 1, max(data) + 1, 400)
    pdf_estimada = 1 / (np.sqrt(2 * np.pi * sample_var)) * np.exp(-0.5 * ((x - sample_mean) ** 2) / sample_var)

    plt.figure()
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label="Histograma")
    plt.plot(x, pdf_estimada, 'r--', lw=2, label="PDF estimada")
    plt.title("Estimación Paramétrica: Distribución Normal")
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.legend()
    plt.show()


# ========================================================
# 2.2 Estimación Paramétrica: Distribución Logística (caso especial)
# --------------------------------------------------------
# La pdf logística:
#   f(x; μ, s) = exp(-(x-μ)/s) / [ s * (1+exp(-(x-μ)/s))^2 ]
# La log-verosimilitud para un conjunto {x_i} es:
#   L(μ,s) = -n*log(s) - Σ((x_i-μ)/s) - 2 Σ(log(1+exp(-(x_i-μ)/s)))
# Se derivan las ecuaciones de gradiente para μ y s.
# ========================================================
def ejemplo_estimacion_parametrica_logistic():
    np.random.seed(1)
    # Generamos datos a partir de una distribución logística con parámetros reales:
    mu_true = 1.0
    s_true = 0.8
    n = 500
    # Generar datos a partir de la fórmula inversa: x = μ + s * log(u/(1-u))
    u = np.random.rand(n)
    data = mu_true + s_true * np.log(u / (1 - u))

    # Inicializamos estimadores
    mu_est = 0.0
    s_est = 1.0
    lr = 0.01
    n_epochs = 2000

    for epoch in range(n_epochs):
        z = (data - mu_est) / s_est
        # Gradiente respecto a mu: dL/dμ = (1/s_est) * Σ[1 - 2/(1+exp(z))]
        grad_mu = np.sum((1 - 2 / (1 + np.exp(z)))) / s_est
        # Gradiente respecto a s:
        # dL/ds = -n/s_est + Σ[(data - mu_est)/s_est^2] - 2 * Σ[((data - mu_est)/s_est^2) * exp(-z)/(1+exp(-z))]
        grad_s = -n / s_est + np.sum((data - mu_est)) / s_est ** 2 - 2 * np.sum(
            ((data - mu_est) / s_est ** 2) * (np.exp(-z) / (1 + np.exp(-z))))
        mu_est -= lr * grad_mu
        s_est -= lr * grad_s
        # Opcional: podríamos imprimir cada cierto número de iteraciones

    print("Logística: Parámetros estimados -> mu =", mu_est, "s =", s_est)

    # Graficamos la densidad real y la estimada:
    x = np.linspace(np.min(data) - 1, np.max(data) + 1, 400)
    # Densidad real:
    pdf_real = np.exp(-(x - mu_true) / s_true) / (s_true * (1 + np.exp(-(x - mu_true) / s_true)) ** 2)
    # Densidad estimada:
    pdf_est = np.exp(-(x - mu_est) / s_est) / (s_est * (1 + np.exp(-(x - mu_est) / s_est)) ** 2)

    plt.figure()
    plt.hist(data, bins=30, density=True, alpha=0.6, color='gray', label="Histograma")
    plt.plot(x, pdf_real, 'g--', lw=2, label="PDF real (logística)")
    plt.plot(x, pdf_est, 'r-', lw=2, label="PDF estimada (logística)")
    plt.title("Estimación Paramétrica: Distribución Logística")
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.legend()
    plt.show()


# ========================================================
# 3. ESTIMACIÓN NO PARAMÉTRICA
# 3.1 Parzen Window en 1D (ya visto)
# ========================================================
def gaussian_kernel(u):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * u ** 2)


def ejemplo_parzen():
    np.random.seed(1)
    n = 500
    data = np.random.normal(0, 1, size=n)
    h = 0.3
    x_grid = np.linspace(-5, 5, 400)
    densidad_est = np.zeros_like(x_grid)
    for i, x_val in enumerate(x_grid):
        u = (x_val - data) / h
        densidad_est[i] = np.sum(gaussian_kernel(u))
    densidad_est /= (n * h)
    densidad_real = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x_grid ** 2)
    plt.figure()
    plt.plot(x_grid, densidad_est, label="Parzen Window", lw=2)
    plt.plot(x_grid, densidad_real, 'r--', label="Densidad real", lw=2)
    plt.title("Estimación No Paramétrica: Parzen Window (1D)")
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.legend()
    plt.show()


# ========================================================
# 3.2 k-Nearest Neighbor para estimación de densidad en 1D (ya visto)
# ========================================================
def ejemplo_knn_density_1d():
    np.random.seed(2)
    n = 500
    data = np.random.normal(0, 1, size=n)
    k = 10
    x_grid = np.linspace(-5, 5, 200)
    densidad_knn = []
    for x_val in x_grid:
        dist = np.abs(data - x_val)
        r_k = np.sort(dist)[k - 1]
        densidad_knn.append(k / (n * (2 * r_k)))
    densidad_knn = np.array(densidad_knn)
    densidad_real = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x_grid ** 2)
    plt.figure()
    plt.plot(x_grid, densidad_knn, label="k-NN (k=10)", lw=2)
    plt.plot(x_grid, densidad_real, 'r--', label="Densidad real", lw=2)
    plt.title("k-Nearest Neighbor para densidad (1D)")
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.legend()
    plt.show()


# ========================================================
# 3.3 k-Nearest Neighbor para densidad en 2D (caso especial)
# --------------------------------------------------------
def ejemplo_knn_density_2d():
    np.random.seed(3)
    n = 300
    # Generamos datos 2D de una normal estándar
    data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n)
    k = 10
    # Creamos una grilla en 2D
    x_grid = np.linspace(-4, 4, 100)
    y_grid = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    density = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            # Calculamos las distancias Euclidianas
            dists = np.linalg.norm(data - point, axis=1)
            r_k = np.sort(dists)[k - 1]
            # Área de un círculo de radio r_k en 2D: π*r_k^2
            density[i, j] = k / (n * np.pi * r_k ** 2)
    # Densidad real: para una normal 2D estándar
    norm_const = 1 / (2 * np.pi)
    density_real = norm_const * np.exp(-0.5 * (X ** 2 + Y ** 2))

    plt.figure()
    cp = plt.contourf(X, Y, density, cmap="plasma", alpha=0.7)
    plt.colorbar(cp)
    plt.title("Estimación k-NN (2D) de la densidad")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plt.figure()
    cp2 = plt.contourf(X, Y, density_real, cmap="plasma", alpha=0.7)
    plt.colorbar(cp2)
    plt.title("Densidad real 2D (Normal estándar)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# ========================================================
# 4. CLASIFICADOR SIMPLE: PERCEPTRÓN (ya visto)
# ========================================================
def ejemplo_perceptron():
    np.random.seed(3)
    n1 = 50
    X1 = np.random.randn(n1, 2) * 0.5 + np.array([1, 1])
    n2 = 50
    X2 = np.random.randn(n2, 2) * 0.5 + np.array([3, 3])
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(n1), -np.ones(n2)))
    X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
    w = np.random.randn(3)
    lr = 0.1
    max_iter = 1000
    errores = []
    for it in range(max_iter):
        error_total = 0
        for i in range(X_bias.shape[0]):
            if y[i] * np.dot(w, X_bias[i]) <= 0:
                w += lr * y[i] * X_bias[i]
                error_total += 1
        errores.append(error_total)
        if error_total == 0:
            break
    print("Perceptrón convergió en {} iteraciones.".format(it + 1))
    plt.figure()
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', color='blue', label="Clase 1")
    plt.scatter(X2[:, 0], X2[:, 1], marker='s', color='red', label="Clase -1")
    x_vals = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, 100)
    if np.abs(w[1]) > 1e-6:
        y_vals = -(w[0] * x_vals + w[2]) / w[1]
        plt.plot(x_vals, y_vals, 'k--', lw=2, label="Frontera")
    plt.title("Clasificador Perceptrón (Lineal)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(errores, marker='o')
    plt.title("Errores por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Errores")
    plt.grid(True)
    plt.show()


# ========================================================
# 5. CLASIFICADOR BAYESIANO
# 5.1 Con covarianza igual (ya mostrado)
# 5.2 Con covarianza DIAGONAL (caso especial)
# --------------------------------------------------------
def ejemplo_bayes_diagonal_cov():
    np.random.seed(4)
    n_samples = 200
    # Clase 0: generamos con covarianza diagonal (varianzas diferentes)
    mu0 = np.array([0, 0])
    sigma0 = np.diag([1.0, 2.0])
    X0 = np.random.multivariate_normal(mu0, sigma0, n_samples)
    # Clase 1:
    mu1 = np.array([3, 3])
    sigma1 = np.diag([1.5, 0.5])
    X1 = np.random.multivariate_normal(mu1, sigma1, n_samples)

    # Priori iguales
    P0 = 0.5;
    P1 = 0.5

    # Función discriminante para cada clase usando pdf producto de univariadas:
    def g(x, mu, sigma_diag, P):
        # sigma_diag: vector de desviaciones (raíz de varianzas)
        # g(x) = - Σ((x_i - μ_i)^2/(2σ_i^2)) - Σ(log σ_i) + log(P)
        return -np.sum((x - mu) ** 2 / (2 * (sigma_diag ** 2))) - np.sum(np.log(sigma_diag)) + np.log(P)

    # Evaluamos en una grilla
    x_min, x_max = -4, 8
    y_min, y_max = -4, 8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    g_diff = np.array([g(p, mu0, np.sqrt(np.diag(sigma0)), P0) - g(p, mu1, np.sqrt(np.diag(sigma1)), P1) for p in grid])
    g_diff = g_diff.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, g_diff, levels=[-np.inf, 0, np.inf], colors=['lightblue', 'lightcoral'], alpha=0.5)
    plt.contour(xx, yy, g_diff, levels=[0], colors='k', linewidths=2)
    plt.scatter(X0[:, 0], X0[:, 1], marker='o', color='blue', label="Clase 0")
    plt.scatter(X1[:, 0], X1[:, 1], marker='s', color='red', label="Clase 1")
    plt.title("Clasificador Bayesiano: Covarianza Diagonal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# ========================================================
# 6. PARZEN NEURAL NETWORK: Aproximación de la densidad exponencial
# --------------------------------------------------------
# En este ejemplo entrenamos una red neuronal de dos capas (1 capa oculta)
# desde cero para aproximar la densidad de una distribución exponencial:
#   f(x)= exp(-x)  para x>=0, 0 para x<0.
# La red tiene:
#   - Entrada 1D.
#   - Capa oculta: 10 neuronas con función sigmoide.
#   - Capa de salida: activación lineal seguida de ReLU para asegurar salida no negativa.
# Se entrena usando descenso de gradiente (MSE).
# ========================================================
def ejemplo_pnn_exponencial():
    np.random.seed(5)
    # Generamos datos en x en el intervalo [0, 5]
    n_train = 200
    x_train = np.linspace(0, 5, n_train).reshape(-1, 1)
    # Densidad verdadera de una exponencial (con lambda=1)
    y_true = np.exp(-x_train)

    # Arquitectura de la red:
    input_dim = 1
    hidden_size = 10
    output_dim = 1

    # Inicialización de pesos y sesgos
    W1 = np.random.randn(input_dim, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_dim) * 0.1
    b2 = np.zeros((1, output_dim))

    # Funciones de activación y sus derivadas
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(z):
        s = sigmoid(z)
        return s * (1 - s)

    def relu(z):
        return np.maximum(0, z)

    def d_relu(z):
        return (z > 0).astype(float)

    # Parámetros de entrenamiento
    epochs = 3000
    lr = 0.01
    losses = []

    for epoch in range(epochs):
        # Forward pass
        z1 = x_train.dot(W1) + b1  # Capa oculta lineal
        a1 = sigmoid(z1)  # Activación sigmoide
        z2 = a1.dot(W2) + b2  # Capa de salida lineal
        a2 = relu(z2)  # Activación ReLU en la salida

        # Cálculo del error (MSE)
        error = a2 - y_true
        loss = np.mean(error ** 2)
        losses.append(loss)

        # Backpropagation
        # Capa de salida: dL/dz2 = error * d_relu(z2)
        delta2 = error * d_relu(z2)
        dW2 = a1.T.dot(delta2) / n_train
        db2 = np.sum(delta2, axis=0, keepdims=True) / n_train

        # Capa oculta:
        delta1 = delta2.dot(W2.T) * d_sigmoid(z1)
        dW1 = x_train.T.dot(delta1) / n_train
        db1 = np.sum(delta1, axis=0, keepdims=True) / n_train

        # Actualización de pesos
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.5f}")

    # Visualización de la función estimada vs verdadera
    x_test = np.linspace(0, 5, 400).reshape(-1, 1)
    z1_test = x_test.dot(W1) + b1
    a1_test = sigmoid(z1_test)
    z2_test = a1_test.dot(W2) + b2
    a2_test = relu(z2_test)
    plt.figure()
    plt.plot(x_test, np.exp(-x_test), 'g--', lw=2, label="Exponencial real")
    plt.plot(x_test, a2_test, 'r-', lw=2, label="Función aprendida")
    plt.scatter(x_train, y_true, color='blue', s=10, label="Datos entrenamiento")
    plt.title("Aproximación de la densidad exponencial con NN")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(losses)
    plt.title("Curva de pérdida (MSE) durante el entrenamiento")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()


from mpl_toolkits.mplot3d import Axes3D  # mplot3d viene con matplotlib


# Ejemplo 3D: Scatter plot de datos 3D de una Gaussiana
def ejemplo_scatter_3d():
    np.random.seed(10)
    n = 300
    # Generamos datos 3D a partir de una distribución normal (media 0, covarianza identidad)
    data = np.random.multivariate_normal([0, 0, 0], np.eye(3), n)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', marker='o', alpha=0.6)
    ax.set_title("Scatter 3D: Datos de Gaussiana")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


# Ejemplo 3D: Visualización de cortes (slices) de la densidad de una Gaussiana 3D
def ejemplo_densidad_3d():
    # Configuramos el grid para x e y y definimos varios cortes en z
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    z_slices = [-2, 0, 2]  # Cortes en z
    # Parámetros de la Gaussiana 3D: media (0,0,0) y covarianza identidad
    mu = np.array([0, 0, 0])
    d = 3
    # Factor constante: 1/((2pi)^(d/2)) para sigma = I
    factor = 1 / ((2 * np.pi) ** (d / 2))

    plt.figure(figsize=(15, 4))
    for i, z_val in enumerate(z_slices):
        X, Y = np.meshgrid(x, y)
        # Para cada corte, z es constante
        Z = z_val * np.ones_like(X)
        # Dado que sigma = I, la función densidad es:
        # f(x,y,z) = factor * exp(-0.5*(x^2+y^2+z^2))
        exponent = -0.5 * (X ** 2 + Y ** 2 + Z ** 2)
        densidad = factor * np.exp(exponent)
        plt.subplot(1, len(z_slices), i + 1)
        cp = plt.contourf(X, Y, densidad, cmap="viridis")
        plt.title(f"Corte en z = {z_val}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(cp)
    plt.suptitle("Densidad de una Gaussiana 3D en cortes fijos de z")
    plt.show()


# ========================================================
# BLOQUE PRINCIPAL: EJECUCIÓN DE EJEMPLOS
# ========================================================
if __name__ == "__main__":
    print("1. Ejemplo: Distribuciones Gaussianas")
    ejemplo_gaussiana()

    print("2. Ejemplo: Estimación Paramétrica - Normal")
    ejemplo_estimacion_parametrica_normal()

    print("3. Ejemplo: Estimación Paramétrica - Logística")
    ejemplo_estimacion_parametrica_logistic()

    print("4. Ejemplo: Estimación No Paramétrica: Parzen Window (1D)")
    ejemplo_parzen()

    print("5. Ejemplo: Estimación No Paramétrica: k-NN (1D)")
    ejemplo_knn_density_1d()

    print("6. Ejemplo: Estimación No Paramétrica: k-NN (2D)")
    ejemplo_knn_density_2d()

    print("7. Ejemplo: Clasificador Simple: Perceptrón")
    ejemplo_perceptron()

    print("8. Ejemplo: Clasificador Bayesiano: Covarianza Diagonal")
    ejemplo_bayes_diagonal_cov()

    print("9. Ejemplo: Aproximación de densidad exponencial con NN (PNN simple)")
    ejemplo_pnn_exponencial()

    print("Ejemplo 3D: Scatter Plot")
    ejemplo_scatter_3d()

    print("Ejemplo 3D: Cortes de la densidad (slices)")
    ejemplo_densidad_3d()
