#Seconde partie
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def one_hot(y, n_classes):
    m = y.shape[0]
    y_hot = np.zeros((m, n_classes))
    y_hot[np.arange(m), y] = 1
    return y_hot

def softmax(z):
    # Softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(y_hot, p_hat, W, l2_lambda):
    # Calcul du coût : Entropie Croisée + Pénalité L2
    m = y_hot.shape[0]

    # terme d'erreur
    cross_entropy = - (1/m) * np.sum(y_hot * np.log(p_hat + 1e-9))

    # terme de régularisation (L2)
    l2_penalty = (l2_lambda / (2 * m)) * np.sum(W * W)

    return cross_entropy + l2_penalty

def fit(X, y, lr, epochs, l2_lambda):
    m, n_features = X.shape
    n_classes = len(np.unique(y))

    # Initialisation des poids W et du biais b
    W = np.zeros((n_features, n_classes))
    b = np.zeros((1, n_classes))

    y_hot = one_hot(y, n_classes)
    loss_history = []

    print(f"entrainement (lr={lr}, lambda={l2_lambda})")

    for i in range(epochs):
        z = X.dot(W) + b
        p_hat = softmax(z)

        # Loss
        loss = compute_loss(y_hot, p_hat, W, l2_lambda)
        loss_history.append(loss)

        E = p_hat - y_hot

        # Gradient W (Base + L2)
        dW_base = (1/m) * X.T.dot(E)
        dW_reg = (l2_lambda / m) * W
        dW = dW_base + dW_reg

        # Gradient b
        db = (1/m) * np.sum(E, axis=0, keepdims=True)

        # Update
        W = W - lr * dW
        b = b - lr * db

    return W, b, loss_history

def predict(X, W, b):
    z = X.dot(W) + b
    p_hat = softmax(z)
    return np.argmax(p_hat, axis=1)


# dataset digits et régularisation
print("régularisation l2 sur digits : ")
print("\n")

# Chargement
digits = load_digits()
X_dig = digits.data
y_dig = digits.target

# Split et scale
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_dig, y_dig, test_size=0.2, random_state=42
)
scaler_d = StandardScaler()
X_train_d = scaler_d.fit_transform(X_train_d)
X_test_d = scaler_d.transform(X_test_d)

# régularisation faible (lambda = 0.1)
W_1, b_1, loss_1 = fit(X_train_d, y_train_d, lr=0.1, epochs=2000, l2_lambda=0.1)
acc_1 = accuracy_score(y_test_d, predict(X_test_d, W_1, b_1))
print(f"resultat (Lambda=0.1) : {acc_1 * 100:.2f} %")

# régularisation élevée (lambda = 50)
W_2, b_2, loss_2 = fit(X_train_d, y_train_d, lr=0.1, epochs=2000, l2_lambda=50)
acc_2 = accuracy_score(y_test_d, predict(X_test_d, W_2, b_2))
print(f"resultat (Lambda=50)  : {acc_2 * 100:.2f} %")


# Dataset MNIST
print("dataset MNIST :")
print("\n")

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

# Sous échantillonnage (10000)
np.random.seed(42)
indices = np.random.choice(len(mnist.data), 10000, replace=False)
X_mn = mnist.data[indices]
y_mn = mnist.target[indices].astype(int)


# Split / scale
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_mn, y_mn, test_size=0.2, random_state=42
)
scaler_m = StandardScaler()
X_train_m = scaler_m.fit_transform(X_train_m)
X_test_m = scaler_m.transform(X_test_m)

# Sklearn
model_sk = LogisticRegression(max_iter=1000, solver='lbfgs')
model_sk.fit(X_train_m, y_train_m)
acc_sk = accuracy_score(y_test_m, model_sk.predict(X_test_m))

# modèle from scratch
W_mn, b_mn, loss_mn = fit(X_train_m, y_train_m, lr=0.1, epochs=2000, l2_lambda=0.1)
acc_mn = accuracy_score(y_test_m, predict(X_test_m, W_mn, b_mn))

print(f"\nmoyenne 'from scratch': {acc_mn * 100:.2f} %")
print(f"moyenne 'Sklearn' : {acc_sk * 100:.2f} %")



#comparaison k-nn
print("comparaison avec k-NN")
print("\n")

# Entraînement KNN
print("Entraînement k-NN (k=3)")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_m, y_train_m)

# Prediction
acc_knn = accuracy_score(y_test_m, knn.predict(X_test_m))

print(f"\nRégression logistique : {acc_mn * 100:.2f} %")
print(f"k-NN (k=3) : {acc_knn * 100:.2f} %")


#visualisation

# Courbes de Loss (Digits L2 comparaison)
plt.figure(figsize=(10, 4))
plt.plot(loss_1, label='Lambda=0.1 (Normal)')
plt.plot(loss_2, label='Lambda=50 (Excessif)', linestyle='--')
plt.title("Impact de L2 sur la fonction de coût (Digits)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Poids MNIST
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("Visualisation des poids appris sur MNIST (28x28)")
for i, ax in enumerate(axes.flat):
    img = W_mn[:, i].reshape(28, 28)
    ax.imshow(img, cmap='viridis')
    ax.set_title(f"Chiffre {i}")
    ax.axis('off')
plt.show()

# Matrice de Confusion MNIST
cm = confusion_matrix(y_test_m, predict(X_test_m, W_mn, b_mn))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de Confusion (MNIST From Scratch)")
plt.show()