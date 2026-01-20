#Première partie 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Charger les données
digits = load_digits()
X = digits.data
y = digits.target

print(f"Forme de X: {X.shape}")
print(f"Forme de y: {y.shape}")

# 2. Séparation Train / Test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Modèle sklearn

print("\n entrainement sklearn")
model_sklearn = LogisticRegression(max_iter=1000)
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"modèle sklearn : {acc_sklearn * 100:.2f} %")


# Modèle from Scratch

def one_hot(y, n_classes):
    m = y.shape[0]
    y_hot = np.zeros((m, n_classes))
    y_hot[np.arange(m), y] = 1
    return y_hot

def softmax(z):
    # softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(y_hot, p_hat):
    # entropie croisée
    m = y_hot.shape[0]
    # Erreur d'entropie croisée
    loss = - (1/m) * np.sum(y_hot * np.log(p_hat + 1e-9))

    return loss

# Descente de Gradient
def fit(X, y, lr, epochs):

    m, n_features = X.shape
    n_classes = len(np.unique(y))

    # Initialisation des paramètres théta (W et b)
    W = np.zeros((n_features, n_classes))
    b = np.zeros((1, n_classes))

    y_hot = one_hot(y, n_classes)

    loss_history = []

    # Entrainement
    for i in range(epochs):

        z = X.dot(W) + b

        p_hat = softmax(z)

        # Erreur
        loss = compute_loss(y_hot, p_hat)
        loss_history.append(loss)

        E = p_hat - y_hot

        # Gradient pour W (dW)
        dW = (1/m) * X.T.dot(E)

        # Gradient pour b (db)
        db = (1/m) * np.sum(E, axis=0, keepdims=True)

        W = W - lr * dW
        b = b - lr * db

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss}")

    print("Entraînement 'from scratch' terminé")
    return W, b, loss_history

def predict(X, W, b):
    z = X.dot(W) + b
    p_hat = softmax(z)
    return np.argmax(p_hat, axis=1)


print("\nEntraînement 'From Scratch' ")

# Hyperparamètres
learning_rate = 0.1
nb_epochs = 2000

# Lancer l'entraînement
W_final, b_final, loss_hist = fit(X_train, y_train, lr=learning_rate, epochs=nb_epochs)

# Évaluation
y_pred_scratch = predict(X_test, W_final, b_final)
acc_scratch = accuracy_score(y_test, y_pred_scratch)

print("\nrésultats :")
print(f"moyenne 'From Scratch' : {acc_scratch * 100:.2f} %")
print(f"moyenne 'Sklearn' (cible): {acc_sklearn * 100:.2f} %")

# courbe du loss
plt.figure(figsize=(8, 5))
plt.plot(loss_hist)
plt.title("Évolution du Coût (Loss) pendant l'entraînement")
plt.xlabel("Epochs")
plt.ylabel("Entropie Croisée")
plt.grid(True)
plt.show()

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_scratch)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de Confusion - Modèle 'From Scratch'")
plt.ylabel('Vrai Chiffre')
plt.xlabel('Chiffre Prédit')
plt.show()

# Poids
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("Visualisation des Poids 'W' ")

for i, ax in enumerate(axes.flat):
    image_poids = W_final[:, i].reshape(8, 8)

    ax.imshow(image_poids, cmap='viridis')
    ax.set_title(f"Poids pour: {i}")
    ax.axis('off')

plt.show()
