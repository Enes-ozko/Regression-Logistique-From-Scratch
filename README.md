# Régression Logistique "From Scratch" 

Projet réalisé en 2ème année d'école d'ingénieur à l'ENSISA.

**Objectif :** Comprendre les fondements mathématiques des réseaux de neurones en implémentant un classifieur d'images multi-classes sans utiliser de frameworks de haut niveau.

## Structure du Projet
* `data1.py` : Implémentation sur le dataset **Digits** (images 8x8 pixels). Inclut la boucle d'entraînement et la visualisation.
* `data2.py` : Implémentation sur le dataset **MNIST** (images 28x28 pixels). Ajoute la **Régularisation L2** et une comparaison avec k-NN.
* `docs/` : Contient le rapport complet du projet.

## Détails d'Implémentation
Le modèle est construit en **Python** pur avec **NumPy** pour la vectorisation.

### 1. Fondations Mathématiques
Nous avons codé manuellement les composants suivants :
* **Modèle :** Score linéaire $Z = XW + b$
* **Activation :** Fonction Softmax pour la distribution de probabilité.
    $$\hat{y} = \text{softmax}(Z)$$
* **Fonction de Coût :** Entropie Croisée (Cross-Entropy) avec régularisation L2 optionnelle.
    $$J(W) = -\frac{1}{m} \sum y \log(\hat{y}) + \frac{\lambda}{2m} \|W\|^2$$
* **Optimisation :** Descente de Gradient par Batch (Batch Gradient Descent).

### 2. Fonctionnalités Clés
* **Opérations vectorisées :** Utilisation du calcul matriciel pour remplacer les boucles Python et optimiser les performances.
* **Hyperparamètres :** Ajustement du taux d'apprentissage ($\alpha$) et du nombre d'époques.
* **Régularisation :** Ridge (L2) implémentée pour réduire le sur-apprentissage sur les données complexes (MNIST).

## Résultats
Nous avons comparé notre implémentation "maison" avec `LogisticRegression` et `KNeighborsClassifier` de Scikit-Learn.

| Dataset | Notre Modèle (NumPy) | Baseline Sklearn | k-NN (k=3) |
| :--- | :--- | :--- | :--- |
| **Digits (8x8)** | **97.50%** | 97.22% | N/A |
| **MNIST (28x28)** | **89.90%** | 88.15% | 90.80% |
