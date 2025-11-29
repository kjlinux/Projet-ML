# Analyse et Modélisation des Données de Facturation d'Eau

## Description du Projet

Ce projet propose une analyse complète et une modélisation machine learning des données de facturation d'eau. Il comprend trois tâches principales:

1. **Classification des Catégories de Clients** - Prédire la catégorie de client (PRIVE, PUBLIC, etc.)
2. **Prédiction des Résiliations** - Identifier les clients à risque de résiliation
3. **Prédiction des Montants** - Estimer les montants TTC de facturation

## Structure des Données

Les données sont au format CSV avec les colonnes suivantes:
- **Identifiants:** DR, CEN, POLICE, O, P, ENR
- **Temporelles:** MM, AAAA, DATE_FACT, DATE_ABON, DATE_RESIL, DATE_REGLT, DATE_REGLT_ENC
- **Consommation:** DIAM, CUBCONS, CUBFAC, FORFAIT
- **Types de consommation:** SOCIAL, DOMEST, NORMAL, INDUST, ADMINI
- **Montants:** MONT_SOD, MONT_TVA, MONT_FDE, MONT_FNE, MONT_ASS_TTC, MONT_FRAIS_CPT, MONT_TTC
- **Autres:** TOURNEE, AAENC, MMENC, RESILIE, CATEGORIE, NOUVEAU

## Organisation du Projet

```
.
├── datasets/               # Dossier contenant les fichiers de données (.txt)
│   ├── DR2.txt
│   ├── DR6.txt
│   ├── DR7.txt
│   ├── DR9.txt
│   ├── DR16.txt
│   └── DR21.txt
├── analyse_facturation_eau.ipynb   # Notebook principal
└── README.md              # Ce fichier
```

## Prérequis

### Installation des Bibliothèques

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn --break-system-packages
```

### Bibliothèques Requises

- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- imbalanced-learn >= 0.9.0

## Utilisation du Notebook

### 1. Préparation

Assurez-vous que:
- Les fichiers de données sont dans le dossier `datasets/`
- Toutes les bibliothèques sont installées
- Vous êtes dans le bon répertoire

### 2. Exécution

Ouvrez le notebook avec Jupyter:

```bash
jupyter notebook analyse_facturation_eau.ipynb
```

Ou avec JupyterLab:

```bash
jupyter lab analyse_facturation_eau.ipynb
```

### 3. Structure du Notebook

Le notebook est organisé en 12 sections:

1. **Import des Bibliothèques** - Chargement de tous les modules nécessaires
2. **Chargement des Données** - Lecture et concaténation de tous les fichiers
3. **Exploration des Données** - Analyse descriptive et statistiques
4. **Prétraitement des Données** - Nettoyage et transformation
5. **Analyse Exploratoire Visuelle** - Visualisations et insights
6. **Préparation pour la Modélisation** - Sélection des features
7. **Modélisation - Classification Catégorie** - Prédiction des catégories de clients
8. **Modélisation - Classification Résiliation** - Prédiction des résiliations
9. **Modélisation - Régression Montant** - Prédiction des montants TTC
10. **Optimisation** - Tuning des hyperparamètres
11. **Sauvegarde des Résultats** - Export des modèles et résultats
12. **Conclusions** - Synthèse et recommandations

## Techniques Anti-Surapprentissage Utilisées

Le projet implémente plusieurs techniques pour éviter le surapprentissage:

### 1. Validation Croisée
- **Stratified K-Fold (k=5)** pour la classification
- Validation sur plusieurs splits pour une évaluation robuste

### 2. Régularisation
- **Ridge (L2)** - Pénalise les coefficients élevés
- **Lasso (L1)** - Sélection de features automatique
- **ElasticNet** - Combinaison L1 et L2

### 3. Contrôle de la Complexité
Pour les modèles à base d'arbres:
- `max_depth` - Limite la profondeur des arbres
- `min_samples_split` - Minimum d'échantillons pour diviser un noeud
- `min_samples_leaf` - Minimum d'échantillons dans une feuille

### 4. Méthodes d'Ensemble
- **Random Forest** - Bagging avec bootstrap
- **AdaBoost** - Boosting adaptatif
- **Gradient Boosting** - Boosting par gradient avec subsample
- **Bagging** - Agrégation de bootstrap

### 5. Traitement du Déséquilibre
- **SMOTE** - Synthetic Minority Over-sampling Technique
- Appliqué pour la prédiction des résiliations

### 6. Normalisation
- **StandardScaler** - Normalisation des features
- Évite les biais dus aux échelles différentes

## Modèles Utilisés

### Classification
1. Random Forest Classifier
2. AdaBoost Classifier
3. Gradient Boosting Classifier
4. Bagging Classifier
5. Logistic Regression avec régularisation L2

### Régression
1. Random Forest Regressor
2. AdaBoost Regressor
3. Gradient Boosting Regressor
4. Ridge Regression
5. Lasso Regression
6. ElasticNet Regression

## Résultats Générés

Le notebook génère automatiquement:

1. **Visualisations:**
   - Distribution des catégories et résiliations
   - Boxplots de consommation par catégorie
   - Matrices de corrélation
   - Matrices de confusion
   - Graphiques de comparaison des modèles
   - Feature importance

2. **Métriques:**
   - Scores de validation croisée
   - Accuracy, Precision, Recall, F1-Score
   - R², RMSE, MAE pour la régression
   - Niveau de surapprentissage (Train-Test gap)

3. **Fichiers Exportés:**
   - `resultats_classification_categorie.csv`
   - `resultats_classification_resiliation.csv`
   - `resultats_regression_montant.csv`
   - `meilleur_modele_categorie.pkl`
   - `meilleur_modele_resiliation.pkl`
   - `meilleur_modele_regression.pkl`
   - Scalers correspondants

## Interprétation des Résultats

### Score de Validation Croisée
- **Bon:** > 0.80 pour la classification, > 0.70 pour R² en régression
- **Acceptable:** 0.70-0.80 pour la classification, 0.50-0.70 pour R²
- **Faible:** < 0.70 pour la classification, < 0.50 pour R²

### Surapprentissage
- **Bon:** Différence Train-Test < 0.05
- **Acceptable:** Différence Train-Test < 0.10
- **Problématique:** Différence Train-Test > 0.10

### Actions si Surapprentissage Détecté
1. Augmenter la régularisation (alpha plus élevé)
2. Réduire la complexité (max_depth plus faible)
3. Augmenter min_samples_split et min_samples_leaf
4. Utiliser plus de données d'entraînement
5. Réduire le nombre de features (feature selection)

### Actions si Sous-apprentissage Détecté
1. Augmenter la complexité du modèle
2. Créer de nouvelles features
3. Réduire la régularisation
4. Essayer des modèles plus complexes
5. Vérifier la qualité des données

## Recommandations pour l'Amélioration

### 1. Feature Engineering Avancé
- Créer des ratios et interactions entre features
- Extraire des features temporelles (saison, jour de la semaine)
- Agréger par client (consommation moyenne, variance)

### 2. Techniques Avancées
Si les résultats ne sont pas satisfaisants, essayer:
- **XGBoost** - Gradient boosting optimisé
- **LightGBM** - Boosting rapide et efficace
- **CatBoost** - Spécialisé pour les variables catégorielles
- **Neural Networks** - Pour capturer des relations complexes

### 3. Validation
- Utiliser une validation temporelle (train sur passé, test sur futur)
- Implémenter un monitoring en production
- Tester sur différentes périodes

## Troubleshooting

### Problème: "No module named 'imblearn'"
```bash
pip install imbalanced-learn --break-system-packages
```

### Problème: Mémoire insuffisante
- Réduire le nombre de fichiers chargés
- Utiliser `chunksize` pour le chargement
- Réduire `n_estimators` dans les modèles

### Problème: Temps d'exécution trop long
- Réduire `n_estimators`
- Utiliser `n_jobs=-1` pour paralléliser
- Réduire le nombre de folds en validation croisée
- Limiter la grille de recherche dans GridSearchCV

### Problème: Résultats incohérents
- Vérifier la qualité des données
- Vérifier les valeurs manquantes
- Vérifier les outliers
- S'assurer que les données sont bien normalisées

## Support et Questions

Pour toute question ou problème:
1. Vérifiez que toutes les bibliothèques sont installées
2. Vérifiez que les données sont dans le bon format
3. Consultez les messages d'erreur dans le notebook
4. Vérifiez les statistiques descriptives des données

## Licence

Ce projet est à usage académique et éducatif.

## Auteur

Projet de Data Science - Analyse de Facturation d'Eau
Date: Novembre 2025
