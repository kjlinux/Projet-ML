# PLAN DU PROJET MACHINE LEARNING
## Analyse du Déclin des Revenus FDE - Compagnie de Distribution d'Eau

---

## 1. CONTEXTE DU PROJET

### Informations Générales
- **Organisation** : AIMS-SENEGAL / Institut International de Data Science - INPHB
- **Cours** : Machine Learning 2024-2025
- **Durée** : 2 semaines
- **Date limite** : 6 décembre 2024
- **Format** : Travail en binôme

### Problématique Business
Une compagnie de distribution d'eau (le "Producteur") fait face à un **déclin continu des revenus FDE (Fonds de Développement de l'Eau)**, malgré :
- Augmentation des volumes de production d'eau
- Augmentation des volumes de distribution
- Investissements significatifs en infrastructures

### Questions Clés
1. Comment le FDE a-t-il évolué depuis 2014 (facturation et encaissement) ?
2. Qui sont les contributeurs FDE et comment évoluent-ils (par type de client et zone) ?
3. Les clients ont-ils changé leurs habitudes ? Consomment-ils moins ? Sont-ils devenus mauvais payeurs ?
4. Quels facteurs expliquent le déclin des ressources FDE ?

---

## 2. DONNÉES DISPONIBLES

### Vue d'ensemble
- **Volume** : 23 119 112 observations (23 millions+)
- **Variables** : 46 indicateurs
- **Période** : 2014-2020+
- **Régions** : Plusieurs Directions Régionales (DR2, DR6, DR7, DR9, DR16, DR21)

### Fichiers de Données
```
datasets/
├── DR2.txt (276 MB)
├── DR6.txt (228 MB)
├── DR7.txt (254 MB)
├── DR9.txt (115 MB)
├── DR16.txt (240 MB)
└── DR21.txt (431 MB)
```

### Structure des Variables

#### A. Variables DATE (7 variables)
- `DATE-FACT` : Date d'émission de facture
- `DATE-CPTA-EMIS` : Date comptable d'émission
- `DATE-CPTA-ENC` : Date comptable d'encaissement
- `DATE-ABON` : Date d'abonnement
- `DATE-RESIL` : Date de résiliation
- `DATE-REALISA` : Date de réalisation du branchement
- `DATE-REGLT` : Date de règlement de la facture

#### B. Variables QUALITATIVES (22 variables)
- `DR` : Direction régionale (découpage du pays)
- `CEN` : Centre (subdivision de DR)
- `POLICE` : Police (subdivision de centre)
- `CATEGORIE` : Catégorie client (PRIVE ou ADMINISTRATIF)
- **`RETARD`** : Paiement à temps ou en retard (0=à temps, 1=en retard) - **VARIABLE CIBLE**
- **`RESILIE`** : Abonné résilié ou non (0=résilié, 1=actif) - **VARIABLE CIBLE**
- `ENCAISSE` : Facture encaissée ou non (0=non, 1=oui)
- `NOUVEAU` : Ancienneté abonné (0=<5 ans, 1=>5 ans)
- `DIAM` : Diamètre du branchement
- `P` : Période de facturation (mensuelle/trimestrielle)
- `ENR` : Type d'enregistrement facture
- `MM` : Mois de facturation
- `AAAA` : Année de facturation

#### C. Variables QUANTITATIVES (17 variables)

**Consommation (Cubage) :**
- `CUBCONS` : Volume consommé
- `CUBFAC` : Volume facturé
- `FORFAIT` : Volume forfaitaire
- `SOCIAL` : Volume tranche sociale
- `DOMEST` : Volume tranche domestique
- `NORMAL` : Volume tranche normale
- `INDUST` : Volume tranche industrielle
- `ADMINI` : Volume tranche administrative

**Montants (Facturation) :**
- `MONT-SOD` : Montant concessionnaire (part distributeur)
- `MONT-TVA` : Montant TVA
- **`MONT-FDE`** : Montant FDE - **VARIABLE CIBLE PRINCIPALE (Régression)**
- `MONT-FNE` : Montant FNE
- `MONT-ASS-TTC` : Montant assainissement
- `MONT-FRAIS-CPT` : Montant frais compteur
- `MONT-TTC` : Montant total facture
- `DELAI_REGL` : Délai de règlement (durée entre émission et paiement)

---

## 3. OBJECTIFS DU PROJET

### Objectifs Généraux
1. **Comprendre et expliquer les profils d'abonnés**
2. **Développer un pipeline ML complet** pour :
   - Modéliser le montant FDE et la probabilité de non-paiement/résiliation
   - Segmenter les clients en groupes homogènes selon leurs comportements
   - Sélectionner les variables pertinentes
   - Valider rigoureusement les performances des modèles
   - Formuler des recommandations opérationnelles

### Variables Cibles
1. **`MONT-FDE`** (Régression) - Prédire le montant FDE
2. **`RETARD`** (Classification) - Prédire le retard de paiement
3. **`RESILIE`** (Classification) - Prédire la résiliation client

---

## 4. MÉTHODOLOGIE & TÂCHES À ACCOMPLIR

### PHASE 1 : EXPLORATION DES DONNÉES (Jours 1-2)

#### 4.1 Chargement et Prétraitement
- [ ] Charger les 6 fichiers régionaux (DR2, DR6, DR7, DR9, DR16, DR21)
- [ ] Analyser la structure des données (types, dimensions)
- [ ] Identifier et traiter les valeurs manquantes
- [ ] Gérer les doublons éventuels
- [ ] Encoder les variables qualitatives
- [ ] Normaliser/standardiser les variables quantitatives si nécessaire

#### 4.2 Analyse Descriptive
- [ ] Statistiques descriptives (moyennes, médianes, écarts-types, quartiles)
- [ ] Distribution des variables quantitatives
- [ ] Visualisations :
  - Histogrammes des consommations (CUBCONS, CUBFAC)
  - Boxplots des montants (MONT-FDE, MONT-TTC)
  - Distributions des classes cibles (RETARD, RESILIE)

#### 4.3 Analyse des Relations
- [ ] Matrice de corrélation entre variables quantitatives
- [ ] Relations entre CUBFAC, CUBCONS, MONT-FDE
- [ ] Visualisations scatter plots pour relations clés
- [ ] Vérifier la multicolinéarité

#### 4.4 Analyse Temporelle
- [ ] Évolution de MONT-FDE depuis 2014 (facturation)
- [ ] Évolution de l'encaissement (collection) depuis 2014
- [ ] Tendances saisonnières (par mois/trimestre)
- [ ] Évolution des comportements de paiement (RETARD)

#### 4.5 Analyse Segmentée
- [ ] Comportements de paiement par zone (DR, CEN)
- [ ] Comparaison PRIVE vs ADMINISTRATIF
- [ ] Analyse par ancienneté (NOUVEAU)
- [ ] Analyse par type d'émission (ENR)
- [ ] Analyse par tranches de consommation

---

### PHASE 2 : MODÉLISATION RÉGRESSION (Jours 3-5)

#### Objectif : Prédire `MONT-FDE`

#### 4.6 Préparation des Données
- [ ] Définir les features (X) et la cible (y = MONT-FDE)
- [ ] Split train/test (80/20 ou 70/30)
- [ ] Considérer un échantillonnage si nécessaire (23M observations)

#### 4.7 Modèles à Comparer
1. **MCO (Ordinary Least Squares)**
   - [ ] Entraîner le modèle
   - [ ] Évaluer avec validation croisée (5-fold ou 10-fold)
   - [ ] Calculer MSE, RMSE, R², MAE

2. **Ridge Regression**
   - [ ] Optimiser l'hyperparamètre alpha (GridSearch/RandomSearch)
   - [ ] Entraîner avec alpha optimal
   - [ ] Évaluer les performances

3. **Lasso Regression**
   - [ ] Optimiser alpha
   - [ ] Entraîner le modèle
   - [ ] Analyser les coefficients (sélection de variables)

4. **Elastic Net**
   - [ ] Optimiser alpha et l1_ratio
   - [ ] Entraîner et évaluer

5. **PCR (Principal Component Regression)**
   - [ ] Effectuer PCA
   - [ ] Choisir le nombre de composantes
   - [ ] Entraîner la régression sur les composantes
   - [ ] Évaluer les performances

6. **PLS (Partial Least Squares)**
   - [ ] Optimiser le nombre de composantes
   - [ ] Entraîner et évaluer

#### 4.8 Validation et Comparaison
- [ ] Tableau comparatif des performances (MSE, R², RMSE, MAE)
- [ ] Analyse biais/variance
- [ ] Courbes d'apprentissage
- [ ] Résidus analysis (normalité, homoscédasticité)
- [ ] Sélectionner le meilleur modèle

---

### PHASE 3 : MODÉLISATION CLASSIFICATION (Jours 6-8)

#### Objectif : Prédire `RETARD` et/ou `RESILIE`

#### 4.9 Préparation
- [ ] Définir les features et la cible (RETARD ou RESILIE)
- [ ] Vérifier l'équilibre des classes (déséquilibre ?)
- [ ] Appliquer SMOTE/sous-échantillonnage si nécessaire
- [ ] Split train/test

#### 4.10 Modèles à Comparer
1. **Régression Logistique**
   - [ ] Entraîner le modèle
   - [ ] Validation croisée
   - [ ] Matrice de confusion, métriques (Accuracy, Precision, Recall, F1-score)

2. **CART (Decision Tree)**
   - [ ] Optimiser profondeur/paramètres
   - [ ] Entraîner et évaluer
   - [ ] Visualiser l'arbre

3. **Random Forest**
   - [ ] Optimiser n_estimators, max_depth, min_samples_split
   - [ ] Entraîner et évaluer
   - [ ] Feature importance

4. **Boosting (XGBoost / AdaBoost)**
   - [ ] Optimiser hyperparamètres
   - [ ] Entraîner XGBoost
   - [ ] Entraîner AdaBoost (optionnel)
   - [ ] Évaluer les performances

5. **k-NN (k-Nearest Neighbors)**
   - [ ] Optimiser k
   - [ ] Entraîner et évaluer

#### 4.11 Validation et Comparaison
- [ ] Matrices de confusion pour chaque modèle
- [ ] Tableau comparatif (Accuracy, Precision, Recall, F1, AUC)
- [ ] Courbes ROC et calcul AUC
- [ ] Sélectionner le meilleur modèle
- [ ] Répéter pour la deuxième variable cible si nécessaire

---

### PHASE 4 : CLUSTERING (Jours 9-10)

#### Objectif : Segmenter les clients en groupes homogènes

#### 4.12 Préparation
- [ ] Sélectionner les variables pertinentes pour le clustering
- [ ] Normaliser les données
- [ ] Définir les critères de segmentation

#### 4.13 Segmentation Proposée
1. **Par Ancienneté**
   - [ ] Clustering basé sur DATE-ABON, NOUVEAU
   - [ ] Analyser les profils par groupe d'âge

2. **Par Volume de Consommation Unitaire**
   - [ ] Clustering basé sur CUBCONS, CUBFAC
   - [ ] Identifier petits/moyens/gros consommateurs

3. **Par Qualité de Paiement**
   - [ ] Clustering basé sur RETARD, DELAI_REGL, ENCAISSE
   - [ ] Identifier bons/mauvais payeurs

#### 4.14 Méthodes de Clustering
- [ ] K-Means : Optimiser k (méthode du coude, silhouette)
- [ ] Clustering Hiérarchique (optionnel)
- [ ] DBSCAN (optionnel si clusters non-sphériques)

#### 4.15 Analyse des Clusters
- [ ] Profiler chaque cluster (statistiques descriptives)
- [ ] Visualisations (PCA 2D, scatter plots)
- [ ] Interpréter les caractéristiques de chaque groupe
- [ ] Recommandations par segment

---

### PHASE 5 : SÉLECTION DE VARIABLES (Jours 9-10)

#### Objectif : Identifier les variables les plus importantes

#### 4.16 Méthodes à Comparer
1. **Ridge Regression**
   - [ ] Analyser les coefficients
   - [ ] Classer les variables par importance

2. **Lasso Regression**
   - [ ] Identifier les variables avec coefficient non-nul
   - [ ] Classer par magnitude

3. **Random Forest**
   - [ ] Feature importance (impureté moyenne)
   - [ ] Classer les variables

4. **Out-Of-Bag (OOB) Method**
   - [ ] Calculer l'erreur OOB
   - [ ] Permutation feature importance

#### 4.17 Synthèse
- [ ] Tableau comparatif des rankings de variables
- [ ] Identifier les variables les plus importantes (consensus)
- [ ] Analyser les variables explicatives retenues

---

### PHASE 6 : SYNTHÈSE ET RECOMMANDATIONS (Jours 11-14)

#### 4.18 Réponses aux Questions Business
- [ ] Comment le FDE a-t-il évolué depuis 2014 ?
- [ ] Qui sont les contributeurs FDE et comment évoluent-ils ?
- [ ] Les clients ont-ils changé leurs habitudes ?
- [ ] Quels facteurs expliquent le déclin des ressources FDE ?

#### 4.19 Recommandations Opérationnelles
- [ ] Actions pour améliorer la collecte FDE
- [ ] Segments clients à cibler en priorité
- [ ] Stratégies de recouvrement
- [ ] Prévention des résiliations
- [ ] Optimisation de la facturation

---

## 5. LIVRABLES

### 5.1 Rapport (5-15 pages)
**Structure :**
1. **Introduction**
   - Présentation du problème
   - Contexte et enjeux
   - Objectifs de l'étude

2. **Approche Méthodologique**
   - Données utilisées
   - Prétraitement
   - Méthodes appliquées
   - Justification des choix

3. **Résultats**
   - Exploration des données (visualisations clés)
   - Performance des modèles de régression
   - Performance des modèles de classification
   - Résultats du clustering
   - Sélection de variables
   - Réponses aux questions business

4. **Conclusion**
   - Synthèse des résultats
   - Recommandations opérationnelles
   - Limites de l'étude
   - Perspectives

**Annexes :**
- Code/programmes utilisés
- Tableaux détaillés
- Visualisations supplémentaires

### 5.2 Présentation
- [ ] Slides pour présentation orale
- [ ] Durée : 15-20 minutes typiquement
- [ ] Graphiques et tableaux clés
- [ ] Messages principaux et recommandations

### 5.3 Code
- [ ] Scripts Python commentés et structurés
- [ ] Jupyter Notebooks reproductibles
- [ ] Requirements.txt (dépendances)
- [ ] Documentation (README)

### 5.4 Visualisations
- [ ] Graphiques d'exploration
- [ ] Matrices de confusion
- [ ] Courbes ROC
- [ ] Tableaux de performance
- [ ] Visualisations des clusters

---

## 6. TIMELINE DÉTAILLÉ (14 jours)

### Jours 1-2 : Exploration et Prétraitement
- Chargement des données
- Nettoyage et préparation
- Analyse descriptive
- Analyse temporelle et segmentée
- Visualisations exploratoires

### Jours 3-5 : Modélisation Régression
- Préparation des features
- Entraînement des 6 modèles (MCO, Ridge, Lasso, Elastic Net, PCR, PLS)
- Validation croisée
- Comparaison et sélection

### Jours 6-8 : Modélisation Classification
- Préparation des données
- Entraînement des 5 modèles (Logistic, CART, RF, Boosting, k-NN)
- Validation et métriques
- Comparaison et sélection

### Jours 9-10 : Clustering et Sélection de Variables
- Segmentation clients (3 approches)
- Analyse des profils
- Sélection de variables (4 méthodes)
- Comparaison des importances

### Jours 11-12 : Synthèse et Rédaction
- Réponses aux questions business
- Formulation des recommandations
- Rédaction du rapport
- Création des visualisations finales

### Jours 13-14 : Finalisation
- Revue et relecture du rapport
- Préparation de la présentation
- Tests de reproductibilité du code
- Finalisation des livrables

---

## 7. DÉFIS TECHNIQUES ET POINTS D'ATTENTION

### Défis Techniques
1. **Volume de données massif (23M observations)**
   - Considérer échantillonnage stratifié
   - Optimiser le code (vectorisation, pandas efficient)
   - Utiliser Dask ou PySpark si nécessaire
   - Travailler sur un échantillon représentatif

2. **Variables corrélées**
   - CUBCONS vs CUBFAC
   - Différents montants (MONT-*)
   - Gérer la multicolinéarité

3. **Classes potentiellement déséquilibrées**
   - Vérifier distribution RETARD et RESILIE
   - Appliquer SMOTE/sous-échantillonnage si nécessaire
   - Utiliser métriques adaptées (F1, AUC)

4. **Validation rigoureuse**
   - Utiliser validation croisée systématiquement
   - Éviter l'overfitting
   - Tester sur données holdout

### Points d'Attention
- **Toujours justifier les résultats**
- **Expliquer l'approche choisie**
- **Mettre en valeur les contributions de l'analyse**
- **Formuler des recommandations actionnables**
- **Assurer la reproductibilité du code**

---

## 8. OUTILS ET TECHNOLOGIES RECOMMANDÉS

### Environnement Python
```
- Python 3.8+
- Jupyter Notebook / JupyterLab
```

### Bibliothèques Essentielles
```python
# Manipulation de données
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Régression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# Métriques
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score
)

# Déséquilibre de classes
from imblearn.over_sampling import SMOTE
```

---

## 9. RESSOURCES ET RÉFÉRENCES

### Documentation
- Scikit-learn : https://scikit-learn.org/
- XGBoost : https://xgboost.readthedocs.io/
- Pandas : https://pandas.pydata.org/
- Seaborn : https://seaborn.pydata.org/

### Fichiers du Projet
- `init.md` : Guide complet du projet
- `projetML.pdf` : Instructions officielles
- `Explication-des-variable.pdf` : Dictionnaire des variables
- `datasets/` : Données régionales (DR2, DR6, DR7, DR9, DR16, DR21)

---

## 10. CHECKLIST FINALE

### Avant Soumission
- [ ] Tous les modèles ont été entraînés et évalués
- [ ] Les résultats sont justifiés et interprétés
- [ ] Le rapport est complet (5-15 pages)
- [ ] La présentation est prête
- [ ] Le code est commenté et reproductible
- [ ] Les visualisations sont claires et pertinentes
- [ ] Les recommandations opérationnelles sont formulées
- [ ] Les annexes incluent le code
- [ ] Tous les livrables sont finalisés
- [ ] Date limite respectée : 6 décembre 2024

---

**Bonne chance pour ce projet !**

*Ce plan est un guide complet pour réussir votre projet de Machine Learning. Suivez méthodiquement chaque phase et n'hésitez pas à adapter selon vos découvertes durant l'analyse.*
