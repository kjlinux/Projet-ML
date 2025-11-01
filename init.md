# Projet Machine Learning - Water IA (AIMS-SENEGAL)

## Vue d'ensemble du projet

**Institution:** AIMS-SENEGAL / International Data Science Institute - INPHB  
**Cours:** Machine Learning 2024-2025  
**Durée:** 2 semaines  
**Date limite:** 6 décembre 2024

---

## 1. Contexte et problématique

### Organisation
Le projet porte sur l'analyse des données d'un organisme de distribution d'eau (le Producteur) confronté à une **baisse continue des revenus du FDE** (Fonds de Développement de l'Eau), malgré l'augmentation des volumes d'eau produite et distribuée suite à d'importants investissements.

### Questions principales
1. Comment évolue le FDE depuis 2014 (facturation et recouvrement) ?
2. Quels sont les contributeurs du FDE et comment évoluent-ils (par type de client et par zone) ?
3. Les clients ont-ils changé leurs habitudes ? Consomment-ils moins ? Sont-ils devenus de mauvais payeurs ?
4. Quels sont les facteurs qui expliquent la baisse des ressources du FDE ?

---

## 2. Objectifs du projet

### Objectifs généraux
- Comprendre et expliquer le profil des abonnés
- Développer une chaîne de traitement complète en Machine Learning pour :
  - Modéliser le montant du FDE et la probabilité de non-paiement ou de résiliation
  - Segmenter les clients en groupes homogènes selon leur comportement
  - Sélectionner les variables pertinentes
  - Valider rigoureusement les performances des modèles
  - Formuler des recommandations opérationnelles

### Thèmes d'analyse suggérés
1. **Segmentation des clients** par :
   - Ancienneté
   - Volume unitaire de consommation
   - Qualité de paiement des factures

2. **Granulométrie de l'analyse** par :
   - Type d'émission
   - Catégorie de client
   - Zone géographique

3. **Analyse de l'évolution** :
   - Facturation et recouvrement (paiement des factures)
   - Par type de client, qualité de client, zones géographiques
   - Par tranches de facturation
   - De façon globale et selon le FDE

---

## 3. Données

### Caractéristiques générales
- **Volume:** ~23 millions d'observations
- **Variables:** 46 indicateurs
- **Objectifs à prédire:** RETARD, RESILIE, MONT-FDE

### Variables par catégorie

#### A. Variables de DATE (7 variables)
| Variable | Description |
|----------|-------------|
| DATE-FACT | Date d'édition de la facture |
| DATE-CPTA-EMIS | Date comptable émission |
| DATE-CPTA-ENC | Date comptable encaissement |
| DATE-ABON | Date de l'abonnement |
| DATE-RESIL | Date de résiliation de l'abonné |
| DATE-REALISA | Date de réalisation du branchement |
| DATE-REGLT | Date de règlement de la facture |

#### B. Variables QUALITATIVES (22 variables)
| Variable | Description | Valeurs |
|----------|-------------|---------|
| DR | Direction régionale (découpage du pays) | |
| CEN | Centre (subdivision d'une DR) | |
| POLICE | Police (subdivision d'un CENTRE) | |
| O | Ordre de l'Abonné | |
| C | CLE contrôle saisie | |
| P | PERIODE de facturation | Mensuelle/Trimestrielle |
| C/C | Compte client | |
| ENR | Type d'enregistrement de la facture | |
| MM | Mois de facturation | |
| AAAA | Année de facturation | |
| ABONNE | Identifiant (CENTRE + POLICE + ORDRE) | |
| **RETARD** | **Paiement à temps ou en retard** | **0 = à temps, 1 = en retard** |
| **RESILIE** | **Abonné résilié ou non** | **0 = résilié, 1 = non résilié** |
| ENCAISSE | Facture encaissée ou pas | 0 = non encaissé, 1 = encaissé |
| CATEGORIE | Catégorie du client | PRIVE ou ADMINISTRATIF |
| NOUVEAU | Age de l'abonné | 0 = < 5 ans, 1 = > 5 ans |
| DIAM | Diamètre du branchement | |
| AnABON | Année d'abonnement | |
| NumMOIS | Numérotation du mois (analyse statistique) | |
| NumANNEE | Numérotation de l'année (analyse statistique) | QT |

#### C. Variables QUANTITATIVES (17 variables)

**Cubages (consommation):**
| Variable | Description |
|----------|-------------|
| CUBCONS | Cubage consommé |
| CUBFAC | Cubage facturé |
| FORFAIT | Cubage Forfait |
| SOCIAL | Cubage Social |
| DOMEST | Cubage Domestique |
| NORMAL | Cubage Normal |
| INDUST | Cubage Industriel |
| ADMINI | Cubage Administratif |

**Montants (facturation):**
| Variable | Description |
|----------|-------------|
| MONT-SOD | Montant Concessionnaire (part dans la facture) |
| MONT-TVA | Montant TVA |
| **MONT-FDE** | **Montant FDE (Variable cible pour régression)** |
| MONT-FNE | Montant FNE |
| MONT-ASS-TTC | Montant Assainissement |
| MONT-FRAIS-CPT | Montant frais compteur |
| MONT-TTC | Montant TTC (total de la facture) |
| DELAI_REGL | Délai de règlement (durée entre émission et règlement) |

---

## 4. Tâches à réaliser

### 4.1 Exploration des données
- Décrire la distribution des variables quantitatives
- Étudier les relations entre CUBFAC, CUBCONS, MONT-FDE, etc.
- Visualiser les comportements de paiement par zone et par type
- Vérifier les corrélations
- Analyser les valeurs manquantes

### 4.2 Régression (étude comparative)
**Objectif:** Modéliser MONT-FDE

**Modèles à comparer:**
- MCO (Moindres Carrés Ordinaires)
- Ridge
- Lasso
- Elastic Net
- PCR (Principal Component Regression)
- PLS (Partial Least Squares)

**Validation:**
- Utiliser la validation croisée
- Évaluer les performances : MSE, R², biais/variance

### 4.3 Classification (étude comparative)
**Objectifs:** Prédire RETARD ou RESILIE

**Modèles à comparer:**
- Régression logistique
- CART (arbres de décision)
- Random Forest
- Boosting
- k-NN

**Validation:**
- Construire la matrice de confusion
- Évaluer avec : Accuracy, Precision, Recall, F1-score, AUC

### 4.4 Clustering
- Segmenter les clients en groupes homogènes selon leur comportement

### 4.5 Sélection de variables
**Méthodes à comparer:**
- Ridge (coefficients)
- Lasso (coefficients)
- Random Forest (importance des variables)
- Méthode Out-Of-Bag
- Analyser les variables explicatives retenues

---

## 5. Livrables attendus

### 5.1 Rapport de projet
- **Format:** 5 à 10 pages (partie rédigée)
- **Structure:**
  - Introduction présentant le problème posé
  - Démarche choisie et apports des analyses
  - Résultats (justifiés)
  - Conclusion avec recommandations opérationnelles
- **Annexes:** Programmes utilisés

### 5.2 Présentation
- Slides de présentation

### 5.3 Code
- Python/R propre et commenté
- Reproductible

### 5.4 Visualisations
- Graphiques
- Matrices de confusion
- Tableaux d'évaluation des performances

---

## 6. Méthodologie recommandée

### Étape 1: Prétraitement
1. Charger les données (23M observations)
2. Analyser la nature statistique des variables (cf. tableaux)
3. Traiter les valeurs manquantes
4. Encoder les variables qualitatives
5. Normaliser/standardiser les variables quantitatives si nécessaire

### Étape 2: Analyse exploratoire
1. Statistiques descriptives
2. Visualisations (distributions, boxplots, histogrammes)
3. Analyse de corrélations
4. Analyse temporelle (évolution depuis 2014)
5. Analyse par segments (DR, CATEGORIE, etc.)

### Étape 3: Modélisation
1. Définir train/test split
2. Pour la régression (MONT-FDE):
   - Tester tous les modèles
   - Valider avec CV
   - Comparer les performances
3. Pour la classification (RETARD/RESILIE):
   - Tester tous les modèles
   - Valider avec CV
   - Comparer les performances
4. Pour le clustering:
   - Choisir le nombre optimal de clusters
   - Interpréter les profils

### Étape 4: Sélection de variables
1. Appliquer les différentes méthodes
2. Comparer les résultats
3. Identifier les variables les plus importantes

### Étape 5: Interprétation et recommandations
1. Synthétiser les résultats
2. Répondre aux questions initiales
3. Formuler des recommandations opérationnelles

---

## 7. Points d'attention

### Techniques
- **Volume de données:** 23M observations → optimiser le code, possibilité d'échantillonner
- **Variables corrélées:** CUBCONS vs CUBFAC, différents montants
- **Classes déséquilibrées:** Vérifier la distribution de RETARD et RESILIE
- **Validation rigoureuse:** Utiliser la validation croisée systématiquement

### Analyse
- Toujours **justifier les résultats**
- Mettre en évidence la **démarche choisie**
- Expliquer les **apports des analyses**
- Formuler des **recommandations actionnables**

---

## 8. Ressources

### Fichiers fournis
- `projetML.pdf` : Description du projet
- `Explication-des-variable.pdf` : Documentation des variables
- Base de données (à préciser)

### Documents
- Documents non autorisés (projet individuel/binôme)

---

## 9. Organisation du travail

### Format
- Travail en **binôme**
- Chaque binôme choisit un jeu de données dans la base transmise

### Planning suggéré
- **Jours 1-2:** Exploration et prétraitement
- **Jours 3-5:** Modélisation régression
- **Jours 6-8:** Modélisation classification
- **Jours 9-10:** Clustering et sélection de variables
- **Jours 11-12:** Synthèse et rédaction
- **Jours 13-14:** Finalisation et préparation présentation