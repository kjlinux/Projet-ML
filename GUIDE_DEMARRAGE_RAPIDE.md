# Guide de Démarrage Rapide

## Étapes Rapides pour Commencer

### 1. Vérifier l'Installation (IMPORTANT)

Avant de commencer, exécutez le script de vérification:

```bash
python verifier_environnement.py
```

Ce script va:
- Vérifier que toutes les bibliothèques sont installées
- Vérifier que vos fichiers de données sont présents
- Vérifier la structure de vos données
- Vérifier que Jupyter est installé

### 2. Installer les Dépendances (si nécessaire)

Si le script de vérification indique des bibliothèques manquantes:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter --break-system-packages
```

### 3. Préparer les Données

Assurez-vous que vos fichiers de données sont dans le dossier `datasets/`:

```
datasets/
├── DR2.txt
├── DR6.txt
├── DR7.txt
├── DR9.txt
├── DR16.txt
└── DR21.txt
```

### 4. Lancer Jupyter Notebook

```bash
jupyter notebook analyse_facturation_eau.ipynb
```

Ou avec JupyterLab:

```bash
jupyter lab analyse_facturation_eau.ipynb
```

### 5. Exécuter le Notebook

Dans Jupyter:
1. Cliquez sur "Cell" > "Run All" pour exécuter toutes les cellules
2. Ou exécutez cellule par cellule avec Shift+Enter

### Temps d'Exécution Estimé

- Petit dataset (< 100,000 lignes): 5-10 minutes
- Dataset moyen (100,000 - 500,000 lignes): 15-30 minutes
- Grand dataset (> 500,000 lignes): 30-60 minutes

## Structure des Fichiers Générés

Après l'exécution, vous aurez:

```
.
├── analyse_facturation_eau.ipynb
├── README.md
├── verifier_environnement.py
├── GUIDE_DEMARRAGE_RAPIDE.md
│
├── datasets/                              # Vos données
│   └── ...
│
└── Fichiers générés:
    ├── resultats_classification_categorie.csv
    ├── resultats_classification_resiliation.csv
    ├── resultats_regression_montant.csv
    ├── meilleur_modele_categorie.pkl
    ├── meilleur_modele_resiliation.pkl
    ├── meilleur_modele_regression.pkl
    ├── scaler_categorie.pkl
    ├── scaler_resiliation.pkl
    └── scaler_regression.pkl
```

## Sections Importantes du Notebook

### Section 3: Exploration des Données
- Vérifiez les statistiques descriptives
- Identifiez les valeurs manquantes
- Analysez la distribution des classes

### Section 7: Classification des Catégories
- Compare 5 modèles différents
- Utilise validation croisée stratifiée
- Détecte le surapprentissage

### Section 8: Prédiction des Résiliations
- Traite le déséquilibre des classes avec SMOTE
- Important pour identifier les clients à risque

### Section 9: Prédiction des Montants
- Teste 6 modèles de régression
- Utilise régularisation pour éviter surapprentissage

### Section 10: Optimisation
- GridSearchCV pour optimiser les hyperparamètres
- Peut prendre du temps sur grands datasets

## Conseils pour Optimiser les Performances

### Si le notebook est trop lent:

1. **Réduire le nombre d'estimateurs:**
   ```python
   # Dans les définitions de modèles
   n_estimators=50  # au lieu de 100
   ```

2. **Réduire les folds de validation croisée:**
   ```python
   # Dans StratifiedKFold
   n_splits=3  # au lieu de 5
   ```

3. **Limiter le GridSearchCV:**
   ```python
   # Commentez ou sautez la section 10
   # Ou réduisez la grille de paramètres
   ```

4. **Utiliser un échantillon:**
   ```python
   # Après le chargement des données
   df = df.sample(n=50000, random_state=42)
   ```

### Si vous manquez de mémoire:

1. **Charger moins de fichiers:**
   ```python
   # Modifiez dans la section 2
   fichiers = fichiers[:3]  # Charger seulement les 3 premiers
   ```

2. **Supprimer des colonnes inutiles:**
   ```python
   # Après le chargement
   df = df.drop(['colonne_inutile1', 'colonne_inutile2'], axis=1)
   ```

## Interprétation Rapide des Résultats

### Pour la Classification:

**Bon modèle:**
- Accuracy > 0.80
- Différence Train-Test < 0.05
- Score CV stable (faible std)

**Surapprentissage détecté si:**
- Train Accuracy >> Test Accuracy (différence > 0.10)
- Score CV très variable (std > 0.05)

**Actions:**
- Augmenter min_samples_split
- Réduire max_depth
- Utiliser régularisation

### Pour la Régression:

**Bon modèle:**
- R² > 0.70
- RMSE faible (relatif à la plage de valeurs)
- Résidus bien distribués autour de 0

**Surapprentissage détecté si:**
- R² Train >> R² Test
- Résidus avec pattern visible

**Actions:**
- Augmenter alpha (Ridge/Lasso)
- Réduire complexité du modèle
- Feature selection

## Erreurs Courantes et Solutions

### Erreur: "FileNotFoundError: datasets/*.txt"
**Solution:** Créez le dossier `datasets/` et placez-y vos fichiers

### Erreur: "ModuleNotFoundError: No module named 'imblearn'"
**Solution:** 
```bash
pip install imbalanced-learn --break-system-packages
```

### Erreur: "ValueError: could not convert string to float"
**Solution:** Vérifiez le format de vos données, le séparateur doit être ','

### Erreur: Kernel died / Mémoire insuffisante
**Solution:** 
- Réduire la taille du dataset
- Fermer autres applications
- Utiliser un échantillon

### Warning: "ConvergenceWarning"
**Solution:** 
- Augmenter max_iter dans les modèles linéaires
- Normaliser les données (déjà fait dans le notebook)

## Questions Fréquentes

**Q: Puis-je modifier les paramètres des modèles?**
R: Oui, modifiez les paramètres dans la section 7, 8 ou 9 selon le modèle

**Q: Comment sauvegarder mes résultats?**
R: Les résultats sont automatiquement sauvegardés en CSV et PKL dans la section 11

**Q: Comment utiliser les modèles sauvegardés?**
R: 
```python
import pickle
with open('meilleur_modele_categorie.pkl', 'rb') as f:
    modele = pickle.load(f)
```

**Q: Puis-je ajouter d'autres modèles?**
R: Oui, ajoutez-les dans le dictionnaire `modeles_classification` ou `modeles_regression`

**Q: Comment améliorer les performances?**
R: 
1. Créer de meilleures features (section 4)
2. Optimiser les hyperparamètres (section 10)
3. Essayer d'autres modèles (XGBoost, LightGBM)
4. Augmenter la quantité de données

## Prochaines Étapes

Après avoir exécuté le notebook:

1. **Analyser les résultats:**
   - Consultez les fichiers CSV générés
   - Examinez les visualisations
   - Identifiez le meilleur modèle

2. **Optimiser si nécessaire:**
   - Ajustez les hyperparamètres
   - Créez de nouvelles features
   - Testez d'autres modèles

3. **Déployer en production:**
   - Utilisez les modèles .pkl sauvegardés
   - Mettez en place un monitoring
   - Réentraînez régulièrement

4. **Documenter:**
   - Notez vos observations
   - Documentez les décisions
   - Partagez les insights

## Support

En cas de problème:
1. Vérifiez les prérequis avec `verifier_environnement.py`
2. Consultez le README.md pour plus de détails
3. Vérifiez les logs d'erreur dans le notebook
4. Assurez-vous que vos données sont au bon format

## Résumé des Commandes

```bash
# Vérifier l'environnement
python verifier_environnement.py

# Installer les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter --break-system-packages

# Lancer Jupyter
jupyter notebook analyse_facturation_eau.ipynb

# Ou avec JupyterLab
jupyter lab analyse_facturation_eau.ipynb
```

Bon travail!
