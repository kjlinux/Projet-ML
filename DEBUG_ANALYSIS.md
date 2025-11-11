# X_clf Empty DataFrame Issue - Root Cause Analysis

## Problem Summary
In cell 37 (CODE CELL 29 in the notebook), `train_test_split()` fails because `X_clf` has 0 samples.

Error: `ValueError: n_samples=0` at `train_test_split(X_clf, y_clf, ...)`

## Code Location

### Cell 29 (Notebook Index 35) - Classification Setup - LINES 1-18

Feature columns definition:
```python
feature_cols_clf = [
    'CUBCONS', 'CUBFAC', 'FORFAIT', 'SOCIAL', 'DOMEST', 'NORMAL', 'INDUST', 'ADMINI',
    'MONT-FDE', 'MONT-TTC', 'MONT-SOD', 'DIAM', 'TENURE_YEARS',
    'DR_ENCODED', 'CEN_ENCODED', 'ENR_ENCODED', 'CATEGORIE_ENCODED', 'P_ENCODED',
    'MM', 'TRIMESTRE', 'NUOVO', 'RESILIE'
]

# Remove rows with missing target
df_classification = df[df['RETARD'].notna()].copy()

# Prepare X and y
X_clf = df_classification[feature_cols_clf]
y_clf = df_classification['RETARD']

# Remove any remaining NaN values
mask = ~(X_clf.isna().any(axis=1) | y_clf.isna())
X_clf = X_clf[mask]
y_clf = y_clf[mask]

print(f"Classification dataset: {X_clf.shape[0]} samples, {X_clf.shape[1]} features")
```

### Cell 30 (Notebook Index 36) - Train-Test Split - LINES 1-3

```python
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
```

## Root Causes Identified

### CAUSE #1: RETARD Column Might Be All NaN
Cell 10 creates RETARD:
```python
df['RETARD'] = ((df['DELAI_REGL'] > 30) | df['DATE-REGLT'].isnull()).astype(int)
```

Where DELAI_REGL is:
```python
df['DELAI_REGL'] = (df['DATE-REGLT'] - df['DATE-FACT']).dt.days
```

If DATE-FACT or DATE-REGLT columns have issues, DELAI_REGL becomes all NaN, making the RETARD filter drop all rows.

### CAUSE #2: After NaN Removal, All Rows Dropped
The mask operation on line 17-18:
```python
mask = ~(X_clf.isna().any(axis=1) | y_clf.isna())
X_clf = X_clf[mask]
y_clf = y_clf[mask]
```

If ANY of the feature columns has NaN in every row after filtering, this mask will be all False, making X_clf empty.

### CAUSE #3: Missing Encoded Columns
The feature list includes:
- 'DR_ENCODED'
- 'CEN_ENCODED'
- 'ENR_ENCODED'
- 'CATEGORIE_ENCODED'
- 'P_ENCODED'

These are created in Cell 12. If that cell didn't run or failed, these columns won't exist, causing X_clf[feature_cols_clf] to fail.

## Data Creation Flow

| Cell | Operation | Details |
|------|-----------|---------|
| 10   | Feature Engineering | Creates RETARD, TENURE_YEARS, DELAI_REGL, etc. |
| 11   | Missing Value Handling | Fills NaN with median/mode |
| 12   | Categorical Encoding | Creates _ENCODED columns for: DR, CEN, ENR, CATEGORIE, P |
| 13   | Display Info | Shows sample of processed data |
| 29   | Classification Setup | **FILTERS AND SELECTS FEATURES** |
| 30   | Train-Test Split | **FAILS: X_clf is empty** |

## Where RETARD is Created (Cell 10)

```python
# Payment delay (days between invoice and payment)
df['DELAI_REGL'] = (df['DATE-REGLT'] - df['DATE-FACT']).dt.days

# Payment status: late payment indicator (payment delay > 30 days)
df['RETARD'] = ((df['DELAI_REGL'] > 30) | df['DATE-REGLT'].isnull()).astype(int)
```

Possible issues:
1. DATE-FACT might not be parsed as datetime
2. DATE-REGLT might not be parsed as datetime
3. Date subtraction results in NaN if either date is missing
4. NaN > 30 evaluates to False, not True

## How to Debug This

Add these lines BEFORE Cell 29 to diagnose:

```python
print("=== DEBUGGING X_clf Empty Issue ===")
print(f"
1. DataFrame shape: {df.shape}")
print(f"2. Columns in df: {df.columns.tolist()}")

# Check RETARD
print(f"
3. RETARD column stats:")
print(f"   - Exists: {'RETARD' in df.columns}")
print(f"   - Non-null count: {df['RETARD'].notna().sum()}")
print(f"   - Data type: {df['RETARD'].dtype}")
print(f"   - Unique values: {df['RETARD'].unique()}")

# Check date columns
print(f"
4. Date columns (for RETARD creation):")
print(f"   - DATE-FACT non-null: {df['DATE-FACT'].notna().sum()}")
print(f"   - DATE-REGLT non-null: {df['DATE-REGLT'].notna().sum()}")
print(f"   - DELAI_REGL non-null: {df['DELAI_REGL'].notna().sum()}")

# Check encoded columns
print(f"
5. Encoded columns:")
for col in ['DR_ENCODED', 'CEN_ENCODED', 'ENR_ENCODED', 'CATEGORIE_ENCODED', 'P_ENCODED']:
    print(f"   - {col}: {col in df.columns}")

# Check after RETARD filter
df_classification = df[df['RETARD'].notna()].copy()
print(f"
6. After df_classification = df[df['RETARD'].notna()]:")
print(f"   - Rows: {len(df_classification)}")

# Check feature availability
print(f"
7. Feature columns check:")
missing_features = [col for col in feature_cols_clf if col not in df_classification.columns]
if missing_features:
    print(f"   - MISSING: {missing_features}")
else:
    print(f"   - All {len(feature_cols_clf)} features present")

# Check X_clf before mask
X_clf_before = df_classification[feature_cols_clf]
print(f"
8. X_clf shape BEFORE mask: {X_clf_before.shape}")
print(f"   - NaN counts per column:")
for col in feature_cols_clf[:5]:  # Show first 5
    print(f"     {col}: {X_clf_before[col].isna().sum()}")

# Check mask
y_clf = df_classification['RETARD']
mask = ~(X_clf_before.isna().any(axis=1) | y_clf.isna())
print(f"
9. Mask statistics:")
print(f"   - True values (keep rows): {mask.sum()}")
print(f"   - False values (drop rows): {(~mask).sum()}")
```

## Most Likely Issue

The RETARD column might have been created but df_classification becomes empty after filtering because RETARD contains all NaN or all 0/1 values with no variation that causes the mask to drop everything.

Check if:
1. RETARD is being created correctly (not all NaN)
2. All feature columns exist in df after encoding
3. The mask operation is dropping all rows due to NaN in features
