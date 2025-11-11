# Water IA ML Project Notebook - Structure Analysis

## Critical Cells Identified for Modification

### 1. RETARD Column Creation
**Cell 10** (Execution Count: 7)
- **Location**: Feature Engineering section
- **Purpose**: Creates RETARD column (payment delay indicator)
- **Current Code**:
\- **Status**: Working correctly

---

### 2. Regression Feature Selection (MONT-FDE prediction)
**Cell 24** (Execution Count: 19)
- **Location**: Regression Models section
- **Purpose**: Selects features for MONT-FDE prediction
- **Current Feature List**:
\- **Note**: MONT-TTC is excluded (all NaN values)
- **Recommendation**: ADD correlation analysis and VIF calculation HERE

---

### 3. Classification Feature Selection & X_clf Creation (BUG LOCATION)
**Cell 35** (Execution Count: 29)
- **Location**: Classification Models section  
- **Purpose**: Prepares features for RETARD prediction
- **BUG**: Includes MONT-TTC which has all NaN values
- **Current Feature List**:
\- **FIX REQUIRED**: Remove "MONT-TTC" from feature_cols_clf
