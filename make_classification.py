import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

# --- 1. CONFIGURATION ---

# Synthetic Dataset Parameters
# N_SAMPLES = 100       # Total number of data points
N_FEATURES = 5        # Number of features
# N_INFORMATIVE = 3     # Number of features that are actually useful
# N_CLASSES = 2         # Number of classes (e.g., 3)
# CLASS_WEIGHTS = [0.9, 0.1] # Class Imbalance
# RANDOM_STATE = 42     # For reproducibility

# Output File Paths
OUTPUT_CSV_PATH = 'Data/New-Datasets-D=2/DS4.csv'
OUTPUT_DAT_PATH = 'Data/New-Datasets-D=2/DS4.dat'

# Pyomo Model Parameters
DEPTH = 2
NMIN = 3
ALPHA = 0.01

# --- 2. GENERATE AND SAVE SYNTHETIC DATASET ---

print("Generating synthetic dataset...")
X, y_raw = make_classification(n_samples=700, n_features=5, n_informative=3,
                               n_redundant=0, n_classes=2, n_clusters_per_class=2,
                               weights=[0.5, 0.5], flip_y=0.0, class_sep=2.0,
                               random_state=18)

# Create a DataFrame for easy manipulation and saving
df = pd.DataFrame(X, columns=[f'Feature_{j+1}' for j in range(N_FEATURES)])
# Note: make_classification creates labels 0..K-1. We will use these for the CSV.
df['Class'] = y_raw
df.insert(0, 'ID', range(1, len(df) + 1))

# Save the generated dataset to a CSV file
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Successfully generated and saved dataset to '{OUTPUT_CSV_PATH}'")
print("\nDataset Preview:")
print(df.head())
print("\nActual Class Distribution:")
print(df['Class'].value_counts(normalize=True).sort_index())


# --- 3. PREPARE DATA & GENERATE .DAT FILE ---

print(f"\nGenerating .dat file for Pyomo model...")
# Normalize features to [0, 1] range for the model
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels to 1..K for the Pyomo model format
y = y_raw + 1

# Get model dimensions
n, p = X_scaled.shape
K = len(np.unique(y))
T = 2**(DEPTH + 1) - 1
TB = list(range(1, 2**DEPTH))
TL = list(range(2**DEPTH, T + 1))

# Helper function to find ancestors for a leaf node
def get_ancestors(t):
    AL, AR = [], []
    # Note: A's list of all ancestors is not needed by the model file
    while t > 1:
        parent = t // 2
        if t % 2 == 0:
            AL.append(parent)
        else:
            AR.append(parent)
        t = parent
    return AL, AR

# Precompute ancestor sets for all terminal leaves
AL_map, AR_map = {}, {}
for t in TL:
    AL_map[t], AR_map[t] = get_ancestors(t)

# Write the .dat file
with open(OUTPUT_DAT_PATH, 'w') as f:
    # Write model parameters
    f.write(f"param DEPTH := {DEPTH};\n")
    f.write(f"param Nmin := {NMIN};\n")
    # Lhat is the misclassification of a baseline model (always predict majority class)
    majority_class = np.bincount(y).argmax()
    f.write(f"param Lhat := {np.sum(y != majority_class)};\n")
    f.write(f"param alpha := {ALPHA};\n")

    # Write sets
    f.write("\nset I := " + " ".join(str(i+1) for i in range(n)) + ";\n")
    f.write("set J := " + " ".join(str(j+1) for j in range(p)) + ";\n")
    f.write("set K := " + " ".join(str(k+1) for k in range(K)) + ";\n")
    f.write("set TB := " + " ".join(str(t) for t in TB) + ";\n")
    f.write("set TL := " + " ".join(str(t) for t in TL) + ";\n")

    # Write X data
    f.write("\nparam X :=\n")
    for i in range(n):
        for j in range(p):
            f.write(f"{i+1} {j+1} {X_scaled[i,j]:.6f}\n")
    f.write(";\n")

    # Write Y data
    f.write("\nparam Y :=\n")
    for i in range(n):
        for k in range(1, K + 1):
            f.write(f"{i+1} {k} {1 if y[i] == k else -1}\n")
    f.write(";\n")

    # Write Ancestor sets
    f.write("\n")
    for t in TL:
        if AL_map[t]:
            f.write(f"set AL[{t}] := " + " ".join(str(m) for m in AL_map[t]) + ";\n")
        if AR_map[t]:
            f.write(f"set AR[{t}] := " + " ".join(str(m) for m in AR_map[t]) + ";\n")

print(f"Successfully generated Pyomo data file at '{OUTPUT_DAT_PATH}'")