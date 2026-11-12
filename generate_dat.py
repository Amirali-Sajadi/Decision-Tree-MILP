# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
#
# # CONFIG
# INPUT_PATH = 'Data/Blood_Transfusion.csv'
# OUTPUT_PATH = 'Data/Blood_Transfusion.dat'
# DEPTH = 2
# NMIN = 3
# ALPHA = 0.01
#
# df = pd.read_csv(INPUT_PATH)
# X = df.iloc[:, 1:-1].values
# X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
#
# # Encode labels to 1..K
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(df.iloc[:, -1]) + 1
#
# n, p = X.shape
# K = len(np.unique(y))
# T = 2**(DEPTH + 1) - 1
# TB = list(range(1, 2**DEPTH))
# TL = list(range(2**DEPTH, T + 1))
#
# # AL(t) and AR(t)
# def get_ancestors(t):
#     A, AL, AR = [], [], []
#     while t > 1:
#         parent = t // 2
#         A.append(parent)
#         if t % 2 == 0:
#             AL.append(parent)
#         else:
#             AR.append(parent)
#         t = parent
#     return AL, AR
#
# AL_map, AR_map = {}, {}
# for t in TL:
#     AL_map[t], AR_map[t] = get_ancestors(t)
#
# with open(OUTPUT_PATH, 'w') as f:
#     f.write(f"param DEPTH := {DEPTH};\n")
#     f.write(f"param Nmin := {NMIN};\n")
#     f.write(f"param Lhat := {np.sum(y != np.bincount(y - 1).argmax() + 1)};\n")
#     f.write(f"param alpha := {ALPHA};\n")
#     # f.write(f"param epsilon_max := {epsilon_max};\n")
#
#     f.write("\nset I := " + " ".join(str(i+1) for i in range(n)) + ";\n")
#     f.write("set J := " + " ".join(str(j+1) for j in range(p)) + ";\n")
#     f.write("set K := " + " ".join(str(k+1) for k in range(K)) + ";\n")
#     f.write("set TB := " + " ".join(str(t) for t in TB) + ";\n")
#     f.write("set TL := " + " ".join(str(t) for t in TL) + ";\n")
#
#     f.write("\nparam X :=\n")
#     for i in range(n):
#         for j in range(p):
#             f.write(f"{i+1} {j+1} {X[i,j]:.6f}\n")
#     f.write(";\n")
#
#     f.write("\nparam Y :=\n")
#     for i in range(n):
#         for k in range(1, K+1):
#             f.write(f"{i+1} {k} {1 if y[i] == k else -1}\n")
#     f.write(";\n")
#
#     # f.write("\nparam epsilon :=\n")
#     # for j in range(p):
#     #     f.write(f"{j+1} {epsilons[j]}\n")
#     # f.write(";\n")
#
#     f.write("\n")  # AL and AR
#     for t in TL:
#         if AL_map[t]:
#             f.write(f"set AL[{t}] := " + " ".join(str(m) for m in AL_map[t]) + ";\n")
#         if AR_map[t]:
#             f.write(f"set AR[{t}] := " + " ".join(str(m) for m in AR_map[t]) + ";\n")






'''New Generate_Dat Script'''

# # generate_dat.py
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
#
# # CONFIG
# INPUT_PATH = 'Data/Categorical/Zoo.csv'
# OUTPUT_PATH = 'Data/Categorical/Zoo.dat'
# DEPTH = 2
# NMIN = 3
# ALPHA = 0.01
#
# df = pd.read_csv(INPUT_PATH)
#
# # Features = all columns except the first and the last (you used this convention)
# feat_df = df.iloc[:, 1:-1]
# y_raw = df.iloc[:, -1]
#
# # Detect numeric vs categorical
# num_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
# cat_cols = [c for c in feat_df.columns if c not in num_cols]
#
# # Scale numeric features into [0,1]
# X_num = None
# if len(num_cols) > 0:
#     X_num = feat_df[num_cols].astype(float).copy()
#     X_num = (X_num - X_num.min(axis=0)) / (X_num.max(axis=0) - X_num.min(axis=0) + 1e-8)
#
# # Encode labels to 1..K
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y_raw) + 1
#
# n = len(df)
# pN = len(num_cols)
# pC = len(cat_cols)
# K = len(np.unique(y))
# T = 2**(DEPTH + 1) - 1
# TB = list(range(1, 2**DEPTH))
# TL = list(range(2**DEPTH, T + 1))
#
# # Build AL(t) and AR(t)
# def get_ancestors(t):
#     AL, AR = [], []
#     while t > 1:
#         parent = t // 2
#         if t % 2 == 0:
#             AL.append(parent)  # came from left
#         else:
#             AR.append(parent)  # came from right
#         t = parent
#     return AL, AR
#
# AL_map, AR_map = {}, {}
# for t in TL:
#     AL_map[t], AR_map[t] = get_ancestors(t)
#
# # Categorical levels: index levels per categorical feature as 1..|V_j|
# cat_levels = {}        # feature name -> list of levels (labels)
# cat_level_to_id = {}   # feature name -> dict(level -> id)
# for c in cat_cols:
#     levels = pd.Categorical(feat_df[c].astype('category')).categories.tolist()
#     # Ensure every observed level is listed (even if some rows are NaN)
#     # If NaNs exist, treat them as a separate category "NaN"
#     if feat_df[c].isna().any():
#         levels = levels + ["__NaN__"]
#     cat_levels[c] = levels
#     cat_level_to_id[c] = {lev: idx+1 for idx, lev in enumerate(levels)}
#
# # Build Delta[i,j,c] (i in 1..n, j in 1..pC, c in V_j)
# # Map categorical features to JC indices 1..pC in the order of cat_cols
# # Map numeric features to JN indices 1..pN in the order of num_cols
# with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
#     f.write(f"param DEPTH := {DEPTH};\n")
#     f.write(f"param Nmin := {NMIN};\n")
#
#     # Baseline misclassification (same as your original approach)
#     # Majority-class errors count
#     majority = np.bincount(y - 1).argmax() + 1
#     Lhat = int(np.sum(y != majority))
#     f.write(f"param Lhat := {Lhat};\n")
#     f.write(f"param alpha := {ALPHA};\n\n")
#
#     # Sets
#     f.write("set I := " + " ".join(str(i+1) for i in range(n)) + ";\n")
#     if pN > 0:
#         f.write("set JN := " + " ".join(str(j+1) for j in range(pN)) + ";\n")
#     else:
#         f.write("set JN := ;\n")
#     if pC > 0:
#         f.write("set JC := " + " ".join(str(j+1) for j in range(pC)) + ";\n")
#     else:
#         f.write("set JC := ;\n")
#     f.write("set K := " + " ".join(str(k+1) for k in range(K)) + ";\n")
#     f.write("set TB := " + " ".join(str(t) for t in TB) + ";\n")
#     f.write("set TL := " + " ".join(str(t) for t in TL) + ";\n\n")
#
#     # Numeric data
#     if pN > 0:
#         f.write("param X :=\n")
#         for i in range(n):
#             for j in range(pN):
#                 f.write(f"{i+1} {j+1} {X_num.iloc[i, j]:.6f}\n")
#         f.write(";\n\n")
#
#     # Labels Y (+1 / -1 per class)
#     f.write("param Y :=\n")
#     for i in range(n):
#         for k in range(1, K+1):
#             f.write(f"{i+1} {k} {1 if y[i] == k else -1}\n")
#     f.write(";\n\n")
#
#     # Ancestor sets
#     for t in TL:
#         if AL_map[t]:
#             f.write(f"set AL[{t}] := " + " ".join(str(m) for m in AL_map[t]) + ";\n")
#         if AR_map[t]:
#             f.write(f"set AR[{t}] := " + " ".join(str(m) for m in AR_map[t]) + ";\n")
#     f.write("\n")
#
#     # Categorical: sets of levels per categorical feature j in JC
#     if pC > 0:
#         # V[j] sets
#         for j_idx, c in enumerate(cat_cols, start=1):
#             Vj = cat_levels[c]
#             if len(Vj) == 0:
#                 f.write(f"set V[{j_idx}] := ;\n")
#             else:
#                 f.write(f"set V[{j_idx}] := " + " ".join(str(cid) for cid in range(1, len(Vj)+1)) + ";\n")
#         f.write("\n")
#
#         # Delta(i,j,c)
#         f.write("param Delta :=\n")
#         for i in range(n):
#             for j_idx, c in enumerate(cat_cols, start=1):
#                 val = feat_df[c].iloc[i]
#                 if pd.isna(val):
#                     key = "__NaN__"
#                 else:
#                     key = val
#                 cid = cat_level_to_id[c][key]
#                 # 1 at the observed category, 0 for others in V[j]
#                 Vj = cat_levels[c]
#                 for ccid in range(1, len(Vj)+1):
#                     v = 1 if ccid == cid else 0
#                     f.write(f"{i+1} {j_idx} {ccid} {v}\n")
#         f.write(";\n\n")

# generate_dat.py  -- robust numeric/categorical detection
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
INPUT_PATH = 'Data/Categorical/Adult_100.csv'   # <- put your CSV here
OUTPUT_PATH = 'Data/Categorical/Adult_100.dat'
DEPTH = 2
NMIN = 3
ALPHA = 0.01

# If a numeric column has <= this many unique values, treat it as categorical
MAX_CAT_UNIQUE = 10

# Optionally force specific feature names to be categorical (useful if auto-detect fails)
FORCE_CAT = []   # e.g. ['legs']

# ------------------------------
df = pd.read_csv(INPUT_PATH)
# Assume same layout as your pipeline: first col = id, last col = class, features in between
feat_df = df.iloc[:, 1:-1].copy()
target_ser = df.iloc[:, -1].copy()

n = len(df)

# Decide numeric vs categorical
num_cols = []
cat_cols = []
for col in feat_df.columns:
    ser = feat_df[col]
    # Forced categorical takes precedence
    if col in FORCE_CAT:
        cat_cols.append(col)
        continue
    # Non-numeric dtypes -> categorical
    if ser.dtype.kind not in 'biufc':  # not boolean/integer/unsigned/float/complex
        cat_cols.append(col)
        continue
    # Numeric dtype: check unique count (after dropping NaNs)
    nuniq = ser.dropna().unique().size
    # If few distinct values -> categorical
    if nuniq <= MAX_CAT_UNIQUE:
        cat_cols.append(col)
    else:
        num_cols.append(col)

pN = len(num_cols)
pC = len(cat_cols)

print("Detected numeric cols (JN):", num_cols)
print("Detected categorical cols (JC):", cat_cols)

# Normalize numeric features into [0,1] if exist
X_num = None
if pN > 0:
    X_num = feat_df[num_cols].astype(float)
    X_num = (X_num - X_num.min(axis=0)) / (X_num.max(axis=0) - X_num.min(axis=0) + 1e-8)

# Encode labels to 1..K
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(target_ser.fillna("##NA##")) + 1
K = len(np.unique(y))

# Build tree index sets
T = 2**(DEPTH + 1) - 1
TB = list(range(1, 2**DEPTH))
TL = list(range(2**DEPTH, T + 1))

# Ancestors AL and AR per leaf
def get_ancestors(t):
    AL, AR = [], []
    while t > 1:
        parent = t // 2
        if t % 2 == 0:
            AL.append(parent)
        else:
            AR.append(parent)
        t = parent
    return AL, AR

AL_map, AR_map = {}, {}
for t in TL:
    AL_map[t], AR_map[t] = get_ancestors(t)

# Prepare categorical levels and mapping
cat_levels = {}        # feature name -> list of levels (preserve order)
cat_level_to_id = {}
for c in cat_cols:
    # treat NA as separate category
    ser = feat_df[c].astype(object).fillna("__NA__")
    levels = pd.Categorical(ser).categories.tolist()
    # levels is an Index — convert to list of str
    levels = [str(l) for l in levels]
    cat_levels[c] = levels
    cat_level_to_id[c] = {lev: idx+1 for idx, lev in enumerate(levels)}

# compute majority-class baseline
majority = np.bincount(y - 1).argmax() + 1
Lhat = int(np.sum(y != majority))

# ---------- write .dat ----------
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(f"param DEPTH := {DEPTH};\n")
    f.write(f"param Nmin := {NMIN};\n")
    f.write(f"param alpha := {ALPHA};\n")
    f.write(f"param Lhat := {Lhat};\n\n")

    # Sets
    f.write("set I := " + " ".join(str(i+1) for i in range(n)) + ";\n")
    if pN > 0:
        f.write("set JN := " + " ".join(str(j+1) for j in range(pN)) + ";\n")
    else:
        f.write("set JN := ;\n")
    if pC > 0:
        f.write("set JC := " + " ".join(str(j+1) for j in range(pC)) + ";\n")
    else:
        f.write("set JC := ;\n")
    f.write("set K := " + " ".join(str(k+1) for k in range(K)) + ";\n")
    f.write("set TB := " + " ".join(str(t) for t in TB) + ";\n")
    f.write("set TL := " + " ".join(str(t) for t in TL) + ";\n\n")

    # Ancestor sets AL/AR
    for t in TL:
        if AL_map[t]:
            f.write(f"set AL[{t}] := " + " ".join(str(m) for m in AL_map[t]) + ";\n")
        if AR_map[t]:
            f.write(f"set AR[{t}] := " + " ".join(str(m) for m in AR_map[t]) + ";\n")
    f.write("\n")

    # Numeric data
    if pN > 0:
        f.write("param X :=\n")
        for i in range(n):
            for j in range(pN):
                f.write(f"{i+1} {j+1} {X_num.iloc[i, j]:.6f}\n")
        f.write(";\n\n")

    # Labels Y (+1 / -1 per class)
    f.write("param Y :=\n")
    for i in range(n):
        for k in range(1, K+1):
            f.write(f"{i+1} {k} {1 if y[i] == k else -1}\n")
    f.write(";\n\n")

    # Categorical V[j] sets and Delta
    if pC > 0:
        # V[j] sets (1..|V_j|)
        for j_idx, c in enumerate(cat_cols, start=1):
            Vj = cat_levels[c]
            if len(Vj) == 0:
                f.write(f"set V[{j_idx}] := ;\n")
            else:
                f.write(f"set V[{j_idx}] := " + " ".join(str(cid) for cid in range(1, len(Vj)+1)) + ";\n")
        f.write("\n")

        # Delta(i,j,c) as param lines: i j c v
        f.write("param Delta :=\n")
        for i in range(n):
            for j_idx, c in enumerate(cat_cols, start=1):
                val = feat_df[c].iloc[i]
                if pd.isna(val):
                    key = "__NA__"
                else:
                    key = str(val)
                cid = cat_level_to_id[c][key]
                Vj = cat_levels[c]
                for ccid in range(1, len(Vj)+1):
                    v = 1 if ccid == cid else 0
                    f.write(f"{i+1} {j_idx} {ccid} {v}\n")
        f.write(";\n\n")

print("Wrote dat file to:", OUTPUT_PATH)
print("Numeric features (JN):", num_cols)
print("Categorical features (JC):", cat_cols)
