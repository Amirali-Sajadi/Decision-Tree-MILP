# mixed_model.py
import pyomo.environ as pyo

model = pyo.AbstractModel()

# ----- Sets -----
model.I  = pyo.Set(doc='Data points')
model.JN = pyo.Set(doc='Numeric features')        # may be empty
model.JC = pyo.Set(doc='Categorical features')    # may be empty
model.K  = pyo.Set(doc='Classes')
model.TB = pyo.Set(doc='Branch nodes')
model.TL = pyo.Set(doc='Leaf nodes')
model.AL = pyo.Set(model.TL, within=model.TB)     # left ancestors
model.AR = pyo.Set(model.TL, within=model.TB)     # right ancestors

# Variable-size category sets: V[j] for each categorical feature j
model.V = pyo.Set(model.JC)

# ----- Params -----
# Numeric features (UnitInterval) only if |JN|>0
model.X = pyo.Param(model.I, model.JN, within=pyo.UnitInterval, default=0)

# Labels (+1/-1) as before
model.Y = pyo.Param(model.I, model.K, within=pyo.Integers)

model.DEPTH = pyo.Param(within=pyo.PositiveIntegers)
model.Nmin  = pyo.Param(within=pyo.PositiveIntegers)
model.Lhat  = pyo.Param(within=pyo.PositiveReals)
model.alpha = pyo.Param(within=pyo.NonNegativeReals)

# Define a cross-product index set for Delta
def delta_index_init(m):
    return ((i, j, c) for i in m.I for j in m.JC for c in m.V[j])
model.Delta_index = pyo.Set(dimen=3, initialize=delta_index_init)

model.Delta = pyo.Param(model.Delta_index, within=pyo.Binary, default=0)

# ----- Vars -----
# Numeric split selection and threshold
model.a = pyo.Var(model.JN, model.TB, domain=pyo.Binary)
model.b = pyo.Var(model.TB, domain=pyo.NonNegativeReals)

# Categorical split selection and subset-left markers
model.u = pyo.Var(model.JC, model.TB, domain=pyo.Binary)                     # choose categorical feature j at node t

# Index set for (j,t,c) triples
def w_index_init(m):
    return ((j, t, c) for j in m.JC for t in m.TB for c in m.V[j])
model.W_index = pyo.Set(dimen=3, initialize=w_index_init)

# w[j,t,c] = 1 if category c of feature j is sent left at node t
model.w = pyo.Var(model.W_index, domain=pyo.Binary)

# Node/leaf, assignment, loss (same as OCT)
model.d = pyo.Var(model.TB, domain=pyo.Binary)
model.z = pyo.Var(model.I, model.TL, domain=pyo.Binary)
model.l = pyo.Var(model.TL, domain=pyo.Binary)
model.c = pyo.Var(model.K, model.TL, domain=pyo.Binary)
model.L = pyo.Var(model.TL, domain=pyo.NonNegativeReals)

# Intermediate counts
model.Nkt = pyo.Var(model.K, model.TL, domain=pyo.NonNegativeIntegers)
model.Nt  = pyo.Var(model.TL, domain=pyo.NonNegativeIntegers)

# ----- Constraints -----

# Univariate split per branching node: either one numeric feature or one categorical feature
def univariate_split_rule(m, t):
    return (sum(m.a[j, t] for j in m.JN) + sum(m.u[j, t] for j in m.JC)) == m.d[t]
model.UnivariateSplit = pyo.Constraint(model.TB, rule=univariate_split_rule)

# Threshold b applies only to numeric; force b=0 if no numeric feature selected
def b_numeric_only_rule(m, t):
    return sum(m.u[j, t] for j in m.JC) <= 1 - m.b[t]
model.BNumericOnly = pyo.Constraint(model.TB, rule=b_numeric_only_rule)

def b_bounds_rule(m, t):
    return m.b[t] <= m.d[t]
model.BBounds = pyo.Constraint(model.TB, rule=b_bounds_rule)

# Hierarchy (skip root)
def hierarchy_rule(m, t):
    if t == min(m.TB):
        return pyo.Constraint.Skip
    parent = t // 2
    return m.d[t] <= m.d[parent]
model.Hierarchy = pyo.Constraint(model.TB, rule=hierarchy_rule)

# Point-to-leaf assignment and min leaf size
def z_bounds_rule(m, i, t):
    return m.z[i, t] <= m.l[t]
model.ZBounds = pyo.Constraint(model.I, model.TL, rule=z_bounds_rule)

def leaf_min_rule(m, t):
    return sum(m.z[i, t] for i in m.I) >= m.Nmin * m.l[t]
model.LeafMin = pyo.Constraint(model.TL, rule=leaf_min_rule)

def one_leaf_per_point_rule(m, i):
    return sum(m.z[i, t] for t in m.TL) == 1
model.OneLeafPerPoint = pyo.Constraint(model.I, rule=one_leaf_per_point_rule)

# Nkt, Nt
def nkt_rule(m, k, t):
    return m.Nkt[k, t] == 0.5 * sum((1 + m.Y[i, k]) * m.z[i, t] for i in m.I)
model.NktCalc = pyo.Constraint(model.K, model.TL, rule=nkt_rule)

def nt_rule(m, t):
    return m.Nt[t] == sum(m.z[i, t] for i in m.I)
model.NtCalc = pyo.Constraint(model.TL, rule=nt_rule)

# One class per leaf and misclassification linearization
def one_class_per_leaf_rule(m, t):
    return sum(m.c[k, t] for k in m.K) == m.l[t]
model.OneClassPerLeaf = pyo.Constraint(model.TL, rule=one_class_per_leaf_rule)

def lt_lower_rule(m, k, t):
    return m.L[t] >= m.Nt[t] - m.Nkt[k, t] - len(m.I) * (1 - m.c[k, t])
model.LtLower = pyo.Constraint(model.K, model.TL, rule=lt_lower_rule)

# Categorical subset non-triviality (if u[j,t]=1 then at least 1 left and at least 1 right)
def cat_subset_lb_rule(m, j, t):
    # sum_c w[j,t,c] >= u[j,t]
    return sum(m.w[j, t, c] for c in m.V[j]) >= m.u[j, t]
model.CatSubsetLB = pyo.Constraint(model.JC, model.TB, rule=cat_subset_lb_rule)

def cat_subset_ub_rule(m, j, t):
    # sum_c w[j,t,c] <= (|V_j|-1) * u[j,t]
    return sum(m.w[j, t, c] for c in m.V[j]) <= (len(m.V[j]) - 1) * m.u[j, t]
model.CatSubsetUB = pyo.Constraint(model.JC, model.TB, rule=cat_subset_ub_rule)

# w can be 1 only if its feature is chosen at that node
# def w_implies_u_rule(m, j, t, c):
#     return m.w[j, t, c] <= m.u[j, t]
# model.WImpliesU = pyo.Constraint(model.W_index, rule=w_implies_u_rule)

# def leaf_count_rule(m):
#     return sum(m.l[t] for t in m.TL) == 2 + sum(m.d[t] for t in m.TB)
# model.LeafCount = pyo.Constraint(rule=leaf_count_rule)

# Objective: same as OCT
def obj_rule(m):
    return (1 / m.Lhat) * sum(m.L[t] for t in m.TL) + m.alpha * sum(m.d[t] for t in m.TB)
model.Objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
