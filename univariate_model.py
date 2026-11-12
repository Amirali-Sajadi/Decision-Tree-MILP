import pyomo.environ as pyo

uni_model = pyo.AbstractModel()

# Sets
uni_model.I = pyo.Set(doc='Data points')
uni_model.J = pyo.Set(doc='Features')
uni_model.K = pyo.Set(doc='Classes')
uni_model.TB = pyo.Set(doc='Branch nodes')
uni_model.TL = pyo.Set(doc='Leaf nodes')
uni_model.AL = pyo.Set(uni_model.TL, within=uni_model.TB)
uni_model.AR = pyo.Set(uni_model.TL, within=uni_model.TB)

# Parameters
uni_model.X = pyo.Param(uni_model.I, uni_model.J, within=pyo.UnitInterval)
uni_model.Y = pyo.Param(uni_model.I, uni_model.K, within=pyo.Integers)  # +1/-1 labels
uni_model.DEPTH = pyo.Param(within=pyo.PositiveIntegers)
uni_model.Nmin = pyo.Param(within=pyo.PositiveIntegers)
uni_model.Lhat = pyo.Param(within=pyo.PositiveReals)
uni_model.alpha = pyo.Param(within=pyo.NonNegativeReals)
uni_model.epsilon = pyo.Param(uni_model.J, within=pyo.PositiveReals)
uni_model.epsilon_max = pyo.Param(within=pyo.PositiveReals)

# Variables
uni_model.a = pyo.Var(uni_model.J, uni_model.TB, domain=pyo.Binary)
uni_model.b = pyo.Var(uni_model.TB, domain=pyo.NonNegativeReals)
uni_model.d = pyo.Var(uni_model.TB, domain=pyo.Binary)
uni_model.gamma = pyo.Var(uni_model.TB, domain=pyo.Binary)
uni_model.theta = pyo.Var(uni_model.J, uni_model.TB, domain=pyo.Binary)
uni_model.z = pyo.Var(uni_model.I, uni_model.TL, domain=pyo.Binary)
uni_model.l = pyo.Var(uni_model.TL, domain=pyo.Binary)
uni_model.c = pyo.Var(uni_model.K, uni_model.TL, domain=pyo.Binary)
uni_model.L = pyo.Var(uni_model.TL, domain=pyo.NonNegativeReals)

# Intermediate calculations
uni_model.Nkt = pyo.Var(uni_model.K, uni_model.TL, domain=pyo.NonNegativeIntegers)
uni_model.Nt = pyo.Var(uni_model.TL, domain=pyo.NonNegativeIntegers)

# Constraints
def univar_split_rule(model, t):
    return sum(model.a[j, t] for j in model.J) == model.d[t]
uni_model.UnivariateSplit = pyo.Constraint(uni_model.TB, rule=univar_split_rule) #

def b_bounds_rule(model, t):
    return model.b[t] <= model.d[t]
uni_model.BBounds = pyo.Constraint(uni_model.TB, rule=b_bounds_rule)

def theta_linearization1_rule(model, j, t):
    return 2 * model.theta[j, t] <= model.gamma[t] + model.a[j, t]
uni_model.theta1 = pyo.Constraint(uni_model.J, uni_model.TB, rule=theta_linearization1_rule)

def theta_linearization2_rule(model, j, t):
    return model.gamma[t] + model.a[j, t] <= model.theta[j, t] + 1
uni_model.theta2 = pyo.Constraint(uni_model.J, uni_model.TB, rule=theta_linearization2_rule)

# Hierarchy (skip root node)
def hierarchy_rule(model, t):
    if t == min(model.TB):
        return pyo.Constraint.Skip
    parent = t // 2
    return model.d[t] <= model.d[parent]
uni_model.Hierarchy = pyo.Constraint(uni_model.TB, rule=hierarchy_rule)

# Point assignment
def z_bounds_rule(model, i, t):
    return model.z[i, t] <= model.l[t]
uni_model.ZBounds = pyo.Constraint(uni_model.I, uni_model.TL, rule=z_bounds_rule)

def zit_rule(model, t):
    return sum(model.z[i, t] for i in model.I) >= model.Nmin * model.l[t]
uni_model.Zit = pyo.Constraint(uni_model.TL, rule=zit_rule)

def OneLeafPerPoint_rule(model, i):
    return sum(model.z[i, t] for t in model.TL) == 1
uni_model.OneLeafPerPoint = pyo.Constraint(uni_model.I, rule=OneLeafPerPoint_rule) #

# Nkt and Nt computation
def nkt_rule(model, k, t):
    return model.Nkt[k, t] == 0.5 * sum((1 + model.Y[i, k]) * model.z[i, t] for i in model.I)
uni_model.NktCalc = pyo.Constraint(uni_model.K, uni_model.TL, rule=nkt_rule)

def nt_rule(model, t):
    return model.Nt[t] == sum(model.z[i, t] for i in model.I)
uni_model.NtCalc = pyo.Constraint(uni_model.TL, rule=nt_rule)

# Misclassification loss linearization
def OneClassPerLeaf_rule(model, t):
    return sum(model.c[k, t] for k in model.K) == model.l[t]
uni_model.OneClassPerLeaf = pyo.Constraint(uni_model.TL, rule=OneClassPerLeaf_rule)

def lt_lower(model, k, t):
    return model.L[t] >= model.Nt[t] - model.Nkt[k, t] - len(model.I)*(1 - model.c[k, t])
uni_model.LtLower = pyo.Constraint(uni_model.K, uni_model.TL, rule=lt_lower)

# Objective function
def obj_rule(model):
    return (1 / model.Lhat) * sum(model.L[t] for t in model.TL) + model.alpha * sum(model.d[t] for t in model.TB)

uni_model.Objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)