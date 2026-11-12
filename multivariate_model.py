import pyomo.environ as pyo

multi_model = pyo.AbstractModel()

# Sets
multi_model.I = pyo.Set(doc='Data points')
multi_model.J = pyo.Set(doc='Features')
multi_model.K = pyo.Set(doc='Classes')
multi_model.TB = pyo.Set(doc='Branch nodes')
multi_model.TL = pyo.Set(doc='Leaf nodes')
multi_model.AL = pyo.Set(multi_model.TL, within=multi_model.TB)
multi_model.AR = pyo.Set(multi_model.TL, within=multi_model.TB)

# Parameters
multi_model.X = pyo.Param(multi_model.I, multi_model.J, within=pyo.UnitInterval)
multi_model.Y = pyo.Param(multi_model.I, multi_model.K, within=pyo.Integers)  # +1/-1 labels
multi_model.DEPTH = pyo.Param(within=pyo.PositiveIntegers)
multi_model.Nmin = pyo.Param(within=pyo.PositiveIntegers)
multi_model.Lhat = pyo.Param(within=pyo.PositiveReals)
multi_model.alpha = pyo.Param(within=pyo.NonNegativeReals)

# Variables
multi_model.a = pyo.Var(multi_model.J, multi_model.TB, domain=pyo.Reals)
multi_model.aprime = pyo.Var(multi_model.J, multi_model.TB, domain=pyo.NonNegativeReals)
multi_model.s = pyo.Var(multi_model.J, multi_model.TB, domain=pyo.Binary)
multi_model.b = pyo.Var(multi_model.TB, domain=pyo.NonNegativeReals)
multi_model.d = pyo.Var(multi_model.TB, domain=pyo.Binary)
multi_model.z = pyo.Var(multi_model.I, multi_model.TL, domain=pyo.Binary)
multi_model.l = pyo.Var(multi_model.TL, domain=pyo.Binary)
multi_model.c = pyo.Var(multi_model.K, multi_model.TL, domain=pyo.Binary)
multi_model.L = pyo.Var(multi_model.TL, domain=pyo.NonNegativeReals)

# Intermediate calculations
multi_model.Nkt = pyo.Var(multi_model.K, multi_model.TL, domain=pyo.NonNegativeIntegers)
multi_model.Nt = pyo.Var(multi_model.TL, domain=pyo.NonNegativeIntegers)

# Constraints
def multivar_split_rule(model, t):
    return sum(model.aprime[j, t] for j in model.J) <= model.d[t]
multi_model.MultivariateSplit = pyo.Constraint(multi_model.TB, rule=multivar_split_rule)

def aprime1_bounds_rule(model, j, t):
    return model.aprime[j, t] >= model.a[j, t]
multi_model.Aprime1Bounds = pyo.Constraint(multi_model.J, multi_model.TB, rule=aprime1_bounds_rule)

def aprime2_bounds_rule(model, j, t):
    return model.aprime[j, t] >= -model.a[j, t]
multi_model.Aprime2Bounds = pyo.Constraint(multi_model.J, multi_model.TB, rule=aprime2_bounds_rule)

def a1_bounds_rule(model, j, t):
    return model.a[j, t] <= model.s[j, t]
multi_model.A1Bounds = pyo.Constraint(multi_model.J, multi_model.TB, rule=a1_bounds_rule)

def a2_bounds_rule(model, j, t):
    return model.a[j, t] >= -model.s[j, t]
multi_model.A2Bounds = pyo.Constraint(multi_model.J, multi_model.TB, rule=a2_bounds_rule)

def b1_bounds_rule(model, t):
    return model.b[t] <= model.d[t]
multi_model.B1Bounds = pyo.Constraint(multi_model.TB, rule=b1_bounds_rule)

def b2_bounds_rule(model, t):
    return model.b[t] >= -model.d[t]
multi_model.B2Bounds = pyo.Constraint(multi_model.TB, rule=b2_bounds_rule)

def split1_s_rule(model, j, t):
    return model.s[j, t] <= model.d[t]
multi_model.Split1S = pyo.Constraint(multi_model.J, multi_model.TB, rule=split1_s_rule)

def split2_s_rule(model, t):
    return sum(model.s[j, t] for j in model.J) >= model.d[t]
multi_model.Split2S = pyo.Constraint(multi_model.TB, rule=split2_s_rule)


# Hierarchy (skip root node)
def hierarchy_rule(model, t):
    if t == min(model.TB):
        return pyo.Constraint.Skip
    parent = t // 2
    return model.d[t] <= model.d[parent]
multi_model.Hierarchy = pyo.Constraint(multi_model.TB, rule=hierarchy_rule)

# Point assignment
def z_bounds_rule(model, i, t):
    return model.z[i, t] <= model.l[t]
multi_model.ZBounds = pyo.Constraint(multi_model.I, multi_model.TL, rule=z_bounds_rule)

def zit_rule(model, t):
    return sum(model.z[i, t] for i in model.I) >= model.Nmin * model.l[t]
multi_model.Zit = pyo.Constraint(multi_model.TL, rule=zit_rule)

def OneLeafPerPoint_rule(model, i):
    return sum(model.z[i, t] for t in model.TL) == 1
multi_model.OneLeafPerPoint = pyo.Constraint(multi_model.I, rule=OneLeafPerPoint_rule)

# Nkt and Nt computation
def nkt_rule(model, k, t):
    return model.Nkt[k, t] == 0.5 * sum((1 + model.Y[i, k]) * model.z[i, t] for i in model.I)
multi_model.NktCalc = pyo.Constraint(multi_model.K, multi_model.TL, rule=nkt_rule)

def nt_rule(model, t):
    return model.Nt[t] == sum(model.z[i, t] for i in model.I)
multi_model.NtCalc = pyo.Constraint(multi_model.TL, rule=nt_rule)

# Misclassification loss linearization
def OneClassPerLeaf_rule(model, t):
    return sum(model.c[k, t] for k in model.K) == model.l[t]
multi_model.OneClassPerLeaf = pyo.Constraint(multi_model.TL, rule=OneClassPerLeaf_rule)

def lt_lower(model, k, t):
    return model.L[t] >= model.Nt[t] - model.Nkt[k, t] - len(model.I)*(1 - model.c[k, t])
multi_model.LtLower = pyo.Constraint(multi_model.K, multi_model.TL, rule=lt_lower)

# Objective function
def obj_rule(model):
    return (1 / model.Lhat) * sum(model.L[t] for t in model.TL) + model.alpha * sum(model.s[j, t] for j in model.J for t in model.TB)

multi_model.Objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)