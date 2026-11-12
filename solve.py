from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
from univariate_model import uni_model
from multivariate_model import multi_model
from apply_warm_start import apply_warm_start_univariate, apply_warm_start_multivariate
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Defining a small tolerance for Split Constraints
TOL = 1e-5

# Load the same dataset used to create .dat
df = pd.read_csv('Data/New-Datasets-D=2/DS1.csv')

X = df.iloc[:, 1:-1].values  # assumes you skip ID and use only features
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df.iloc[:, -1]) + 1  # match Pyomo 1-based label encoding

# Normalize
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

instance = uni_model.create_instance('Data/New-Datasets-D=2/DS2.dat')

""" Use these constraints for univariate models """

# instance.UniLeftSplit = ConstraintList()
# for i in instance.I:
#     for t in instance.TL:
#         for m in instance.AL[t]:
#             lhs = sum(instance.a[j, m] * instance.X[i, j] for j in instance.J) + TOL
#             rhs = instance.b[m] + (1 + TOL) * (1 - instance.z[i, t])
#             instance.UniLeftSplit.add(lhs <= rhs)
#
# instance.UniRightSplit = ConstraintList()
# for i in instance.I:
#     for t in instance.TL:
#         for m in instance.AR[t]:
#             lhs = sum(instance.a[j, m] * instance.X[i, j] for j in instance.J)
#             rhs = instance.b[m] - (1 - instance.z[i, t])
#             instance.UniRightSplit.add(lhs >= rhs)

instance.UniLeftSplit = ConstraintList()
for i in instance.I:
    for t in instance.TL:
        for m in instance.AL[t]:
            lhs = sum(instance.theta[j, m] * (1 - instance.X[i, j]) for j in instance.J) + sum(instance.a[j, m] * instance.X[i, j] for j in instance.J) - sum(instance.theta[j, m] * instance.X[i, j] for j in instance.J) + TOL
            rhs = instance.b[m] + (1 + TOL) * (1 - instance.z[i, t])
            instance.UniLeftSplit.add(lhs <= rhs)

instance.UniRightSplit = ConstraintList()
for i in instance.I:
    for t in instance.TL:
        for m in instance.AR[t]:
            lhs = sum(instance.theta[j, m] * (1 - instance.X[i, j]) for j in instance.J) + sum(instance.a[j, m] * instance.X[i, j] for j in instance.J) - sum(instance.theta[j, m] * instance.X[i, j] for j in instance.J)
            rhs = instance.b[m] - (1 - instance.z[i, t])
            instance.UniRightSplit.add(lhs >= rhs)

""" Use these constraints for multivariate models """

# instance.MultiLeftSplit = ConstraintList()
# for i in instance.I:
#     for t in instance.TL:
#         for m in instance.AL[t]:
#             lhs = sum((instance.a[j, m] * instance.X[i, j]) + TOL for j in instance.J)
#             rhs = instance.b[m] + (2 + TOL) * (1 - instance.z[i, t])
#             instance.MultiLeftSplit.add(lhs <= rhs)
#
# instance.MultiRightSplit = ConstraintList()
# for i in instance.I:
#     for t in instance.TL:
#         for m in instance.AR[t]:
#             lhs = sum(instance.a[j, m] * instance.X[i, j] for j in instance.J)
#             rhs = instance.b[m] - 2 * (1 - instance.z[i, t])
#             instance.MultiRightSplit.add(lhs >= rhs)

""" Use this function for univariate warm start """
# apply_warm_start_univariate(instance, X, y, depth=instance.DEPTH.value)

""" Use this function for multivariate warm start """
# apply_warm_start_multivariate(instance, X, y, depth=instance.DEPTH.value)

""" Use these cuts for univariate models """
# def feature_cut(instance, t):
#     print(f"{sum(j * instance.a[j, 2*t+1] for j in instance.J) - sum(j * instance.a[j, 2*t] for j in instance.J) >= -len(instance.J)*(2-instance.d[2*t]-instance.d[2*t+1])}")
#     return sum(j * instance.a[j, 2*t+1] for j in instance.J) - sum(j * instance.a[j, 2*t] for j in instance.J) >= -len(instance.J)*(2-instance.d[2*t]-instance.d[2*t+1])
# instance.FeatureCut = Constraint(range(1, 2**(instance.DEPTH-1)-1+1), rule=feature_cut)
#
# def uni_label_cut(instance, t):
#     print(f"{sum(k * instance.c[k, 2*t] for k in instance.K) - sum(k * instance.c[k, 2*t] for k in instance.K) >= -len(instance.K)*(2-instance.l[2*t]-instance.l[2*t+1])}")
#     return sum(k * instance.c[k, 2*t] for k in instance.K) - sum(k * instance.c[k, 2*t] for k in instance.K) >= -len(instance.K)*(2-instance.l[2*t]-instance.l[2*t+1])
# instance.UniLabelCut = Constraint(range(2**(instance.DEPTH-1), 2**(instance.DEPTH)-1+1), rule=uni_label_cut)

""" Use these cuts for multivariate models """
# def label_cut(instance, t):
#     # print(f"{sum(k * instance.c[k, 2*t] for k in instance.K) - sum(k * instance.c[k, 2*t] for k in instance.K) >= -len(instance.K)*(2-instance.l[2*t]-instance.l[2*t+1])}")
#     return sum(k * instance.c[k, 2*t] for k in instance.K) - sum(k * instance.c[k, 2*t] for k in instance.K) >= -len(instance.K)*(2-instance.l[2*t]-instance.l[2*t+1])
# instance.LabelCut = Constraint(range(2**(instance.DEPTH-1), 2**(instance.DEPTH)-1+1), rule=label_cut)

# CPLEX solve
solver = SolverFactory('cplex', executable=r"C:\Program Files\IBM\ILOG\CPLEX_Enterprise_Server1210\CPLEX_Studio\cplex\bin\x64_win64\cplex.exe")
solver.options['timelimit'] = 7200  # Optional: 60 minutes
solver.options['advance'] = 1
solver.options['emphasis_mip'] = 1
solver.options['mip_tolerances_mipgap'] = 0.01
# Use argument "warmstart=True", for initiating warm starts
results = solver.solve(instance, tee=True)
print("Solver's termination condition:", results.solver.termination_condition)

# Check solver status
if (results.solver.status == SolverStatus.ok) and \
        (results.solver.termination_condition == TerminationCondition.optimal or \
         results.solver.termination_condition == TerminationCondition.feasible or \
         results.solver.termination_condition == TerminationCondition.locallyOptimal or \
         results.solver.termination_condition == TerminationCondition.maxTimeLimit):

    print("\n" + "="*30)
    print("SOLUTION RESULTS")
    print("="*30)

    # Objective value
    print(f"\nObjective Value: {value(instance.Objective):.6f}")

    print("\n" + "="*30)
    print("MODEL PARAMETERS")
    print("="*30)
    print(f"Nmin (Minimum points in a leaf): {value(instance.Nmin)}")
    print(f"Lhat (Baseline misclassification): {value(instance.Lhat)}")
    print(f"alpha (Complexity parameter): {value(instance.alpha)}")
    # print(f"epsilon_max: {value(instance.epsilon_max):.6f}")

    print(f"\nSet I (Data points): {len(instance.I)} points")
    print(f"Set J (Features): {len(instance.J)} features")
    print(f"Set K (Classes): {len(instance.K)} classes")
    print(f"Set TB (Branch nodes): {list(instance.TB)}")
    print(f"Set TL (Leaf nodes): {list(instance.TL)}")

    # print("\nParameter epsilon (Smallest non-zero distance for feature j):")
    # for j in instance.J:
    #     print(f"  epsilon[{j}]: {value(instance.epsilon[j]):.6f}")

    # AL and AR are sets of sets, so printing them requires a bit more care
    print("\nParameter AL (Ancestors via left branch for leaf t):")
    for t_leaf in instance.TL:
        if instance.AL[t_leaf]: # Check if the set is not empty
            print(f"  AL[{t_leaf}]: {list(instance.AL[t_leaf])}")
        else:
            print(f"  AL[{t_leaf}]: {{}}")


    print("\nParameter AR (Ancestors via right branch for leaf t):")
    for t_leaf in instance.TL:
        if instance.AR[t_leaf]: # Check if the set is not empty
            print(f"  AR[{t_leaf}]: {list(instance.AR[t_leaf])}")
        else:
            print(f"  AR[{t_leaf}]: {{}}")
    print(f"\nDimensions of X (data features): {len(instance.I)} points x {len(instance.J)} features")
    print(f"Dimensions of Y (data labels): {len(instance.I)} points x {len(instance.K)} classes")


    print("\n" + "="*30)
    print("NON-ZERO DECISION VARIABLES")
    print("="*30)

    print("\nVariable a (Split variable for feature j at branch node t):")
    for t_branch in instance.TB:
        for j_feature in instance.J:
            if value(instance.a[j_feature, t_branch]) > TOL: # Binary, so check if > 0 (or == 1)
                print(f"  a[{j_feature}, {t_branch}]: {value(instance.a[j_feature, t_branch])}")

    print("\nVariable b (Split threshold at branch node t):")
    for t_branch in instance.TB:
        if abs(value(instance.b[t_branch])) > TOL:
            print(f"  b[{t_branch}]: {value(instance.b[t_branch]):.6f}")

    print("\nVariable d (Indicator: branch node t applies a split):")
    for t_branch in instance.TB:
        if value(instance.d[t_branch]) > TOL: # Binary
            print(f"  d[{t_branch}]: {value(instance.d[t_branch])}")

    print("\nVariable z (Indicator: data point i is in leaf node t):")
    for t_leaf in instance.TL:
        points_in_leaf = []
        for i_point in instance.I:
            if value(instance.z[i_point, t_leaf]) > TOL: # Binary
                points_in_leaf.append(i_point)
        if points_in_leaf:
            print(f"  Leaf {t_leaf} contains points: {points_in_leaf} (z values are 1)")


    print("\nVariable l (Indicator: leaf node t contains any points):")
    for t_leaf in instance.TL:
        if value(instance.l[t_leaf]) > TOL: # Binary
            print(f"  l[{t_leaf}]: {value(instance.l[t_leaf])}")

    print("\nVariable c (Indicator: class k is predicted at leaf node t):")
    for t_leaf in instance.TL:
        for k_class in instance.K:
            if value(instance.c[k_class, t_leaf]) > TOL: # Binary
                print(f"  c[{k_class}, {t_leaf}]: {value(instance.c[k_class, t_leaf])}")

    print("\nVariable L (Misclassification loss at leaf node t):")
    for t_leaf in instance.TL:
        if value(instance.L[t_leaf]) > TOL:
            print(f"  L[{t_leaf}]: {value(instance.L[t_leaf]):.6f}")

    print("\nVariable Nkt (Number of points of class k in leaf node t):")
    for t_leaf in instance.TL:
        for k_class in instance.K:
            if value(instance.Nkt[k_class, t_leaf]) > TOL: # Integer
                print(f"  Nkt[{k_class}, {t_leaf}]: {value(instance.Nkt[k_class, t_leaf])}")

    print("\nVariable Nt (Total number of points in leaf node t):")
    for t_leaf in instance.TL:
        if value(instance.Nt[t_leaf]) > TOL: # Integer
            print(f"  Nt[{t_leaf}]: {value(instance.Nt[t_leaf])}")

    print("\n" + "="*30)
    print("END OF SOLUTION RESULTS")
    print("="*30)

elif results.solver.termination_condition == TerminationCondition.infeasible:
    print("Solver Status: Infeasible")
    print("The problem is infeasible. Check your constraints and data.")

else:
    print("Solver Status: Something went wrong.")
    print(f"Status: {results.solver.status}")
    print(f"Termination condition: {results.solver.termination_condition}")


"""New Solve Script"""

# solve.py
# from pyomo.environ import *
# from pyomo.opt import SolverFactory, TerminationCondition
# from Categorical_model import model
#
# TOL = 1e-4
#
# # instance = model.create_instance('Data/New-Datasets-D=2/DS1.dat')
# # instance = model.create_instance('Data/Categorical/Adult_100.dat')
# # instance = model.create_instance('Data/Categorical/Mushrooms.dat')
# instance = model.create_instance('Data/Bank_Authentication.dat')
#
# # ---------- PATH FEASIBILITY CONSTRAINTS ----------
# instance.UniLeftSplit = ConstraintList()
# for i in instance.I:
#     for t in instance.TL:
#         for m in instance.AL[t]:
#             lhs = sum(instance.a[j, m] * instance.X[i, j] for j in instance.JN) + TOL
#             rhs = instance.b[m] + (1 + TOL) * (1 - instance.z[i, t]) + (1 + TOL) * (1 - sum(instance.a[j, m] for j in instance.JN))
#             instance.UniLeftSplit.add(lhs <= rhs)
#
# instance.UniRightSplit = ConstraintList()
# for i in instance.I:
#     for t in instance.TL:
#         for m in instance.AR[t]:
#             lhs = sum(instance.a[j, m] * instance.X[i, j] for j in instance.JN)
#             rhs = instance.b[m] - (1 - instance.z[i, t]) - (1 - sum(instance.a[j, m] for j in instance.JN))
#             instance.UniRightSplit.add(lhs >= rhs)
#
# instance.CatLeftSplit = ConstraintList()
# for i in instance.I:
#     for t in instance.TL:
#         for m in instance.AL[t]:
#             lhs = sum(instance.Delta[i, j, c] * instance.w[j, m, c] for j in instance.JC for c in instance.V[j])
#             rhs = instance.z[i, t] - (1 - sum(instance.u[j, m] for j in instance.JC))
#             instance.CatLeftSplit.add(lhs >= rhs)
#             instance.CatLeftSplit.add(instance.z[i, t] <= instance.d[m])
#
#
# instance.CatRightSplit = ConstraintList()
# for i in instance.I:
#     for t in instance.TL:
#         for m in instance.AR[t]:
#             lhs = sum(instance.Delta[i, j, c] * instance.w[j, m, c] for j in instance.JC for c in instance.V[j])
#             rhs = 1 - instance.z[i, t] + (1 - sum(instance.u[j, m] for j in instance.JC))
#             instance.CatRightSplit.add(lhs <= rhs)
#
# # ---------- SOLVE ----------
# solver = SolverFactory('cplex', executable=r"C:\Program Files\IBM\ILOG\CPLEX_Enterprise_Server1210\CPLEX_Studio\cplex\bin\x64_win64\cplex.exe")
# solver.options['timelimit'] = 3600
# solver.options['advance'] = 1
# solver.options['emphasis_mip'] = 1
# solver.options['mip_tolerances_mipgap'] = 0.01
#
# results = solver.solve(instance, tee=True)
# print("Solver's termination condition:", results.solver.termination_condition)
#
# # ---------- REPORT ----------
# if (results.solver.status == SolverStatus.ok) and \
#         (results.solver.termination_condition in [TerminationCondition.optimal,
#                                                   TerminationCondition.feasible,
#                                                   TerminationCondition.locallyOptimal,
#                                                   TerminationCondition.maxTimeLimit]):
#
#     print("\n" + "="*30)
#     print("SOLUTION RESULTS")
#     print("="*30)
#
#     print(f"\nObjective Value: {value(instance.Objective):.6f}")
#
#     print("\n" + "="*30)
#     print("MODEL PARAMETERS")
#     print("="*30)
#     print(f"Nmin: {value(instance.Nmin)}")
#     print(f"Lhat: {value(instance.Lhat)}")
#     print(f"alpha: {value(instance.alpha)}")
#     print(f"|I|={len(instance.I)}, |JN|={len(instance.JN)}, |JC|={len(instance.JC)}, |K|={len(instance.K)}")
#     print(f"TB={list(instance.TB)}")
#     print(f"TL={list(instance.TL)}")
#
#     print("\nAL sets:")
#     for t_leaf in instance.TL:
#         print(f"  AL[{t_leaf}]: {list(instance.AL[t_leaf]) if instance.AL[t_leaf] else {}}")
#
#     print("\nAR sets:")
#     for t_leaf in instance.TL:
#         print(f"  AR[{t_leaf}]: {list(instance.AR[t_leaf]) if instance.AR[t_leaf] else {}}")
#
#     print("\n" + "="*30)
#     print("NON-ZERO DECISION VARIABLES")
#     print("="*30)
#
#     if len(instance.JN) > 0:
#         print("\nNumeric a[j,t]:")
#         for t_branch in instance.TB:
#             for j in instance.JN:
#                 if value(instance.a[j, t_branch]) > TOL:
#                     print(f"  a[{j},{t_branch}] = {value(instance.a[j, t_branch])}")
#         print("\nb[t]:")
#         for t_branch in instance.TB:
#             if abs(value(instance.b[t_branch])) > TOL:
#                 print(f"  b[{t_branch}] = {value(instance.b[t_branch]):.6f}")
#
#     print("\nVariable d (Indicator: branch node t applies a split):")
#     for t_branch in instance.TB:
#         if value(instance.d[t_branch]) > TOL: # Binary
#             print(f"  d[{t_branch}]: {value(instance.d[t_branch])}")
#
#     if len(instance.JC) > 0:
#         print("\nCategorical u[j,t]:")
#         for t_branch in instance.TB:
#             for j in instance.JC:
#                 if value(instance.u[j, t_branch]) > TOL:
#                     print(f"  u[{j},{t_branch}] = {value(instance.u[j, t_branch])}")
#         print("\nCategorical w[j,t,c] (categories routed left):")
#         for t_branch in instance.TB:
#             for j in instance.JC:
#                 left_cats = [int(c) for c in instance.V[j] if value(instance.w[j, t_branch, c]) > TOL]
#                 if left_cats:
#                     print(f"  node {t_branch}, feature {j} -> left cats {left_cats}")
#
#     print("\nz[i,t]=1 assignments:")
#     for t_leaf in instance.TL:
#         pts = [i for i in instance.I if value(instance.z[i, t_leaf]) > TOL]
#         if pts:
#             print(f"  Leaf {t_leaf}: {pts}")
#
#     print("\nl[t]:")
#     for t_leaf in instance.TL:
#         if value(instance.l[t_leaf]) > TOL:
#             print(f"  l[{t_leaf}] = 1")
#
#     print("\nc[k,t]:")
#     for t_leaf in instance.TL:
#         for k in instance.K:
#             if value(instance.c[k, t_leaf]) > TOL:
#                 print(f"  c[{k},{t_leaf}] = 1")
#
#     print("\nL[t]:")
#     for t_leaf in instance.TL:
#         if value(instance.L[t_leaf]) > TOL:
#             print(f"  L[{t_leaf}] = {value(instance.L[t_leaf]):.6f}")
#
#     print("\nNkt[k,t]:")
#     for t_leaf in instance.TL:
#         for k in instance.K:
#             if value(instance.Nkt[k, t_leaf]) > TOL:
#                 print(f"  Nkt[{k},{t_leaf}] = {int(value(instance.Nkt[k, t_leaf]))}")
#
#     print("\nNt[t]:")
#     for t_leaf in instance.TL:
#         if value(instance.Nt[t_leaf]) > TOL:
#             print(f"  Nt[{t_leaf}] = {int(value(instance.Nt[t_leaf]))}")
#
#     print("\n" + "="*30 + "\n")
#
#     # ---------- ACCURACY ----------
#     total_points = len(instance.I)
#     total_loss = sum(value(instance.L[t]) for t in instance.TL)
#     accuracy = (total_points - total_loss) / total_points
#     print("\n" + "="*30)
#     print(f"MODEL ACCURACY: {accuracy:.4f}")
#     print("="*30 + "\n")
#
# elif results.solver.termination_condition == TerminationCondition.infeasible:
#     print("Solver Status: Infeasible")
# else:
#     print("Solver Status: Something went wrong.")
#     print(f"Status: {results.solver.status}")
#     print(f"Termination condition: {results.solver.termination_condition}")