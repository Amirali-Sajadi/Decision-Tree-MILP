from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import numpy as np
import warnings

def apply_warm_start_univariate(instance, X, y, depth):
    """
    Apply warm start from CART to Pyomo instance.
    Assumes instance is already created via create_instance().
    """

    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=instance.Nmin(), random_state=0)
    clf.fit(X, y)

    # Visualize the CART tree
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=[f"x{j+1}" for j in range(X.shape[1])],
        class_names=[str(k) for k in clf.classes_],
        filled=True,
        rounded=True,
        special_characters=True
    )

    graph = graphviz.Source(dot_data)
    graph.render("cart_warm_start", format="pdf", cleanup=False)  # produces cart_warm_start.pdf

    tree = clf.tree_

    node_to_model_branch = {}   # map sklearn node id -> model branch node id
    leaf_to_model_leaf = {}     # map sklearn leaf id -> model leaf node id

    def find_terminal_leaf(start_node, leaf_set):
        """Follows the right-child path to find the terminal leaf in the maximal tree."""
        current_node = start_node
        while current_node not in leaf_set:
            current_node = 2 * current_node + 1
        return current_node

    def walk_tree(node_id, pyomo_id):
        if tree.children_left[node_id] == tree.children_right[node_id]:
            if pyomo_id in instance.TL:
                leaf_to_model_leaf[node_id] = pyomo_id
            elif pyomo_id in instance.TB:
                terminal_leaf = find_terminal_leaf(pyomo_id, instance.TL)
                leaf_to_model_leaf[node_id] = terminal_leaf
            return

        if pyomo_id in instance.TB:
            node_to_model_branch[node_id] = pyomo_id

        walk_tree(tree.children_left[node_id], 2 * pyomo_id)
        walk_tree(tree.children_right[node_id], 2 * pyomo_id + 1)

    walk_tree(0, 1)

    # 1. Set a[j, t] and d[t]
    for node_id, t in node_to_model_branch.items():
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        if t in instance.TB and (feature + 1) in instance.J:
            instance.a[feature + 1, t].set_value(1)
            instance.d[t].set_value(1)
            instance.b[t].set_value(threshold)
        else:
            warnings.warn(f"Node {t} not in TB or feature out of bounds.")

    # 2. Set c[k, t] and l[t] for leaves
    classes = clf.classes_
    for node_id, t in leaf_to_model_leaf.items():
        if t not in instance.TL:
            continue
        counts = tree.value[node_id][0]  # counts for each class
        predicted_class = classes[np.argmax(counts)]
        instance.l[t].set_value(1)
        instance.c[predicted_class, t].set_value(1)

    # 3. Set z[i, t] for sample-to-leaf assignment
    leaf_ids = clf.apply(X)  # sklearn leaf index for each point
    for i, node_id in enumerate(leaf_ids):
        if node_id not in leaf_to_model_leaf:
            continue
        t = leaf_to_model_leaf[node_id]
        if (i + 1) in instance.I and t in instance.TL:
            instance.z[i + 1, t].set_value(1)
        else:
            print(f"Skipping invalid index: i={i+1}, t={t}")

    for t in instance.TL:
        if instance.l[t].value != 1:
            continue  # skip inactive leaves

        assigned_points = [i for i in instance.I if instance.z[i, t].value == 1]

        if not assigned_points:
            # No data assigned to this leaf — set defaults to 0
            instance.Nt[t].set_value(0)
            for k in instance.K:
                instance.Nkt[k, t].set_value(0)
            instance.L[t].set_value(0)
            continue  # ← this is valid here, inside the for loop over t

        assigned_labels = [y[i - 1] for i in assigned_points]  # y is 0-indexed

        # Nt[t]
        instance.Nt[t].set_value(len(assigned_points))

        # Nkt[k, t]
        for k in instance.K:
            class_count = sum(1 for label in assigned_labels if label == k)
            instance.Nkt[k, t].set_value(class_count)

        # L[t]
        majority_class = max(instance.K, key=lambda k: instance.Nkt[k, t].value)
        L_value = instance.Nt[t].value - instance.Nkt[majority_class, t].value
        instance.L[t].set_value(L_value)

    # 1. a[j, t]
    for j in instance.J:
        for t in instance.TB:
            if instance.a[j, t].value is None:
                instance.a[j, t].set_value(0)

    # 2. b[t] and d[t]
    for t in instance.TB:
        if instance.b[t].value is None:
            instance.b[t].set_value(0)
        if instance.d[t].value is None:
            instance.d[t].set_value(0)

    # 3. z[i, t]
    for i in instance.I:
        for t in instance.TL:
            if instance.z[i, t].value is None:
                instance.z[i, t].set_value(0)

    # 4. l[t] and c[k, t]
    for t in instance.TL:
        if instance.l[t].value is None:
            instance.l[t].set_value(0)
        for k in instance.K:
            if instance.c[k, t].value is None:
                instance.c[k, t].set_value(0)

    # 5. L[t], Nt[t], Nkt[k, t]
    for t in instance.TL:
        if instance.L[t].value is None:
            instance.L[t].set_value(0)
        if instance.Nt[t].value is None:
            instance.Nt[t].set_value(0)
        for k in instance.K:
            if instance.Nkt[k, t].value is None:
                instance.Nkt[k, t].set_value(0)

def apply_warm_start_multivariate(instance, X, y, depth):
    """
    Apply warm start from CART to Pyomo instance.
    Assumes instance is already created via create_instance().
    """

    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=instance.Nmin(), random_state=0)
    clf.fit(X, y)

    # Visualize the CART tree
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=[f"x{j+1}" for j in range(X.shape[1])],
        class_names=[str(k) for k in clf.classes_],
        filled=True,
        rounded=True,
        special_characters=True
    )

    graph = graphviz.Source(dot_data)
    graph.render("cart_warm_start", format="pdf", cleanup=False)  # produces cart_warm_start.pdf

    tree = clf.tree_

    node_to_model_branch = {}   # map sklearn node id -> model branch node id
    leaf_to_model_leaf = {}     # map sklearn leaf id -> model leaf node id

    # Mapping branch splits
    leaf_model_ids = list(instance.TL)
    leaf_index = 1

    def find_terminal_leaf(start_node, leaf_set):
        """Follows the right-child path to find the terminal leaf in the maximal tree."""
        current_node = start_node
        while current_node not in leaf_set:
            current_node = 2 * current_node + 1
        return current_node

    # --- UPDATED WALK_TREE FUNCTION ---
    def walk_tree(node_id, pyomo_id):
        if tree.children_left[node_id] == tree.children_right[node_id]:
            if pyomo_id in instance.TL:
                leaf_to_model_leaf[node_id] = pyomo_id
            elif pyomo_id in instance.TB:
                terminal_leaf = find_terminal_leaf(pyomo_id, instance.TL)
                leaf_to_model_leaf[node_id] = terminal_leaf
            return

        if pyomo_id in instance.TB:
            node_to_model_branch[node_id] = pyomo_id

        walk_tree(tree.children_left[node_id], 2 * pyomo_id)
        walk_tree(tree.children_right[node_id], 2 * pyomo_id + 1)

    walk_tree(0, 1)

    # 1. Set a[j, t] and d[t]
    for node_id, t in node_to_model_branch.items():
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        if t in instance.TB and (feature + 1) in instance.J:
            instance.a[feature + 1, t].set_value(1)
            instance.aprime[feature + 1, t].set_value(1)
            instance.s[feature + 1, t].set_value(1)
            instance.d[t].set_value(1)
            instance.b[t].set_value(threshold)
        else:
            warnings.warn(f"Node {t} not in TB or feature out of bounds.")

    # 2. Set c[k, t] and l[t] for leaves
    classes = clf.classes_
    for node_id, t in leaf_to_model_leaf.items():
        if t not in instance.TL:
            continue
        counts = tree.value[node_id][0]  # counts for each class
        predicted_class = classes[np.argmax(counts)]
        instance.l[t].set_value(1)
        instance.c[predicted_class, t].set_value(1)

    # 3. Set z[i, t] for sample-to-leaf assignment
    leaf_ids = clf.apply(X)  # sklearn leaf index for each point
    for i, node_id in enumerate(leaf_ids):
        if node_id not in leaf_to_model_leaf:
            continue
        t = leaf_to_model_leaf[node_id]
        if (i + 1) in instance.I and t in instance.TL:
            instance.z[i + 1, t].set_value(1)
        else:
            print(f"Skipping invalid index: i={i+1}, t={t}")

    for t in instance.TL:
        if instance.l[t].value != 1:
            continue  # skip inactive leaves

        assigned_points = [i for i in instance.I if instance.z[i, t].value == 1]

        if not assigned_points:
            # No data assigned to this leaf — set defaults to 0
            instance.Nt[t].set_value(0)
            for k in instance.K:
                instance.Nkt[k, t].set_value(0)
            instance.L[t].set_value(0)
            continue  # ← this is valid here, inside the for loop over t

        assigned_labels = [y[i - 1] for i in assigned_points]  # y is 0-indexed

        # Nt[t]
        instance.Nt[t].set_value(len(assigned_points))

        # Nkt[k, t]
        for k in instance.K:
            class_count = sum(1 for label in assigned_labels if label == k)
            instance.Nkt[k, t].set_value(class_count)

        # L[t]
        majority_class = max(instance.K, key=lambda k: instance.Nkt[k, t].value)
        L_value = instance.Nt[t].value - instance.Nkt[majority_class, t].value
        instance.L[t].set_value(L_value)

    # 1. a[j, t]
    for j in instance.J:
        for t in instance.TB:
            if instance.a[j, t].value is None:
                instance.a[j, t].set_value(0)
                instance.aprime[j, t].set_value(0)
                instance.s[j, t].set_value(0)

    # 2. b[t] and d[t]
    for t in instance.TB:
        if instance.b[t].value is None:
            instance.b[t].set_value(0)
        if instance.d[t].value is None:
            instance.d[t].set_value(0)

    # 3. z[i, t]
    for i in instance.I:
        for t in instance.TL:
            if instance.z[i, t].value is None:
                instance.z[i, t].set_value(0)

    # 4. l[t] and c[k, t]
    for t in instance.TL:
        if instance.l[t].value is None:
            instance.l[t].set_value(0)
        for k in instance.K:
            if instance.c[k, t].value is None:
                instance.c[k, t].set_value(0)

    # 5. L[t], Nt[t], Nkt[k, t]
    for t in instance.TL:
        if instance.L[t].value is None:
            instance.L[t].set_value(0)
        if instance.Nt[t].value is None:
            instance.Nt[t].set_value(0)
        for k in instance.K:
            if instance.Nkt[k, t].value is None:
                instance.Nkt[k, t].set_value(0)