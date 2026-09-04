"""
Task 5: Decision Trees and Random Forests
AI & ML Internship - Elevate Labs

Objective: Learn tree-based models for classification & regression.
Dataset: Heart Disease dataset (data/heart.csv)

Steps covered:
1. Train a Decision Tree Classifier and visualize the tree
2. Analyze overfitting and control tree depth
3. Train a Random Forest and compare accuracy
4. Interpret feature importances
5. Evaluate using cross-validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
df = pd.read_csv("data/heart.csv")
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Target balance:\n", df["target"].value_counts())

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# STEP 1: Train a Decision Tree Classifier and visualize the tree
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 1: DECISION TREE - TRAIN & VISUALIZE")
print("=" * 60)

# First, a shallow tree (depth=3) purely so the visualization is readable
dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_shallow.fit(X_train, y_train)
train_acc_shallow = accuracy_score(y_train, dt_shallow.predict(X_train))
test_acc_shallow = accuracy_score(y_test, dt_shallow.predict(X_test))
print(f"\nShallow tree (max_depth=3) - Train acc: {train_acc_shallow:.4f}, "
      f"Test acc: {test_acc_shallow:.4f}")

# Visualize with matplotlib (always works, no external dependency)
plt.figure(figsize=(20, 10))
plot_tree(
    dt_shallow,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title("Decision Tree (max_depth=3) - Matplotlib Visualization")
plt.tight_layout()
plt.savefig("images/decision_tree_matplotlib.png", dpi=120)
plt.close()
print("Saved images/decision_tree_matplotlib.png")

# Also visualize with Graphviz (nicer layout, as suggested in the task tools)
dot_data = export_graphviz(
    dt_shallow,
    out_file=None,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render("images/decision_tree_graphviz", format="png", cleanup=True)
print("Saved images/decision_tree_graphviz.png")

# ---------------------------------------------------------------
# STEP 2: Analyze overfitting and control tree depth
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: OVERFITTING ANALYSIS - TREE DEPTH")
print("=" * 60)

depths = range(1, 21)
train_scores, test_scores = [], []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
    test_scores.append(accuracy_score(y_test, dt.predict(X_test)))

results = pd.DataFrame({"max_depth": list(depths), "train_acc": train_scores, "test_acc": test_scores})
print("\n", results.to_string(index=False))

best_depth = results.loc[results["test_acc"].idxmax(), "max_depth"]
print(f"\nBest test accuracy at max_depth = {best_depth}")

plt.figure(figsize=(9, 6))
plt.plot(depths, train_scores, marker="o", label="Train accuracy")
plt.plot(depths, test_scores, marker="o", label="Test accuracy")
plt.axvline(best_depth, color="gray", linestyle="--", label=f"Best depth = {best_depth}")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Train vs Test Accuracy by max_depth (Overfitting Analysis)")
plt.legend()
plt.tight_layout()
plt.savefig("images/overfitting_vs_depth.png", dpi=120)
plt.close()
print("Saved images/overfitting_vs_depth.png")

print(
    "\nInterpretation: as max_depth grows, train accuracy climbs toward 1.0 (the tree "
    "memorizes the training data) while test accuracy peaks then flattens/declines - "
    "the growing gap between the two curves is the signature of overfitting. Depth "
    f"{best_depth} gives the best generalization here."
)

# Fit the tuned tree for later use
dt_tuned = DecisionTreeClassifier(max_depth=int(best_depth), random_state=42)
dt_tuned.fit(X_train, y_train)
dt_test_acc = accuracy_score(y_test, dt_tuned.predict(X_test))

# ---------------------------------------------------------------
# STEP 3: Train a Random Forest and compare accuracy
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: RANDOM FOREST - TRAIN & COMPARE")
print("=" * 60)

rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_test_acc = accuracy_score(y_test, rf_pred)

print(f"\nTuned Decision Tree (max_depth={best_depth}) test accuracy: {dt_test_acc:.4f}")
print(f"Random Forest (200 trees) test accuracy:              {rf_test_acc:.4f}")

print("\nRandom Forest confusion matrix:\n", confusion_matrix(y_test, rf_pred))
print("\nRandom Forest classification report:\n",
      classification_report(y_test, rf_pred, target_names=["No Disease", "Disease"]))

comparison = pd.DataFrame({
    "Model": [f"Decision Tree (depth={best_depth})", "Random Forest (200 trees)"],
    "Test Accuracy": [dt_test_acc, rf_test_acc],
})
comparison.to_csv("model_comparison.csv", index=False)

plt.figure(figsize=(6, 5))
sns.barplot(data=comparison, x="Model", y="Test Accuracy", hue="Model", legend=False, palette="Set2")
plt.ylim(0, 1)
plt.title("Test Accuracy: Decision Tree vs Random Forest")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("images/model_comparison.png", dpi=120)
plt.close()
print("\nSaved images/model_comparison.png and model_comparison.csv")

# ---------------------------------------------------------------
# STEP 4: Interpret feature importances
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: FEATURE IMPORTANCES (Random Forest)")
print("=" * 60)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n", importances)

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            legend=False, palette="viridis")
plt.xlabel("Importance (mean decrease in impurity)")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("images/feature_importances.png", dpi=120)
plt.close()
print("\nSaved images/feature_importances.png")

print(
    f"\nTop 3 most important features: {', '.join(importances.head(3).index.tolist())}. "
    "These contribute the most to reducing impurity across the forest's splits, "
    "meaning they carry the strongest signal for predicting heart disease in this dataset."
)

# ---------------------------------------------------------------
# STEP 5: Evaluate using cross-validation
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: CROSS-VALIDATION (5-fold)")
print("=" * 60)

dt_cv_scores = cross_val_score(
    DecisionTreeClassifier(max_depth=int(best_depth), random_state=42), X, y, cv=5
)
rf_cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=200, random_state=42), X, y, cv=5
)

print(f"\nDecision Tree 5-fold CV scores: {np.round(dt_cv_scores, 4)}")
print(f"Decision Tree CV mean accuracy: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std():.4f})")

print(f"\nRandom Forest 5-fold CV scores: {np.round(rf_cv_scores, 4)}")
print(f"Random Forest CV mean accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

cv_results = pd.DataFrame({
    "Fold": list(range(1, 6)) * 2,
    "Model": ["Decision Tree"] * 5 + ["Random Forest"] * 5,
    "Accuracy": list(dt_cv_scores) + list(rf_cv_scores),
})
cv_results.to_csv("cross_validation_results.csv", index=False)

plt.figure(figsize=(7, 5))
sns.boxplot(data=cv_results, x="Model", y="Accuracy", hue="Model", legend=False, palette="Set3")
sns.stripplot(data=cv_results, x="Model", y="Accuracy", color="black", alpha=0.6)
plt.title("5-Fold Cross-Validation Accuracy: Decision Tree vs Random Forest")
plt.tight_layout()
plt.savefig("images/cross_validation.png", dpi=120)
plt.close()
print("\nSaved images/cross_validation.png and cross_validation_results.csv")
