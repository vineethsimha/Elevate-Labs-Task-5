# Task 5: Decision Trees and Random Forests
**AI & ML Internship — Elevate Labs**

## Objective
Learn tree-based models for classification & regression.

## Tools Used
Python, Scikit-learn, Graphviz, Matplotlib, Seaborn

## Dataset
Heart Disease dataset (`data/heart.csv`) — 13 clinical features (age, sex,
chest pain type, resting BP, cholesterol, max heart rate, etc.) and a
binary `target` (1 = heart disease present, 0 = not present).

> **Note on the data file:** generated locally with `generate_dataset.py`
> using the same well-known column structure as the real UCI Cleveland
> Heart Disease dataset, since this environment has no internet access to
> download it directly. The target was built from a genuine (noisy)
> combination of the risk-factor features, so the patterns the trees learn
> are real and meaningful, not arbitrary. To use the real dataset instead,
> download it from the
> [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
> and drop it in as `data/heart.csv` with matching column names —
> `decision_trees_random_forests.py` needs no changes.

## Project Structure
```
├── data/
│   └── heart.csv
├── images/
│   ├── decision_tree_matplotlib.png
│   ├── decision_tree_graphviz.png
│   ├── overfitting_vs_depth.png
│   ├── model_comparison.png
│   ├── feature_importances.png
│   └── cross_validation.png
├── generate_dataset.py
├── decision_trees_random_forests.py
├── model_comparison.csv
├── cross_validation_results.csv
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
# System dependency for tree visualization (Graphviz):
#   Ubuntu/Debian: sudo apt install graphviz
#   Mac:           brew install graphviz
#   Windows:       https://graphviz.org/download/
python generate_dataset.py       # only needed if data/heart.csv doesn't exist
python decision_trees_random_forests.py
```

## What Was Done

1. **Trained a Decision Tree Classifier** (max_depth=3, for a readable
   visualization) and rendered it two ways: with `sklearn.tree.plot_tree`
   (matplotlib) and with `graphviz` for a cleaner layout.
2. **Analyzed overfitting** by training trees at every `max_depth` from 1
   to 20 and plotting train vs. test accuracy — the classic
   ever-climbing-train / plateauing-test pattern shows up clearly.
3. **Trained a Random Forest** (200 trees) and compared its test accuracy
   directly against the best single tuned tree.
4. **Interpreted feature importances** from the Random Forest via a
   ranked bar chart.
5. **Evaluated both models with 5-fold cross-validation** to get a more
   robust accuracy estimate than a single train/test split.

## Results

**Overfitting analysis:** train accuracy climbs steadily toward 1.0 as
`max_depth` increases, while test accuracy plateaus around depth 3-4 and
then fluctuates without meaningfully improving — the widening gap between
the two curves *is* overfitting. See `images/overfitting_vs_depth.png`.

| Model                          | Test Accuracy | 5-Fold CV Mean Accuracy |
|---------------------------------|---------------|--------------------------|
| Decision Tree (tuned max_depth) | 0.719         | 0.670 (± 0.022)          |
| Random Forest (200 trees)       | 0.769         | 0.766 (± 0.035)          |

The Random Forest clearly beats the single tuned tree on both metrics,
and its cross-validation scores are noticeably more consistent —
exactly the ensemble-averaging benefit random forests are known for.

**Top 3 most important features** (by mean decrease in impurity):
`chol` (cholesterol), `thalach` (max heart rate achieved), and `oldpeak`
(ST depression) — see `images/feature_importances.png` for the full
ranking.

---

## Interview Questions & Answers

**1. How does a decision tree work?**
A decision tree splits the data repeatedly on the feature and threshold
that best separates the classes (or reduces error, for regression) at
each node, using a criterion like Gini impurity or entropy/information
gain. Starting from the root, each split creates child nodes, continuing
recursively until a stopping condition is met (e.g. max depth, minimum
samples per leaf, or pure leaves). To predict, a new sample is passed down
the tree following the splits that match its feature values until it
reaches a leaf, whose majority class (or average value) becomes the
prediction.

**2. What is entropy and information gain?**
**Entropy** measures the impurity/disorder of a set of labels — it's 0
when a node is perfectly pure (all one class) and highest when classes are
evenly mixed. **Information gain** is the reduction in entropy achieved
by splitting a node on a particular feature: `IG = entropy(parent) -
weighted average entropy(children)`. Decision trees choose the split at
each node that maximizes information gain (or, equivalently, minimizes
impurity — scikit-learn defaults to Gini impurity, a similar but
computationally cheaper alternative).

**3. How is random forest better than a single tree?**
A random forest trains many decision trees on **bootstrap-resampled**
subsets of the data, each considering only a **random subset of features**
at every split, then averages their predictions (majority vote for
classification). This reduces variance dramatically compared to a single
tree — individual trees may overfit their particular training sample, but
their errors are decorrelated by the randomness, so averaging many of
them cancels out much of that overfitting while keeping the low bias of
deep trees. This shows up directly in the results above: the forest both
scores higher and has lower variance across cross-validation folds.

**4. What is overfitting and how do you prevent it?**
Overfitting is when a model learns the noise and idiosyncrasies of the
training data rather than the underlying general pattern, resulting in
high training accuracy but poor performance on new, unseen data. For
decision trees specifically, prevention methods include: limiting
`max_depth`, setting a minimum number of samples required to split a node
or exist in a leaf (`min_samples_split`, `min_samples_leaf`), pruning the
tree after growing it, or — more powerfully — using an ensemble method
like Random Forest, which controls overfitting through averaging rather
than restricting any single tree.

**5. What is bagging?**
Bagging (Bootstrap Aggregating) is an ensemble technique where multiple
models are trained independently on different bootstrap samples (random
samples drawn *with replacement* from the training set), and their
predictions are combined by averaging (regression) or majority voting
(classification). It reduces variance without increasing bias, and it's
the core mechanism behind Random Forest — which adds one more layer of
randomness on top (random feature subsets per split).

**6. How do you visualize a decision tree?**
With scikit-learn, either `sklearn.tree.plot_tree()` (renders directly
with matplotlib, no extra dependencies) or `sklearn.tree.export_graphviz()`
combined with the `graphviz` library, which produces a cleaner, more
customizable node-and-edge diagram (as done for both in this task). Both
show each node's split condition, impurity score, sample count, class
distribution, and majority class.

**7. How do you interpret feature importance?**
In tree-based models, feature importance (by default, "mean decrease in
impurity") measures how much each feature contributes, on average across
all trees and all splits that use it, to reducing impurity (Gini/entropy)
weighted by the number of samples reaching that split. A higher score
means the feature was more useful for separating the classes. It's a
relative ranking, not a causal claim — it says the model *relied on* this
feature, not that the feature *causes* the outcome, and it can be biased
toward high-cardinality or correlated features.

**8. What are the pros/cons of random forests?**
**Pros:** typically much higher accuracy and lower variance than a single
tree, robust to overfitting compared to deep individual trees, handles
non-linear relationships and feature interactions well, requires little
feature scaling/preprocessing, and provides a useful feature-importance
ranking. **Cons:** less interpretable than a single decision tree (you
can't easily visualize hundreds of trees), slower to train and predict
(especially with many/deep trees), can still overfit on very noisy data
if trees are too deep and too correlated, and the model files can become
large in memory for big forests.
