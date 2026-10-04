# ============================================================
# CS-13410 Assignment 1 - ID3 Decision Tree
# Project dataset: CS2 cheat detection (player statistics)
# Dataset file: cs2_engineered_dataset.csv
# (github.com/BadrBerqia/cs2-statistical-anticheat)
# No sklearn decision tree is used. ID3 is written from scratch.
# ============================================================
import math
from collections import Counter
import pandas as pd

URL = ("https://raw.githubusercontent.com/BadrBerqia/"
       "cs2-statistical-anticheat/main/Data/cs2_engineered_dataset.csv")
SEED = 42
TARGET = "Class"
ATTRS = ["kill_death_ratio", "headshot_ratio", "total_accuracy", "win_ratio"]
BIN_NAMES = ["Low", "Medium", "High"]

# ------------------------------------------------------------
# 1. Load and clean the data
# ------------------------------------------------------------
df = pd.read_csv(URL)
print("Full dataset:", df.shape[0], "players")

# Keep only valid rows: at least 100 rounds played, ratios between 0 and 1,
# and a K/D ratio that is possible (the raw file has values like 125000).
df = df[df["total_rounds_played"] >= 100]
df = df[(df["win_ratio"] <= 1) & (df["headshot_ratio"] <= 1)
        & (df["total_accuracy"] <= 1) & (df["kill_death_ratio"] <= 10)]
df = df.copy()
df[TARGET] = df["vac_banned"].map({1: "Cheater", 0: "Legitimate"})
print("After cleaning:", len(df), "players")
print(df[TARGET].value_counts().to_string())

# ------------------------------------------------------------
# 2. Sample: 30 Cheater + 30 Legitimate for training,
#    then 3 different players (never used in training) for testing
# ------------------------------------------------------------
cheaters = df[df[TARGET] == "Cheater"]
legit = df[df[TARGET] == "Legitimate"]

train_c = cheaters.sample(30, random_state=SEED)
train_l = legit.sample(30, random_state=SEED)
train_raw = pd.concat([train_c, train_l])

test_raw = pd.concat([
    cheaters.drop(train_c.index).sample(1, random_state=SEED),
    legit.drop(train_l.index).sample(2, random_state=SEED),
])
test_raw.index = ["Test A", "Test B", "Test C"]

# ------------------------------------------------------------
# 3. Discretize numbers into Low / Medium / High
#    (cut points = 33rd and 66th percentile of the TRAINING data)
# ------------------------------------------------------------
cuts = {}
for a in ATTRS:
    q1, q2 = train_raw[a].quantile([1 / 3, 2 / 3])
    cuts[a] = [-math.inf, q1, q2, math.inf]
    print(f"{a}: Low <= {q1:.4f} < Medium <= {q2:.4f} < High")

def to_categories(raw):
    out = pd.DataFrame(index=raw.index)
    for a in ATTRS:
        out[a] = pd.cut(raw[a], bins=cuts[a], labels=BIN_NAMES).astype(str)
    out[TARGET] = raw[TARGET]
    return out

train = to_categories(train_raw)
test = to_categories(test_raw)

# ------------------------------------------------------------
# 4. Entropy and Information Gain
# ------------------------------------------------------------
def entropy(labels):
    """Entropy in bits of a list of class labels."""
    total = len(labels)
    return -sum((n / total) * math.log2(n / total)
                for n in Counter(labels).values())

def information_gain(data, attribute):
    """Entropy(S) - weighted entropy of the subsets made by attribute."""
    total = len(data)
    remainder = 0.0
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        remainder += len(subset) / total * entropy(subset[TARGET])
    return entropy(data[TARGET]) - remainder

def majority_class(labels):
    counts = Counter(labels)
    return sorted(counts, key=lambda c: (-counts[c], c))[0]  # tie -> alphabetical

# ------------------------------------------------------------
# 5. ID3: build the tree recursively
#    leaf = a class name (text); node = dictionary
# ------------------------------------------------------------
def build_tree(data, attributes, path="Root"):
    labels = list(data[TARGET])

    # Stop (a): all rows have the same class -> leaf
    if len(set(labels)) == 1:
        return labels[0]
    # Stop (b): no attributes left -> leaf with majority class
    if not attributes:
        return majority_class(labels)

    # Choose the attribute with the highest Information Gain
    gains = {a: information_gain(data, a) for a in attributes}
    best = max(gains, key=gains.get)

    counts = Counter(labels)
    print(f"\n[{path}] n={len(data)} Cheater={counts['Cheater']} "
          f"Legitimate={counts['Legitimate']} Entropy={entropy(labels):.4f}")
    print("  Gains: " + ", ".join(f"{a}={g:.4f}" for a, g in gains.items()))
    print(f"  -> split on {best} (Gain={gains[best]:.4f})")

    node = {"attribute": best, "majority": majority_class(labels), "branches": {}}
    remaining = [a for a in attributes if a != best]
    for value in sorted(data[best].unique()):
        subset = data[data[best] == value]
        node["branches"][value] = build_tree(
            subset, remaining, f"{path} > {best}={value}")
    return node

def show_tree(node, indent=0):
    if not isinstance(node, dict):
        return " " * indent + "-> " + node + "\n"
    text = ""
    for value, child in node["branches"].items():
        text += " " * indent + f"{node['attribute']} = {value}\n"
        text += show_tree(child, indent + 4)
    return text

def classify(tree, row):
    """Walk down the tree; return the prediction and the path taken."""
    steps = []
    while isinstance(tree, dict):
        attribute = tree["attribute"]
        value = row[attribute]
        steps.append(f"{attribute} = {value}")
        if value not in tree["branches"]:      # branch never seen in training
            return tree["majority"], steps
        tree = tree["branches"][value]
    return tree, steps

# ------------------------------------------------------------
# 6. Run everything
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ROOT NODE ANALYSIS")
print("=" * 60)
print("Entropy of training set:", round(entropy(train[TARGET]), 4), "bits")

tree = build_tree(train, ATTRS)

# Detail of the root split (for the hand check)
root = tree["attribute"]
print(f"\nRoot split detail for {root}:")
for value in sorted(train[root].unique()):
    sub = train[train[root] == value]
    c = Counter(sub[TARGET])
    print(f"  {value:<7} n={len(sub):<3} Cheater={c['Cheater']:<3} "
          f"Legitimate={c['Legitimate']:<3} Entropy={entropy(sub[TARGET]):.4f}")

print("\n" + "=" * 60)
print("FINAL DECISION TREE")
print("=" * 60)
print(show_tree(tree))

print("=" * 60)
print("CLASSIFICATION OF TEST A, B, C")
print("=" * 60)
for name, row in test.iterrows():
    prediction, steps = classify(tree, row)
    print(f"\n{name}")
    for a in ATTRS:
        print(f"  {a} = {test_raw.loc[name, a]:.4f} -> {row[a]}")
    print("  Path     :", " -> ".join(steps))
    print("  Predicted:", prediction, "| Actual:", row[TARGET])
