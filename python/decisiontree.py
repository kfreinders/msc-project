from dtreeviz.trees import model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    # Load and merge data
    params = pd.read_csv("data/nosoi/master.csv")
    stats = pd.read_csv("data/nosoi/summary_stats_export.csv")
    params["seed"] = params["seed"].astype(str)
    stats["seed"] = stats["seed"].astype(str)
    merged = pd.merge(params, stats[["seed", "SS_11"]], on="seed", how="left")
    merged["non_trivial"] = (merged["SS_11"] > 1).astype(int)

    # Prepare features and labels
    X = merged[[
        "mean_t_incub", "stdv_t_incub", "mean_nContact",
        "p_trans", "p_fatal", "t_recovery"
    ]]
    y = merged["non_trivial"]

    # Train the decision tree
    dtree = DecisionTreeClassifier(max_depth=4)
    dtree.fit(X, y)

    viz_model = model(
        model=dtree,
        X_train=X,
        y_train=y,
        feature_names=X.columns.tolist(),
        target_name="Simulation outcome",
        class_names=["Trivial", "Non-Trivial"]
    )

    # Now get the tree visualization object
    tree_viz = viz_model.view(
        orientation="LR",
        fontname="monospace",
        label_fontsize=16,
        precision=2,
        scale=1.5,
        leaftype="barh"
    )

    # Save to file
    tree_viz.save("decision_tree.svg")


if __name__ == "__main__":
    main()
