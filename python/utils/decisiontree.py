from dtreeviz.trees import model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import logging
from utils.logging_config import setup_logging


def main() -> None:
    # Set up logger
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load and merge data
    logger.info("Reading and merging CSV files...")
    params = pd.read_csv("data/nosoi/master.csv")
    stats = pd.read_csv("data/scarce_stats/scarce_0.00.csv")
    params["seed"] = params["seed"].astype(str)
    stats["seed"] = stats["seed"].astype(str)
    merged = pd.merge(params, stats[["seed", "SST_06"]], on="seed", how="left")
    merged["non_trivial"] = (merged["SST_06"] > 1).astype(int)

    # Prepare features and labels
    logger.info("Preparing feature and label sets...")
    X = merged[[
        "mean_t_incub", "stdv_t_incub", "mean_nContact",
        "p_trans", "p_fatal", "mean_t_recovery"
    ]]
    y = merged["non_trivial"]

    # Train the decision tree
    logger.info("Fitting decision tree...")
    dtree = DecisionTreeClassifier(max_depth=4)
    dtree.fit(X, y)

    logger.info("Preparing dtreeviz model...")
    viz_model = model(
        model=dtree,
        X_train=X,
        y_train=y,
        feature_names=X.columns.tolist(),
        target_name="Simulation outcome",
        class_names=["Trivial", "Non-Trivial"]
    )

    # Now get the tree visualization object
    logger.info("Making dtreeviz visualization...")
    tree_viz = viz_model.view(
        orientation="LR",
        fontname="monospace",
        label_fontsize=16,
        precision=2,
        scale=1.5,
        leaftype="barh"
    )

    # Save to file
    logger.info("Saving decision tree as svg...")
    tree_viz.save("decision_tree.svg")


if __name__ == "__main__":
    main()
