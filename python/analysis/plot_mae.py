"""
DNN vs ABC Error Distribution Comparison

This module provides tools to compare the prediction error distributions of a
deep neural network (DNN) and Approximate Bayesian Computation (ABC) when
inferring parameters from nosoi simulations.

The workflow includes:
- Loading prediction errors from ABC and DNN results
- Computing and reshaping per-parameter error distributions
- Generating KDE and histogram plots for each parameter

This script is intended for benchmarking inference methods and supports both
headless batch execution and optional interactive display.

Dependencies
------------
- matplotlib
- pandas
- seaborn

Examples
--------
Run the script directly with CLI arguments:

    $ PYTHONPATH=python python3 -m compare_error_distributions.py \
        --abc data/benchmarks/abc_data.parquet \
        --dnn data/benchmarks/dnn_data.parquet \
        --output error_distributions.pdf \
        --show

See Also
--------
abc.py : Module for ABC posterior inference
dnn_mae.py : Module for computing DNN prediction errors
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns


def load_and_prepare_data(data_abc: Path, data_dnn: Path) -> pd.DataFrame:
    """
    Load ABC and DNN results, compute prediction errors, and combine a df.

    This function loads ABC posterior estimates and DNN prediction errors,
    computes the raw prediction errors for ABC (true - estimated), reshapes
    both datasets to long format, labels them by method, and combines them for
    joint plotting.

    Parameters
    ----------
    data_abc : Path
        Path to the parquet file containing true and posterior parameter
        estimates from ABC.
    data_dnn : Path
        Path to the parquet file containing DNN prediction errors per
        parameter.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns: ['Parameter', 'Error', 'Method'].
    """
    # Load the data
    abc_df = pd.read_parquet(data_abc)
    dnn_errors = pd.read_parquet(data_dnn)

    # Compute raw prediction errors for ABC
    abc_errors = pd.DataFrame({
        "mean_t_incub": abc_df["true_mean_t_incub"] - abc_df["post_mean_t_incub"],
        "stdv_t_incub": abc_df["true_stdv_t_incub"] - abc_df["post_stdv_t_incub"],
        "infectivity": abc_df["true_infectivity"] - abc_df["post_infectivity"],
        "p_fatal": abc_df["true_p_fatal"] - abc_df["post_p_fatal"],
        "mean_t_recovery": abc_df["true_mean_t_recovery"] - abc_df["post_mean_t_recovery"],
    })

    # Melt for plotting
    abc_melted = abc_errors.melt(var_name="Parameter", value_name="Error")
    abc_melted["Method"] = "ABC"

    dnn_melted = dnn_errors.melt(var_name="Parameter", value_name="Error")
    dnn_melted["Method"] = "DNN"

    combined_df = pd.concat([abc_melted, dnn_melted])
    return combined_df


def plot_error_distributions(
    df: pd.DataFrame,
    output_path: Path,
    show: bool
) -> None:
    """
    Plot and save error distribution histograms with KDE overlays per parameter.

    This function creates a multi-panel Seaborn facet plot comparing the
    prediction error distributions of DNN vs ABC for each parameter. The
    histograms and KDEs are overlaid, colored by method, and saved as a PDF.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the melted prediction error data with columns
        ['Parameter', 'Error', 'Method'].
    output_path : Path
        File path where the PDF plot will be saved.
    show : bool
        Whether to display the plot interactively after saving.
    """
    sns.set_theme(style="whitegrid")
    palette = {
        "ABC": "#23ab81",  # light blue
        "DNN": "#440154",  # dark purple
    }

    g = sns.FacetGrid(
        df,
        col="Parameter",
        col_wrap=2,
        sharex=False,
        sharey=False
    )

    g.map_dataframe(
        sns.histplot,
        x="Error",
        hue="Method",
        element="step",
        stat="density",
        common_norm=False,
        palette=palette,
        alpha=0.4
    )

    g.map_dataframe(
        sns.kdeplot,
        x="Error",
        hue="Method",
        common_norm=False,
        palette=palette,
        alpha=0.8,
        linewidth=2
    )

    for ax in g.axes.flat:
        ax.axvline(0, color="red", linestyle="--", linewidth=2)

    pretty_names = {
        "mean_t_incub": "Mean Incubation Time",
        "stdv_t_incub": "Std. Incubation Time",
        "infectivity": "Infectivity",
        "p_fatal": "Fatality Probability",
        "mean_t_recovery": "Mean Recovery Time"
    }
    g.set_titles(col_template="{col_name}", size=14)
    for ax in g.axes.flat:
        title = ax.get_title().split('=')[-1].strip()
        if title in pretty_names:
            ax.set_title(pretty_names[title], fontsize=14)

    for ax, param in zip(g.axes.flat, df["Parameter"].unique()):
        param_data = df[df["Parameter"] == param]["Error"]
        lower = param_data.quantile(0.001)
        upper = param_data.quantile(0.999)
        ax.set_xlim(lower, upper)

    # Set shared labels
    g.set_axis_labels("", "")  # Clear individual labels
    g.figure.supxlabel("Prediction Error", fontsize=14)
    g.figure.supylabel("Probability Density", fontsize=14)
    g.figure.subplots_adjust(left=0.07, bottom=0.07, wspace=0.1)

    # Optional: Set explicit size
    g.figure.set_size_inches(14, 10)

    g.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def cli_main():
    parser = argparse.ArgumentParser(
        description="Compare prediction error distributions of DNN vs ABC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter

    )
    parser.add_argument(
        "--abc", type=str, default="data/benchmarks/abc_data.parquet",
        help="Path to abc_data.parquet"
    )
    parser.add_argument(
        "--dnn", type=str, default="data/benchmarks/dnn_data.parquet",
        help="Path to dnn_data.parquet"
    )
    parser.add_argument(
        "--output", type=str, default="error_distributions.pdf",
        help="Path to save the resulting PDF figure"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the plot interactively"
    )

    args = parser.parse_args()
    df = load_and_prepare_data(Path(args.abc), Path(args.dnn))
    plot_error_distributions(df, args.output, show=args.show)


if __name__ == "__main__":
    cli_main()
