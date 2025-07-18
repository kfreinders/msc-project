import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import shap
import torch

from models.tuning import HyperParams, model_factory
from dataproc.nosoi_split import NosoiSplit
from utils.logging_config import setup_logging
from dataproc.summary_stats import sst_to_name


class WrappedModel(torch.nn.Module):
    """
    A wrapper around a multi-output neural network for SHAP explainability.

    This wrapper extracts a single output node from a multi-output model,
    allowing SHAP to explain one target variable at a time.

    Parameters
    ----------
    model : torch.nn.Module
        The trained multi-output PyTorch model to wrap.
    output_index : int
        The index of the output parameter to isolate for SHAP analysis.
    """
    def __init__(self, model, output_index: int):
        super().__init__()
        self.model = model
        self.output_index = output_index

    def forward(self, x):
        """
        Forward pass through the model, returning only the selected output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N, 1) corresponding to the selected output
            dimension.
        """
        return self.model(x)[:, self.output_index].unsqueeze(1)


def _sample_inputs(
    X: torch.Tensor,
    max_samples: int,
    device: torch.device
) -> tuple[torch.Tensor, np.ndarray]:
    logger = logging.getLogger(__name__)
    if len(X) < max_samples:
        logger.warning(
            f"Warning: max_samples={max_samples} > len(X)={len(X)}. "
            f"Reducing to len(X)."
        )
        max_samples = len(X)

    idx = np.random.choice(len(X), size=max_samples, replace=False)
    inputs_tensor = X[idx].clone().detach().to(
        dtype=torch.float32, device=device
    )
    inputs_numpy = inputs_tensor.cpu().numpy()
    return inputs_tensor, inputs_numpy


def _get_background(
    X: torch.Tensor,
    background_size: int,
    device: torch.device
) -> torch.Tensor:
    return X[:background_size].clone().detach().to(
        dtype=torch.float32, device=device
    )


def _compute_shap_for_output(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    background: torch.Tensor,
    output_index: int,
    device: torch.device
) -> np.ndarray:
    wrapped_model = WrappedModel(model, output_index).to(device)
    explainer = shap.DeepExplainer(wrapped_model, background)
    shap_values = explainer(inputs).values
    return shap_values.squeeze(-1)


def _plot_shap_violin(
    shap_values: np.ndarray,
    inputs_numpy: np.ndarray,
    feature_names: list[str],
    output_name: str,
    output_dir: Path
) -> Path:
    plt.figure(figsize=(12, 6), constrained_layout=True)
    shap.plots.violin(
        shap_values,
        inputs_numpy,
        feature_names=feature_names,
        max_display=5,
        show=False,
    )
    plt.title(f"{output_name}")
    output_path = output_dir / f"shap_summary_{output_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def make_shap_plots(
    model: torch.nn.Module,
    test_split: NosoiSplit,
    device: torch.device,
    max_samples: int = 5000,
    background_size: int = 100,
    output_dir: Path = Path(".")
) -> list[Path]:
    """
    Generate SHAP violin plots for each output parameter and save them to disk.
    """
    model.eval()

    X = test_split.X
    feature_names = test_split.x_colnames
    feature_names = [sst_to_name(feature) for feature in feature_names]
    output_names = test_split.y_colnames

    background = _get_background(X, background_size, device)
    inputs_tensor, inputs_numpy = _sample_inputs(X, max_samples, device)

    png_files: list[Path] = []
    for i, output_name in enumerate(output_names):
        print(f"Computing SHAP for output: {output_name}")
        shap_values = _compute_shap_for_output(
            model,
            inputs_tensor,
            background,
            i,
            device
        )

        png_path = _plot_shap_violin(
            shap_values,
            inputs_numpy,
            feature_names,
            output_name,
            output_dir
        )

        png_files.append(png_path)

    return png_files


def combine_shap_images(image_paths, out_path, rows=3, cols=2) -> None:
    """
    Combine multiple SHAP summary plots into a single composite image.

    This function reads a list of individual image files (e.g., SHAP violin
    plots) and arranges them in a grid layout (rows × cols), then saves the
    result to disk.

    Parameters
    ----------
    image_paths : list[Path] or list[str]
        Paths to the individual SHAP plot images to be combined.
    out_path : Path or str
        Output file path where the combined image will be saved.
    rows : int, optional
        Number of rows in the output grid layout. Default is 2.
    cols : int, optional
        Number of columns in the output grid layout. Default is 3.
    """
    images = [Image.open(p) for p in image_paths]
    widths, heights = zip(*(img.size for img in images))

    max_width = max(widths)
    max_height = max(heights)

    combined = Image.new("RGB", (cols * max_width, rows * max_height), "white")

    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        combined.paste(img, (col * max_width, row * max_height))

    combined.save(out_path)
    print(f"Saved combined SHAP figure to {out_path}")


def main() -> None:
    # Set up logger
    setup_logging("training")
    logger = logging.getLogger(__name__)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data splits from disk
    splits_path = Path("data/splits/scarce_0.00")
    train_split = NosoiSplit.load("train", splits_path)
    test_split = NosoiSplit.load("test", splits_path)
    logger.info(f"Loaded saved data splits from {splits_path}")

    cfg = HyperParams(
        learning_rate=0.0005186374528320235,
        hidden_size=256,
        num_layers=2,
        dropout_rate=0.10448580769582116,
        batch_size=16
    )
    logger.info(f"HyperParams: {cfg}")

    model = model_factory(
        train_split.input_dim,
        train_split.output_dim,
        cfg,
        device
    )

    # Load model (redundant in case since already in memory)
    model_path = Path("data/dnn/scarce_0.00/regressor.pt")
    logger.info(f"Loading saved model from: {model_path}")
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)

    logger.info("Computing SHAP values...")
    png_files = make_shap_plots(model, test_split, device)
    combine_shap_images(png_files, out_path="shap_summary_combined.png")
    logger.info("SHAP summary saved as shap_summary.png")


if __name__ == "__main__":
    main()
