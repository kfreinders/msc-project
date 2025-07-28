import argparse
import logging
import json
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


class ShapAnalyzer:
    """
    A utility class for computing and visualizing SHAP (SHapley Additive
    exPlanations) values for a trained deep neural network.

    This class loads a trained multi-output PyTorch model, applies the
    DeepExplainer method from the `shap` library to approximate SHAP values,
    and generates violin plots summarizing the contribution of each input
    feature to the model's predictions. The analysis provides insights into
    which epidemiological summary statistics most strongly influence parameter
    inference in nosoi simulations.

    Parameters
    ----------
    splits_path : Path
        Path to the directory containing the saved `NosoiSplit` test set.
    model_path : Path
        Path to the trained model directory, containing `regressor.pt` and
        `best_config.json`.
    output_path : Path
        Directory where SHAP violin plots will be saved.
    n_samples : int
        Number of test set samples to use for computing SHAP values.
    max_display : int
        Maximum number of features to display in each violin plot.
    background_size : int
        Number of background samples used for DeepExplainer integration.
    """
    def __init__(
        self,
        splits_path: Path,
        model_path: Path,
        output_path: Path,
        n_samples: int,
        max_display: int,
        background_size: int,
    ) -> None:
        self.test_split = NosoiSplit.load("test", splits_path)
        self.output_path = output_path
        self.n_samples = n_samples
        self.max_display = max_display
        self.background_size = background_size
        self.feature_names = self.test_split.x_colnames
        self.feature_names = [sst_to_name(feature) for feature in self.feature_names]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._construct_model(model_path)

        self.output_path.mkdir(parents=True, exist_ok=True)

    def _construct_model(self, model_path):
        """
        Load and reconstruct the trained deep neural network from disk.

        This method reads the model hyperparameters from `best_config.json`,
        builds the architecture via `model_factory`, and loads the trained
        weights from `regressor.pt`.

        Parameters
        ----------
        model_path : Path
            Directory containing the saved model state and hyperparameters.

        Returns
        -------
        torch.nn.Module
            The trained PyTorch model ready for inference and SHAP analysis.
        """
        # Retrieve hyperparameters
        with open(model_path / "best_config.json") as handle:
            data = json.load(handle)
            cfg = HyperParams.from_dict(data)

        model = model_factory(
            self.test_split.input_dim,
            self.test_split.output_dim,
            cfg,
            self.device
        )
        # Load model
        model.load_state_dict(
            torch.load(model_path / "regressor.pt", map_location=self.device)
        )

        model.to(self.device)

        return model

    def _sample_inputs(
        self,
        X: torch.Tensor,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Randomly sample a subset of input data for SHAP analysis.

        This ensures computational feasibility by limiting the number of
        explained instances while maintaining representativeness.

        Parameters
        ----------
        X : torch.Tensor
            Full test set input tensor.

        Returns
        -------
        tuple[torch.Tensor, np.ndarray]
            - A PyTorch tensor with the sampled instances, placed on the
              appropriate device.
            - A NumPy array version of the same inputs for plotting.
        """
        logger = logging.getLogger(__name__)
        n_samples = self.n_samples
        if len(X) < self.n_samples:
            logger.warning(
                f"Warning: max_samples={self.n_samples} > len(X)={len(X)}. "
                f"Reducing to len(X)."
            )
            n_samples = len(X)

        idx = np.random.choice(len(X), size=n_samples, replace=False)
        inputs_tensor = X[idx].clone().detach().to(
            dtype=torch.float32, device=self.device
        )
        inputs_numpy = inputs_tensor.cpu().numpy()
        return inputs_tensor, inputs_numpy

    def _get_background(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select background samples for DeepExplainer integration.

        Background samples are used by DeepExplainer to approximate the
        conditional expectations of Shapley values.

        Parameters
        ----------
        X : torch.Tensor
            Full test set input tensor.

        Returns
        -------
        torch.Tensor
            Tensor of shape (background_size, n_features) containing the
            background samples.
        """
        return X[:self.background_size].clone().detach().to(
            dtype=torch.float32, device=self.device
        )

    def _compute_shap_for_output(
        self,
        inputs: torch.Tensor,
        background: torch.Tensor,
        output_index: int,
    ) -> np.ndarray:
        """
        Compute SHAP values for a single model output parameter.

        This wraps the multi-output model to isolate the target output,
        then applies DeepExplainer to compute per-feature attributions.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of test samples for SHAP explanation.
        background : torch.Tensor
            Background dataset used for DeepExplainer integration.
        output_index : int
            Index of the output parameter to explain.

        Returns
        -------
        np.ndarray
            Array of SHAP values with shape (n_samples, n_features).
        """
        wrapped_model = WrappedModel(self.model, output_index).to(self.device)
        explainer = shap.DeepExplainer(wrapped_model, background)
        shap_values = explainer(inputs).values
        return shap_values.squeeze(-1)

    def _plot_shap_violin(
        self,
        shap_values: np.ndarray,
        inputs_numpy: np.ndarray,
        output_name: str,
    ) -> Path:
        """
        Create and save a SHAP violin plot for a single output parameter.

        Violin plots summarize the distribution and directionality of SHAP
        values for the top contributing features.

        Parameters
        ----------
        shap_values : np.ndarray
            SHAP values for the given output parameter, shape (n_samples,
            n_features).
        inputs_numpy : np.ndarray
            NumPy array of the corresponding input samples.
        output_name : str
            Label of the output parameter being explained.

        Returns
        -------
        Path
            Path to the saved violin plot image.
        """
        plt.figure(figsize=(12, 6), constrained_layout=True)
        shap.plots.violin(
            shap_values,
            inputs_numpy,
            feature_names=self.feature_names,
            max_display=self.max_display,
            show=False,
        )
        plt.title(f"{output_name}")
        output_path = self.output_path / f"shap_summary_{output_name}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def make_shap_plots(self) -> list[Path]:
        """
        Generate SHAP violin plots for each output parameter and save to disk.
        """
        logger = logging.getLogger(__name__)

        self.model.eval()

        X = self.test_split.X
        output_names = self.test_split.y_colnames

        background = self._get_background(X)
        inputs_tensor, inputs_numpy = self._sample_inputs(X)

        png_files: list[Path] = []
        for i, output_name in enumerate(output_names):
            logger.info(f"Computing SHAP for output: {output_name}")
            shap_values = self._compute_shap_for_output(
                inputs_tensor,
                background,
                i,
            )

            np.save(
                self.output_path / f"shap_values_{output_name}.npy",
                shap_values
            )

            png_path = self._plot_shap_violin(
                shap_values,
                inputs_numpy,
                output_name,
            )

            png_files.append(png_path)

        return png_files

    @staticmethod
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
        logger = logging.getLogger(__name__)

        images = [Image.open(p) for p in image_paths]
        widths, heights = zip(*(img.size for img in images))

        max_width = max(widths)
        max_height = max(heights)

        combined = Image.new(
            "RGB", (cols * max_width, rows * max_height), "white"
        )

        for idx, img in enumerate(images):
            row, col = divmod(idx, cols)
            combined.paste(img, (col * max_width, row * max_height))

        combined.save(out_path)
        logger.info(f"Saved combined SHAP figure to {out_path}")


def cli_main():
    parser = argparse.ArgumentParser(
        description=(
            "Make SHAP violin plots of summary statistics for a DNN."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--splits-path", type=str, default="data/splits/scarce_0.00",
        help="Path to the directory containing pickled NosoiSplit object."
    )
    parser.add_argument(
        "--model-path", type=str, default="data/dnn/scarce_0.00",
        help=(
            "Path to the directory containing the pickled DNN (regressor.pt) "
            "and hyperparameter configuration (best_config.json)"
        )
    )
    parser.add_argument(
        "--n-samples", type=int, default=1_000,
        help="Number of samples to use."
    )
    parser.add_argument(
        "--background-size", type=int, default=1_000,
        help="Number of background samples."
    )
    parser.add_argument(
        "--max-display", type=int, default=5,
        help="How many features to display in the SHAP violin plots."
    )
    parser.add_argument(
        "--output-path", type=str, default="data/shap",
        help="Directory to save output data."
    )

    args = parser.parse_args()
    analyzer = ShapAnalyzer(
        splits_path=Path(args.splits_path),
        model_path=Path(args.model_path),
        output_path=Path(args.output_path),
        n_samples=args.n_samples,
        max_display=args.max_display,
        background_size=args.background_size,
    )
    png_paths = analyzer.make_shap_plots()
    analyzer.combine_shap_images(
        png_paths, Path(args.output_path) / "shap_summary_combined.png"
    )


if __name__ == "__main__":
    cli_main()
