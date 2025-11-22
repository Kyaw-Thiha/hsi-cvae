import sys

import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner.tuning import Tuner
from matplotlib.figure import Figure

from model import CVAELightningModule
from data_module import HyperspectralDataModule


def _configure_tensorcore_precision() -> None:
    """Switch TF32 configuration to the new PyTorch 2.9 API."""
    if not torch.cuda.is_available():
        return

    torch.set_float32_matmul_precision("medium")

    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    if matmul_backend and hasattr(matmul_backend, "fp32_precision"):
        matmul_backend.fp32_precision = "tf32"

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend and getattr(cudnn_backend, "is_available", lambda: False)():
        conv_backend = getattr(cudnn_backend, "conv", None)
        if conv_backend and hasattr(conv_backend, "fp32_precision"):
            conv_backend.fp32_precision = "tf32"


class CVAELightningCLI(LightningCLI):
    _MODEL_NAME_PLACEHOLDER = "${model.model_name}"

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument(
            "--fit.run_batch_size_finder", type=bool, default=False, help="Whether to run the batch size finder"
        )
        parser.add_argument(
            "--fit.batch_size_finder_mode",
            type=str,
            default="power",
            help="Mode for batch size finder (power|binsearch)",
        )
        parser.add_argument("--fit.run_lr_finder", type=bool, default=False, help="Whether to run learning rate finder")
        parser.add_argument("--fit.show_lr_plot", type=bool, default=True, help="Whether to plot learning rate finder")

    def before_fit(self):
        self._materialize_model_name_placeholders()
        tuner = Tuner(self.trainer)
        fit_config = getattr(self.config, "fit", None)
        run_batch_size_finder = getattr(fit_config, "run_batch_size_finder", False) if fit_config else False
        batch_size_mode = getattr(fit_config, "batch_size_finder_mode", "power") if fit_config else "power"
        run_lr_finder = getattr(fit_config, "run_lr_finder", False) if fit_config else False
        show_lr_plot = getattr(fit_config, "show_lr_plot", True) if fit_config else True

        # ----------------------------------
        # Batch Size Finder
        # ----------------------------------
        # CLI params
        #   run_batch_size_finder (bool): Determines if batch_size finder is ran. Default is True.
        #   batch_size_finder_mode (str): "power" or "binsearch". Determines the mode of batch_size finder
        # ----------------------------------
        if run_batch_size_finder:
            if self.trainer.fast_dev_run:  # pyright: ignore
                print("🚫 Skipping batch finder due to fast_dev_run")
            else:
                print(f"\n📦 Running batch size finder (mode: {batch_size_mode})...")

                new_batch_size = tuner.scale_batch_size(self.model, datamodule=self.datamodule, mode=batch_size_mode)

                print(f"✅ Suggested batch size: {new_batch_size}")

                if new_batch_size is None:
                    print("⚠️ Could not find optimal batch size")
            exit(0)

        # ----------------------------------
        # Finding Optimal Learning Rate
        # ----------------------------------
        # CLI params
        #   run_lr_finder (bool): Determines if LR finder is ran. Default is True.
        #   show_lr_plot (bool): Determines if LR finder plot is show. Default is False.
        # ----------------------------------
        if run_lr_finder:
            if self.trainer.fast_dev_run:  # pyright: ignore
                print("🚫 Skipping LR finder due to fast_dev_run")
            else:
                lr_finder = tuner.lr_find(self.model, datamodule=self.datamodule)

                if lr_finder is not None:
                    if show_lr_plot:
                        fig = lr_finder.plot(suggest=True)
                        if isinstance(fig, Figure):
                            fig.savefig("logs/lr_finder_plot.png")

                    suggested_lr = lr_finder.suggestion()
                    print(f"\n🔎 Suggested Learning Rate: {suggested_lr:.2e}")
                else:
                    print("⚠️ Could not find optimal learning rate")
            exit(0)

    # ----------------------------------
    # Setting the correct model name for checkpoints saving
    # ----------------------------------
    def _materialize_model_name_placeholders(self) -> None:
        """Replace `${model.model_name}` placeholders after CLI instantiation."""
        placeholder = self._MODEL_NAME_PLACEHOLDER
        model_name = getattr(self.model, "model_name", None)
        if not isinstance(model_name, str):
            return

        for callback in getattr(self.trainer, "callbacks", []):
            if isinstance(callback, ModelCheckpoint):
                dirpath = getattr(callback, "dirpath", None)
                if isinstance(dirpath, str) and placeholder in dirpath:
                    callback.dirpath = dirpath.replace(placeholder, model_name)

        loggers = getattr(self.trainer, "loggers", None) or []
        if not loggers:
            logger = getattr(self.trainer, "logger", None)
            loggers = [logger] if logger else []

        for logger in loggers:
            name = getattr(logger, "name", None)
            if isinstance(name, str) and placeholder in name:
                resolved = name.replace(placeholder, model_name)
                try:
                    setattr(logger, "name", resolved)
                except (AttributeError, TypeError):
                    if hasattr(logger, "_name"):
                        logger._name = resolved


def cli_main():
    _configure_tensorcore_precision()

    argv = sys.argv[1:]
    if argv:
        cli_args = None
    else:
        cli_args = ["fit", "--config", "config/cvae.yaml"]

    CVAELightningCLI(
        model_class=CVAELightningModule,
        datamodule_class=HyperspectralDataModule,
        args=cli_args,
    )


if __name__ == "__main__":
    cli_main()
