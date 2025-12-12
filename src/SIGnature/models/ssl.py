from collections import OrderedDict
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix, vstack
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

# Captum imports
from captum.attr import IntegratedGradients, DeepLift, Saliency

from .. import utils


def _reconstruct_mlp_from_state_dict(
    state_dict: OrderedDict, n_layers: int = 5
) -> nn.Module:
    """
    Reconstructs an MLP model from a state dictionary using the first n_layers that
    represent the encoder layers.
    This version correctly handles the layer-skipping seen in the provided models.
    """

    # We want the layers that represent the encoder
    layers = [x for x in state_dict.keys() if x.endswith(".weight")][0:n_layers]

    mlp_layers = []
    mlp_state_dict = OrderedDict()
    for i, key in enumerate(layers):
        in_size = state_dict[key].shape[1]
        out_size = state_dict[key].shape[0]

        prefix = ".".join(key.split(".")[:-1])
        bias = f"{prefix}.bias"
        has_bias = bias in state_dict

        new_prefix = prefix.replace(".", "_")
        mlp_layers.append((new_prefix, nn.Linear(in_size, out_size, bias=has_bias)))

        # All but last layer has activation
        if i < len(layers) - 1:
            # We ignore dropout as it is not used in inference
            # these models used nn.SELU() by default, so it is added as activation
            mlp_layers.append((f"{new_prefix}_act", nn.SELU()))

        mlp_state_dict[f"{new_prefix}.weight"] = state_dict[key]
        if has_bias:
            mlp_state_dict[f"{new_prefix}.bias"] = state_dict[bias]
    model_mlp = nn.Sequential(OrderedDict(mlp_layers))
    model_mlp.load_state_dict(mlp_state_dict)

    return model_mlp


class SSLWrapper(nn.Module):
    """A class to load and use self-supervised learning (SSL) models from Richter et al (DOI: https://doi.org/10.1038/s42256-024-00934-3) for embedding and attribution."""

    def __init__(
        self,
        model_path: str,
        model_filename: str = "classifier",
        n_layers: int = 5,
        use_gpu: bool = False,
    ):
        """Constructor.

        Args:
            model_path: Path to the directory containing model checkpoints.
            model_filename: The filename of the saved model.
            n_layers: The first n layers of the model are used. Default 5.
            use_gpu: Use GPU instead of CPU.
        """

        super().__init__()
        self.model_path = model_path
        self.use_gpu = use_gpu

        var_data = pd.read_parquet(os.path.join(self.model_path, "var.parquet"))
        self.gene_order = var_data["feature_name"].values.tolist()
        self.n_genes = len(self.gene_order)

        ckpt_path = os.path.join(self.model_path, model_filename)
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        state_dict = OrderedDict()
        for key, value in checkpoint.items():
            if key.endswith(".weight") or key.endswith(".bias"):
                state_dict[key] = value

        self.model = _reconstruct_mlp_from_state_dict(state_dict, n_layers=n_layers)

        if self.use_gpu:
            self.model.cuda()
        self.model.eval()
        self.eval()

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        summed_output = torch.sum(torch.abs(weights * self.model(inputs)), dim=1)
        return torch.cat(
            [summed_output.unsqueeze(1), summed_output.unsqueeze(1)], dim=-1
        )

    def preprocess_adata(
        self, adata: "anndata.AnnData", gene_overlap_threshold: int = 500
    ) -> "anndata.AnnData":
        adata.var_names_make_unique()
        adata = utils.align_dataset(
            data=adata,
            target_gene_order=self.gene_order,
            gene_overlap_threshold=gene_overlap_threshold,
        )
        adata = utils.lognorm_counts(adata)
        return adata

    def calculate_attributions(
        self,
        X: Union["torch.Tensor", "numpy.ndarray", "scipy.sparse.csr_matrix"],
        method: str = "ig",
        batch_size: int = 100,
        multiply_by_inputs: bool = True,
        disable_tqdm: bool = False,
        target_sum: float = 1e3,
        npz_path: Optional[str] = None,
    ) -> csr_matrix:
        """
        Calculates gene attributions for the SSL model using a specified method.

        Args:
            X: The input data matrix.
            method: The attribution method to use. Options are "ig" (Integrated Gradients),
                    "dl" (DeepLift), or "ixg" (Saliency).
            batch_size: The number of samples to process in each batch.
            multiply_by_inputs: Whether to multiply attributions by input values.
            disable_tqdm: Whether to disable the progress bar.
            target_sum: The desired sum for each row after normalization.
            npz_path: Path to save the resulting sparse attribution matrix.

        Returns:
            A scipy.sparse.csr_matrix containing the calculated attributions.
        """
        attribution_methods = {
            "ig": IntegratedGradients,
            "dl": DeepLift,
            "ixg": Saliency,
        }

        if method.lower() not in attribution_methods:
            raise ValueError(
                f"Unknown attribution method: {method}. Must be one of {list(attribution_methods.keys())}"
            )

        attr_class = attribution_methods[method.lower()]

        device = next(self.model.parameters()).device
        self.to(device)

        if method.lower() == "ixg":
            attributor = attr_class(self)
        else:
            attributor = attr_class(self, multiply_by_inputs=multiply_by_inputs)

        attrs_list = []
        num_cells = X.shape[0]

        for i in tqdm(range(0, num_cells, batch_size), disable=disable_tqdm):
            torch.cuda.empty_cache()
            X_subset = X[i : i + batch_size, :]

            if isinstance(X_subset, csr_matrix):
                inputs_batch_gpu = torch.tensor(X_subset.todense(), dtype=torch.float32)
            else:
                inputs_batch_gpu = torch.tensor(X_subset, dtype=torch.float32)

            inputs_batch_gpu = inputs_batch_gpu.to(device).requires_grad_(True)

            with torch.no_grad():
                weights = self.model(inputs_batch_gpu).detach()

            additional_forward_args = (weights,)

            if method.lower() == "ixg":
                attrs = attributor.attribute(
                    inputs=inputs_batch_gpu,
                    target=0,
                    additional_forward_args=additional_forward_args,
                )
                if multiply_by_inputs:
                    attrs = attrs * inputs_batch_gpu
            else:
                attrs = attributor.attribute(
                    inputs=inputs_batch_gpu,
                    baselines=torch.zeros_like(inputs_batch_gpu),
                    target=0,
                    additional_forward_args=additional_forward_args,
                )

            attrs_list.append(csr_matrix(torch.abs(attrs).detach().cpu().numpy()))

        return utils.normalize_attribution_matrix(
            matrix=vstack(attrs_list), target_sum=target_sum, npz_path=npz_path
        )
