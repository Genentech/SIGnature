import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack
import os
import pandas as pd

# Captum imports
from captum.attr import IntegratedGradients, DeepLift, Saliency
import scvi

from .. import utils


class SCVIWrapper(nn.Module):
    """A class to load and use a pre-trained scVI (https://doi.org/10.1038/s41592-018-0229-2) model for embedding and attribution."""

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
        cat_value: int = 6006,
    ):
        """Constructor.

        Args:
            model_path: Path to the directory containing the saved scVI model.
            use_gpu: Use GPU instead of CPU.
            cat_value: The categorical value to use for the `cat_list` in the encoder's forward pass.
        """

        super().__init__()
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.cat_value = cat_value

        # The model will be loaded just in time in the calculate_attributions method.
        self.model = None
        # The gene order is defined by the model, loaded just in time in the calculate_attributions method.
        self.gene_order = None

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SCVI model file not found at {self.model_path}")

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        cat_list = [
            torch.full((inputs.size(0), 1), self.cat_value, dtype=torch.long).to(
                inputs.device
            )
        ]

        # scvi is a VAE so it outputs mean and variance; we consider the mean ([0]) for attributions
        model_output = self.model(inputs, *cat_list)[0].loc

        summed_output = torch.sum(torch.abs(weights * model_output), dim=1)

        return torch.cat(
            [summed_output.unsqueeze(1), summed_output.unsqueeze(1)], dim=-1
        )

    def preprocess_adata(
        self,
        adata: "anndata.AnnData",
        ensembl_gene_file: Optional[str] = None,
        gene_overlap_threshold: int = 500,
    ) -> "anndata.AnnData":
        """
        Preprocesses an AnnData object for use with the scVI model.

        This method must be called before calculate_attributions().
        """
        adata.obs["n_counts"] = adata.layers["counts"].sum(axis=1)
        adata.obs["joinid"] = list(range(adata.n_obs))

        # Load the full internal registry dictionary
        registry = scvi.model.SCVI.load_registry(self.model_path)

        # The original setup arguments are stored under the '_setup_args' key
        setup_args = registry["setup_args"]
        ## Go through setup args and add missing obs columns
        for k, v in setup_args.items():
            if "keys" in k:
                if v is not None:
                    for item in v:
                        adata.obs[item] = "unassigned"
            elif "key" in k:
                if v is not None:
                    adata.obs[v] = "unassigned"

        # converts to ensembl gene names if file included
        if ensembl_gene_file is not None:
            xdf = pd.read_table(ensembl_gene_file)
            gl = dict(zip(xdf["HGNC symbol"], xdf["Gene stable ID"]))
            adata.var["symbol"] = adata.var.index.tolist()
            adata.var["ensembl_id"] = [gl.get(x) for x in adata.var["symbol"]]
            adata.var.index = map(str, adata.var["ensembl_id"].values)

        adata.var_names_make_unique()
        adata.obs_names_make_unique()

        # This is the critical, scVI-specific preprocessing step
        scvi.model.SCVI.prepare_query_anndata(adata, self.model_path)

        return adata

    def calculate_attributions(
        self,
        adata: "anndata.AnnData",
        method: str = "ig",
        batch_size: int = 100,
        multiply_by_inputs: bool = True,
        disable_tqdm: bool = False,
        target_sum: float = 1e3,
        npz_path: Optional[str] = None,
    ) -> csr_matrix:
        """
        Calculates gene attributions for the scVI model using a specified method.

        Args:
            adata: A preprocessed AnnData object (must be prepared with `self.preprocess_adata`).
            method: The attribution method to use. Options are "ig" (Integrated Gradients),
                    "dl" (DeepLift), or "ixg" (Saliency).
            batch_size: The number of samples to process in each batch.
            multiply_by_inputs: Whether to multiply attributions by input values.
                                Note: for Integrated Gradients and DeepLift, this is
                                passed to the Captum constructor. For Saliency, the
                                multiplication is done manually after calculation.
            disable_tqdm: Whether to disable the progress bar.
            target_sum: The desired sum for each row after normalization.
            npz_path: Path to save the resulting sparse attribution matrix.

        Returns:
            A scipy.sparse.csr_matrix containing the calculated attributions.
        """
        # The model is loaded here, ensuring it has the correct adata metadata

        vae_q = scvi.model.SCVI.load_query_data(adata, self.model_path)
        self.gene_order = vae_q.adata.var.index.tolist()
        vae_q.is_trained = True
        self.model = vae_q.module.z_encoder
        self.model.eval()
        self.eval()

        if self.use_gpu:
            self.model.cuda()

        X = adata.X

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
                cat_list_batch = [
                    torch.full(
                        (inputs_batch_gpu.size(0), 1), self.cat_value, dtype=torch.long
                    ).to(device)
                ]
                weights = self.model(inputs_batch_gpu, *cat_list_batch)[0].loc.detach()

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
