import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack
import numpy as np
import os
import pandas as pd
import sys

# Captum imports
from captum.attr import IntegratedGradients, DeepLift, Saliency

from .. import utils


def gatherData(x, mask, pad_token_id):
    seqs = []
    masks = []
    for i in range(x.size(0)):
        seq = x[i][mask[i]]
        length = seq.size(0)
        # The padding tuple for a 1D tensor is (pad_left, pad_right)
        padded_seq = F.pad(seq, (0, 512 - length), "constant", pad_token_id)
        seqs.append(padded_seq)

        mask_row = mask[i]
        length = mask_row.size(0)
        padded_mask = F.pad(mask_row, (0, 512 - length), "constant", False)
        masks.append(padded_mask)
    return torch.stack(seqs), torch.stack(masks)


class SCFoundationWrapper(nn.Module):
    """A class to load and use the scFoundation (DOI: https://doi.org/10.1038/s41592-024-02305-7) model for embedding and attribution."""

    def __init__(self, model_path: str, use_gpu: bool = False, pool_type="all"):
        """Constructor."""

        super().__init__()
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.pool_type = pool_type

        # To load the model, the 'load.py' script must be in the Python path.
        # You may need to uncomment and adjust the following line if it's not.
        sys.path.append(self.model_path)
        from load import load_model_frommmf

        # Load the gene order from the provided tsv file
        gene_list_path = os.path.join(self.model_path, "OS_scRNA_gene_index.19264.tsv")
        gene_list_df = pd.read_csv(gene_list_path, header=0, delimiter="\t")
        self.gene_order = list(gene_list_df["gene_name"].values)
        self.n_genes = len(self.gene_order)

        # Load the pre-trained model and config from the checkpoint file
        ckpt_path = os.path.join(self.model_path, "models.ckpt")
        key = "cell"
        self.model, self.pretrainconfig = load_model_frommmf(ckpt_path, key)

        if self.use_gpu is True:
            self.model.cuda()
        self.model.eval()
        self.eval()

        embedding_dim = 768

        if pool_type == "all":
            self.geneembmerge_layer = nn.Linear(
                embedding_dim * 4, embedding_dim * 4, bias=False
            )
        elif pool_type == "max":
            self.geneembmerge_layer = nn.Linear(
                embedding_dim, embedding_dim, bias=False
            )

    def _get_geneembmerge(self, gexpr_feature, gene_ids):
        """modified method from scFoundation code (https://github.com/biomap-research/scFoundation/tree/main) to get embedding"""
        value_labels = gexpr_feature > 0
        x, x_padding = gatherData(
            gexpr_feature, value_labels, self.pretrainconfig["pad_token_id"]
        )
        position_gene_ids, _ = gatherData(
            gene_ids, value_labels, self.pretrainconfig["pad_token_id"]
        )

        if x.size(1) == 0:
            batch_size = gexpr_feature.size(0)
            return torch.zeros(
                batch_size, self.geneembmerge_layer.in_features, device=x.device
            )

        x = self.model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.model.pos_emb(position_gene_ids)
        x += position_emb
        geneemb = self.model.encoder(x, x_padding)

        seq_len = geneemb.size(1)
        embedding_dim = geneemb.size(2)

        if seq_len >= 2:
            geneemb1 = geneemb[:, -1, :]
            geneemb2 = geneemb[:, -2, :]
        else:
            zero_tensor = torch.zeros(
                geneemb.size(0), embedding_dim, device=geneemb.device
            )
            geneemb1 = zero_tensor
            geneemb2 = zero_tensor

        if seq_len >= 3:
            geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
            geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
        else:
            zero_tensor = torch.zeros(
                geneemb.size(0), embedding_dim, device=geneemb.device
            )
            geneemb3 = zero_tensor
            geneemb4 = zero_tensor

        if self.pool_type == "all":
            geneembmerge = torch.concat(
                [geneemb1, geneemb2, geneemb3, geneemb4], axis=1
            )
        else:
            geneembmerge, _ = torch.max(geneemb, dim=1)

        return geneembmerge

    # Captum unpacks the tuple, so we must accept the unpacked arguments.
    def forward(
        self, inputs: torch.Tensor, gene_ids: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        geneembmerge = self._get_geneembmerge(inputs, gene_ids)

        summed_output = torch.sum(torch.abs(weights * geneembmerge), dim=1)

        return torch.cat(
            [summed_output.unsqueeze(1), summed_output.unsqueeze(1)], dim=-1
        )

    def preprocess_adata(
        self, adata: "anndata.AnnData", gene_overlap_threshold: int = 500
    ) -> "anndata.AnnData":
        """
        Preprocesses an AnnData object for use with the scFoundation model.
        """
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
        batch_size: int = 1,
        multiply_by_inputs: bool = True,
        disable_tqdm: bool = False,
        target_sum: float = 1e3,
        npz_path: Optional[str] = None,
    ) -> csr_matrix:
        """
        Calculates gene attributions for the scFoundation model using a specified method.

        Args:
            X: The input data matrix (e.g., log-normalized gene expression).
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

        gene_ids_tensor = torch.arange(self.n_genes).repeat(num_cells, 1)

        for i in tqdm(range(0, num_cells, batch_size), disable=disable_tqdm):
            torch.cuda.empty_cache()
            X_subset = X[i : i + batch_size, :]

            if isinstance(X_subset, csr_matrix):
                inputs_batch_gpu = torch.tensor(X_subset.todense(), dtype=torch.float32)
            else:
                inputs_batch_gpu = torch.tensor(X_subset, dtype=torch.float32)

            inputs_batch_gpu = inputs_batch_gpu.to(device).requires_grad_(True)

            gene_ids_batch_gpu = gene_ids_tensor[i : i + batch_size, :].to(device)

            with torch.no_grad():
                weights = self._get_geneembmerge(
                    inputs_batch_gpu, gene_ids_batch_gpu
                ).detach()

            additional_forward_args = (gene_ids_batch_gpu, weights)

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
