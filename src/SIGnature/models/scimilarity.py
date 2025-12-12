import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from .. import utils


class Encoder(nn.Module):
    """A class that encapsulates the SCimilarity encoder."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        residual: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        n_genes: int
            The number of genes in the gene space, representing the input dimensions.
        latent_dim: int, default: 128
            The latent space dimensions
        hidden_dim: List[int], default: [1024, 1024]
            A list of hidden layer dimensions, describing the number of layers and their dimensions.
            Hidden layers are constructed in the order of the list for the encoder and in reverse
            for the decoder.
        dropout: float, default: 0.5
            The dropout rate for hidden layers
        input_dropout: float, default: 0.4
            The dropout rate for the input layer
        residual: bool, default: False
            Use residual connections.
        """

        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(n_genes, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # output layer
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor corresponding to input layer.

        Returns
        -------
        torch.Tensor
            Output tensor corresponding to output layer.
        """

        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def save_state(self, filename: str):
        """Save model state.

        Parameters
        ----------
        filename: str
            Filename to save the model state.
        """

        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Parameters
        ----------
        filename: str
            Filename containing the model state.
        use_gpu: bool, default: False
            Boolean indicating whether or not to use GPUs.
        """

        if not use_gpu:
            ckpt = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=False
            )
        else:
            ckpt = torch.load(filename, weights_only=False)
        self.load_state_dict(ckpt["state_dict"])


class SCimilarityWrapper(nn.Module):
    """
    A wrapper for the SCimilarity (https://doi.org/10.1038/s41586-024-08411-y) Encoder model to enable attribution methods.

    This adapts the output of the model for use with Captum's attribution
    algorithms. Its forward method requires an additional 'weights' tensor
    to be passed, which is the output of the original model on the input.
    A class loads the SCimilarity model.
    """

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        model_path: str
            Path to the directory containing model files.
        use_gpu: bool, default: False
            Use GPU instead of CPU.
        filenames: dict, optional, default: None
            Use a dictionary of custom filenames for model files instead default.
        residual: bool, default: False
            Use residual connections.

        Examples
        --------
        >>> ce = CellEmbedding(model_path="/opt/data/model")
        """

        import json
        import os
        import pandas as pd

        super().__init__()
        self.model_path = model_path
        self.use_gpu = use_gpu

        self.filenames = {
            "model": os.path.join(self.model_path, "encoder.ckpt"),
            "gene_order": os.path.join(self.model_path, "gene_order.tsv"),
        }

        # get gene order
        with open(self.filenames["gene_order"], "r") as fh:
            self.gene_order = [line.strip() for line in fh]

        # get neural network model and infer network size
        with open(os.path.join(self.model_path, "layer_sizes.json"), "r") as fh:
            layer_sizes = json.load(fh)
        # keys: network.1.weight, network.2.weight, ..., network.n.weight
        layers = [
            (key, layer_sizes[key])
            for key in sorted(list(layer_sizes.keys()))
            if "weight" in key and len(layer_sizes[key]) > 1
        ]
        parameters = {
            "latent_dim": layers[-1][1][0],  # last
            "hidden_dim": [layer[1][0] for layer in layers][0:-1],  # all but last
        }

        self.n_genes = len(self.gene_order)
        self.latent_dim = parameters["latent_dim"]
        self.model = Encoder(
            n_genes=self.n_genes,
            latent_dim=parameters["latent_dim"],
            hidden_dim=parameters["hidden_dim"],
            residual=False,
        )
        if self.use_gpu is True:
            self.model.cuda()
        self.model.load_state(self.filenames["model"])
        self.model.eval()
        self.eval()

        self.int2label = pd.read_csv(
            os.path.join(self.model_path, "label_ints.csv"), index_col=0
        )["0"].to_dict()
        self.label2int = {value: key for key, value in self.int2label.items()}

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        The forward pass designed for Captum.

        This method is a simple pass-through to the original model, with the
        final output multiplied by a pre-computed 'weights' tensor.

        Args:
            inputs: A tensor of shape [batch_size, n_genes].
            weights: The pre-computed model output for the same batch,
                     with shape [batch_size, latent_dim].

        Returns:
            A tensor of shape [batch_size, 2].
        """
        # Your core logic: multiply the output by the pre-computed weights
        summed_output = torch.sum(torch.abs(weights * self.model(inputs)), dim=1)

        # Return a tensor of shape [batch_size, 2] to satisfy DeepLift/GradientShap
        return torch.cat(
            [summed_output.unsqueeze(1), summed_output.unsqueeze(1)], dim=-1
        )

    def preprocess_adata(
        self, adata: "anndata.AnnData", gene_overlap_threshold: int = 500
    ) -> "anndata.AnnData":
        """
        Preprocesses an AnnData object for use with the SCimilarity model.

        This method aligns the gene space, subsets the data to the model's gene order,
        and log-normalizes the counts.

        Args:
            adata: The AnnData object to be preprocessed.
            gene_overlap_threshold: The minimum number of genes in common between
                                    the AnnData object and the model's gene order.

        Returns:
            The preprocessed AnnData object.
        """
        # Ensure the `adata.var.index` is a list of gene symbols
        # (This is a good practice to handle different AnnData setups)
        adata.var_names_make_unique()

        # 1. Align the dataset to the model's gene order
        adata = utils.align_dataset(
            data=adata,
            target_gene_order=self.gene_order,
            gene_overlap_threshold=gene_overlap_threshold,
        )

        # 2. Log-normalize the counts
        adata = utils.lognorm_counts(adata)

        return adata

    def calculate_attributions(
        self,
        X: Union["torch.Tensor", "numpy.ndarray", "scipy.sparse.csr_matrix"],
        method: str = "ig",
        batch_size: int = 500,
        multiply_by_inputs: bool = True,
        disable_tqdm: bool = False,
        target_sum: float = 1e3,
        npz_path: Optional[str] = None,
    ) -> "scipy.sparse.csr_matrix":
        """
        Calculates gene attributions for the SCimilarity model using a specified method.

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
        from captum.attr import IntegratedGradients, DeepLift, Saliency
        from tqdm import tqdm
        from scipy.sparse import csr_matrix, save_npz, vstack

        # Mapping of method strings to Captum classes
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

        # Create the attribution object
        if method.lower() == "ixg":
            # Saliency does not have a multiply_by_inputs constructor parameter
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

            # Some methods require a baseline, some don't
            if method.lower() == "ixg":
                # Saliency (ixg) does not use a baseline
                attrs = attributor.attribute(
                    inputs=inputs_batch_gpu,
                    target=0,
                    additional_forward_args=additional_forward_args,
                )
                # Manually multiply by inputs if requested
                if multiply_by_inputs:
                    attrs = attrs * inputs_batch_gpu
            else:  # IntegratedGradients and DeepLift
                # These methods require a baseline
                attrs = attributor.attribute(
                    inputs=inputs_batch_gpu,
                    baselines=torch.zeros_like(inputs_batch_gpu),
                    target=0,
                    additional_forward_args=additional_forward_args,
                )

            # The final result is always the absolute value of the attribution
            attrs_list.append(csr_matrix(torch.abs(attrs).detach().cpu().numpy()))

        # Concatenate results and normalize using the utility function
        return utils.normalize_attribution_matrix(
            matrix=vstack(attrs_list), target_sum=target_sum, npz_path=npz_path
        )
