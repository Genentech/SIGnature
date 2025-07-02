import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


class Encoder(nn.Module):
    """A class that encapsulates the encoder."""

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


class SCimilarity:
    """A class loads the SCimilarity model."""

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

        self.int2label = pd.read_csv(
            os.path.join(self.model_path, "label_ints.csv"), index_col=0
        )["0"].to_dict()
        self.label2int = {value: key for key, value in self.int2label.items()}
