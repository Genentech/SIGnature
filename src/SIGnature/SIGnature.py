from typing import Dict, List, Optional, Tuple, Union, Set


class SIGnature:
    """A class for working with single cell gene attributions.
 
    Parameters
    ----------
    gene_order: list
        The gene order for the model.
    model: nn.Module
        A pytorch model.
    attributions_tiledb_uri: str
        The URI to the attributions tiledb.
    use_gpu: bool, default: False
        Use GPU instead of CPU.

    Examples
    --------
    >>> sig = SIGnature(model=scim.model, gene_order=scim.gene_order)
    """

    def __init__(
        self,
        gene_order: list,
        model: Optional["nn.Module"] = None,
        attribution_tiledb_uri: Optional[str] = None,
        use_gpu: bool = False,
    ):
        assert (model is not None) or (
            attribution_tiledb_uri is not None
        ), "Must instantiate with model for generating or tiledb for querying"
        self.gene_order = gene_order
        self.model = model
        self.attributions_tdb_uri = attribution_tiledb_uri
        self.n_genes = len(self.gene_order)
        self.use_gpu = use_gpu
        if self.use_gpu is True:
            self.model.cuda()

    def check_genes(
        self,
        gene_list: list,
        print_missing: bool = True,
    ) -> list:
        """Checks genes and returns ones usable by query.

        Parameters
        ----------
        gene_list: list
            A list of genes of interest.
        print_missing: bool
            Print the genes not in the model's gene order.

        Returns
        -------
        list
            A list of usable genes.

        Examples
        --------
        >>> gene_list = sig.check_genes(gene_list)
        """

        output_gene_list = [x for x in gene_list if x in self.gene_order]
        if print_missing and len(output_gene_list) != len(gene_list):
            missing_genes = set(gene_list).difference(output_gene_list)
            print(f"The following genes are not included: {','.join(missing_genes)}")

        return output_gene_list

    def calculate_attributions(
        self,
        X: Union["torch.Tensor", "numpy.ndarray", "scipy.sparse.csr_matrix"],
        buffer_size: int = 1000,
        target_sum: float = 1e3,
        disable_tqdm: bool = False,
        npz_path: Optional[str] = None
    ) -> "scipy.sparse.csr_matrix":
        """Calculate gene attributions from a log normalized expression matrix.

        Parameters
        ----------
        X: torch.Tensor, numpy.ndarray, scipy.sparse.csr_matrix
            Log normalized expression matrix.
        buffer_size: int, default: 1000
            Buffer size for batches.
        target_sum: float, default: 1000
            Target sum for attribution normalization.
        disable_tqdm: bool, default: False
            Disable the tqdm progress bar.
        npz_path: Optional[str], default: None
            Filename for storing the attribution matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            A sparse matrix of normalized gene attributions.

        Examples
        --------
        >>> attr = sig.calculate_attributions(X=adata.X, buffer_size=500)
        """

        from captum.attr import IntegratedGradients
        import numpy as np
        from scipy.sparse import csr_matrix, diags, save_npz
        from tqdm import tqdm
        import torch


        assert (X.shape[1] == len(self.gene_order)), "Expression matrix must have same number of features as model"

        attrs_list = []
        num_cells = X.shape[0]
        for i in tqdm(range(0, num_cells, buffer_size), disable=disable_tqdm):
            torch.cuda.empty_cache()
            X_subset = X[i : i + buffer_size, :]
            if isinstance(X_subset, np.ndarray):
                X_subset = torch.Tensor(X_subset)
            elif isinstance(X_subset, csr_matrix):
                X_subset = torch.Tensor(X_subset.todense())
            if next(self.model.parameters()).is_cuda:
                X_subset = X_subset.cuda()

            
            weights = self.model(X_subset)

            # Wrapper function around the model that takes the sum of the absolute value of the "weights" multiplied by the function.
            # Converts multidimensional embedding into single output for IG
            def model_forward(X_subset):
                n_repeat = int(len(X_subset) / len(weights))
                return torch.sum(
                    torch.abs(weights.repeat(n_repeat, 1) * self.model(X_subset)), dim=1
                )

            ig = IntegratedGradients(model_forward)
            attrs = torch.abs(ig.attribute(X_subset)).detach()
            if next(self.model.parameters()).is_cuda:
                attrs = attrs.cpu()
            attrs_list.append(attrs)
            X_subset = X_subset.detach()

        attrs = csr_matrix(torch.vstack(attrs_list).numpy())

        # normalize to target sum
        row_sums = attrs.sum(axis=1).A1  # .A1 converts to a 1D array
        row_sums[row_sums == 0] = 1
        inv_row_sums = diags(1 / row_sums).tocsr()
        normalized_matrix = inv_row_sums.dot(attrs)
        attrs = normalized_matrix.multiply(target_sum)

        if npz_path is not None:
            save_npz(file=npz_path, matrix=attrs)

        return attrs

    def create_tiledb(
        self,
        npz_path: str,
        batch_size: int = 25000,
        attribution_tiledb_uri: Optional[str] = None,
        overwrite: bool = False,
    ):
        """Create a sparse TileDB array from attribution matrix.

        Parameters
        ----------
        npz_path: str
            Filename for the stored attribution matrix.
        batch_size: int, default: 10000
            Batch size for the tiles.
        attributions_tiledb_uri: str
            The URI to the attributions tiledb.
        overwrite: bool, default: False
            Overwrite the existing TileDB.

        Examples
        --------
        >>> sig.create_tiledb(npz_path="/opt/npz_attribution_matrices/data.npz")
        """

        import numpy as np
        import os
        from scipy.sparse import load_npz
        import shutil
        import tiledb
        from .utils import write_csr_to_tiledb, optimize_tiledb_array

        if attribution_tiledb_uri is not None:
            self.attributions_tdb_uri = attribution_tiledb_uri

        xdimtype = np.uint32
        ydimtype = np.uint32
        value_type = np.float32

        matrix = load_npz(npz_path).tocsr()

        xdim = tiledb.Dim(name="x", domain=(0, matrix.shape[0] - 1), dtype=xdimtype)
        ydim = tiledb.Dim(name="y", domain=(0, matrix.shape[1] - 1), dtype=ydimtype)
        dom = tiledb.Domain(xdim, ydim)

        attr = tiledb.Attr(
            name="vals",
            dtype=value_type,
            filters=tiledb.FilterList([tiledb.GzipFilter()]),
        )

        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            cell_order="col-major",
            tile_order="col-major",
            attrs=[attr],
        )

        if os.path.exists(self.attributions_tdb_uri):
            if overwrite:
                shutil.rmtree(self.attributions_tdb_uri)
            else:
                print(f"TileDB already exists: {self.attributions_tdb_uri}")
                return

        tiledb.SparseArray.create(self.attributions_tdb_uri, schema)

        tdbfile = tiledb.open(self.attributions_tdb_uri, "w")
        write_csr_to_tiledb(tdbfile, matrix, value_type, 0, batch_size)
        tdbfile.close()
        optimize_tiledb_array(self.attributions_tdb_uri)

    def query_attributions(
        self,
        gene_list: List[str],
        cell_indices: Optional[List[int]] = None,
        attribution_tiledb_uri: Optional[str] = None,
        return_aggregate: bool = True,
        aggregate_type: str = "mean",
        weights: Optional[List[float]] = None,
    ):
        """Get attributions from sparse TileDB array.

        Parameters
        ----------
        gene_list: List[str]
            List of gene symbols.
        cell_indices: Optional[List[int]], default: None
            List of cell indices.
        attributions_tiledb_uri: str, optional
            The URI to the attributions tiledb.
        return_aggregate: bool, default: True
            Return an aggregate of attributions for each cell.
        aggregate_type: str, default: "mean"
            Type of aggregation {"mean", "sum"}.
        weights: Optional[List[float]], default: None
            Weights for each gene when calculating weighted averages or sums.
            Must have the same length as gene_list.

        Examples
        --------
        >>> cleaned_genes = sig.check_genes(genes)
        >>> sig.query_attributions(gene_list=cleaned_genes, attributions_tiledb_uri=tiledb_path)
        """

        import numpy as np
        from scipy.sparse import coo_matrix, diags
        import tiledb

        if attribution_tiledb_uri is not None:
            self.attributions_tdb_uri = attribution_tiledb_uri

        if len(set(gene_list)) != len(gene_list):
            raise RuntimeError(
                "There are duplicate genes in gene_list. Please remove duplicates."
            )

        if len(set(gene_list).intersection(self.gene_order)) != len(gene_list):
            raise RuntimeError(
                "Not all genes are included in the model. Please run sig.check_genes(gene_list)"
            )


        # Validate weights if provided
        if weights is not None:
            # Convert weights to numpy array if it's a list
            weights = np.array(weights, dtype=float)

            # Check if weights length matches gene_list length
            if len(weights) != len(gene_list):
                raise ValueError(
                    f"Length of weights ({len(weights)}) must match length of gene_list ({len(gene_list)})"
                )

        tdbfile = tiledb.open(self.attributions_tdb_uri, "r")
        n_cells = tdbfile.nonempty_domain()[0][1] + 1

        gene_indices = []
        for x in gene_list:
            gene_indices.append(self.gene_order.index(x))

        attr = tdbfile.schema.attr(0).name
        if cell_indices is None:
            results = tdbfile.multi_index[:, gene_indices]
            matrix = coo_matrix(
                (results[attr], (results["x"], results["y"])),
                shape=(n_cells, max(gene_indices) + 1),
            ).tocsr()
        else:
            results = tdbfile.multi_index[cell_indices, gene_indices]
            matrix = coo_matrix(
                (results[attr], (results["x"], results["y"])),
                shape=(max(cell_indices) + 1, max(gene_indices) + 1),
            ).tocsr()
            matrix = matrix[cell_indices, :]
        matrix = matrix[:, gene_indices]

        if return_aggregate:
            if weights is not None:
                # Create a diagonal matrix of weights for efficient multiplication
                weight_diag = diags(
                    weights, 0, shape=(len(gene_indices), len(gene_indices))
                )

                # Multiply matrix by weights efficiently using sparse matrix multiplication
                weighted_matrix = matrix.dot(weight_diag)

                if aggregate_type == "mean":
                    # For weighted mean, sum the weighted values and divide by sum of weights
                    return weighted_matrix.sum(axis=1) / np.sum(weights)
                elif aggregate_type == "sum":
                    # For weighted sum, just sum the weighted values
                    return weighted_matrix.sum(axis=1)
            else:
                # Original behavior for unweighted aggregation
                if aggregate_type == "mean":
                    return matrix.mean(axis=1)
                elif aggregate_type == "sum":
                    return matrix.sum(axis=1)

        return matrix
