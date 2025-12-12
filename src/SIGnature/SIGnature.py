from typing import Dict, List, Optional, Tuple, Union, Set


class SIGnature:
    """A class for working with single cell gene attributions.

    Parameters
    ----------
    gene_order: list
        The gene order for the model.
    attributions_tiledb_uri: str
        The URI to the attributions tiledb.

    Examples
    --------
    >>> sig = SIGnature(gene_order=gene_order)
    """

    def __init__(
        self,
        gene_order: list,
        attribution_tiledb_uri: Optional[str] = None,
    ):
        self.gene_order = gene_order
        self.attributions_tdb_uri = attribution_tiledb_uri
        self.n_genes = len(self.gene_order)

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
            print(f"The following genes are not included: {', '.join(missing_genes)}")

        return output_gene_list

    def create_tiledb(
        self,
        npz_path: str,
        batch_size: int = 25000,
        attribution_tiledb_uri: Optional[str] = None,
        overwrite: bool = False,
        attr_name: str = "vals",
        xdim_name: str = "x",
        ydim_name: str = "y",
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
        attr_name: str, default: "vals"
            Name of the tiledb matrix value attribute.
        xdim_name: str, default: "x"
            Name of the tiledb matrix x dimensions.
        ydim_name: str, default: "y"
            Name of the tiledb matrix y dimensions.

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

        xdim = tiledb.Dim(
            name=xdim_name, domain=(0, matrix.shape[0] - 1), dtype=xdimtype
        )
        ydim = tiledb.Dim(
            name=ydim_name, domain=(0, matrix.shape[1] - 1), dtype=ydimtype
        )
        dom = tiledb.Domain(xdim, ydim)

        attr = tiledb.Attr(
            name=attr_name,
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

        cfg = tiledb.Config()
        cfg["sm.mem.total_budget"] = 50000000000  # 50G
        tdbfile = tiledb.open(self.attributions_tdb_uri, "w", config=cfg)
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
        attr_name: str = "vals",
        xdim_name: str = "x",
        ydim_name: str = "y",
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
        attr_name: str, default: "vals"
            Name of the tiledb matrix value attribute.
        xdim_name: str, default: "x"
            Name of the tiledb matrix x dimensions.
        ydim_name: str, default: "y"
            Name of the tiledb matrix y dimensions.

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

        cfg = tiledb.Config()
        cfg["sm.mem.total_budget"] = 50000000000  # 50G
        tdbfile = tiledb.open(self.attributions_tdb_uri, "r", config=cfg)
        n_cells = tdbfile.nonempty_domain()[0][1] + 1

        gene_indices = []
        for x in gene_list:
            gene_indices.append(self.gene_order.index(x))

        if cell_indices is None:
            results = tdbfile.multi_index[:, gene_indices]
            matrix = coo_matrix(
                (results[attr_name], (results[xdim_name], results[ydim_name])),
                shape=(n_cells, max(gene_indices) + 1),
            ).tocsr()
        else:
            results = tdbfile.multi_index[cell_indices, gene_indices]
            matrix = coo_matrix(
                (results[attr_name], (results[xdim_name], results[ydim_name])),
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
