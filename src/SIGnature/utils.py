from typing import Optional, Union, Tuple, List


def align_dataset(
    data: "anndata.AnnData",
    target_gene_order: list,
    keep_obsm: bool = True,
    gene_overlap_threshold: int = 5000,
) -> "anndata.AnnData":
    """Align the gene space to the target gene order.

    Parameters
    ----------
    data: anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    target_gene_order: list
        A list containing the gene space.
    keep_obsm: bool, default: True
        Retain the original data's obsm matrices in output.
    gene_overlap_threshold: int, default: 5000
        The minimum number of genes in common between data and target_gene_order to be valid.

    Returns
    -------
    anndata.AnnData
        A data object with aligned gene space ready to be used for embedding cells.

    Examples
    --------
    >>> data = align_dataset(data, gene_order)
    """

    import anndata
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix

    # raise an error if not enough genes from target_gene_order exists
    if sum(data.var.index.isin(target_gene_order)) < gene_overlap_threshold:
        raise RuntimeError(
            f"Dataset incompatible: gene overlap less than {gene_overlap_threshold}. Check that var.index uses gene symbols."
        )

    # check if X is dense, convert to csr_matrix if so
    if isinstance(data.X, np.ndarray):
        data.X = csr_matrix(data.X)

    # check for negatives in expression data
    if np.min(data.X) < 0:
        raise RuntimeError(f"Dataset contains negative values in expression matrix X.")

    # check if counts is dense, convert to csr_matrix if so
    if "counts" in data.layers and isinstance(data.layers["counts"], np.ndarray):
        data.layers["counts"] = csr_matrix(data.layers["counts"])

    # check for negatives in count data
    if "counts" in data.layers and np.min(data.layers["counts"]) < 0:
        raise RuntimeError(f"Dataset contains negative values in layers['counts'].")

    # return data if already aligned
    if data.var.index.values.tolist() == target_gene_order:
        return data

    orig_genes = data.var.index.values  # record original gene list before alignment
    shell = anndata.AnnData(
        X=csr_matrix((0, len(target_gene_order))),
        var=pd.DataFrame(index=target_gene_order),
    )
    shell = anndata.concat(
        (shell, data[:, data.var.index.isin(shell.var.index)]), join="outer"
    )
    shell.uns["orig_genes"] = orig_genes
    if not keep_obsm and hasattr(data, "obsm"):
        delattr(shell, "obsm")

    if data.var.shape[0] == 0:
        raise RuntimeError(f"Empty gene space detected.")

    return shell


def lognorm_counts(
    data: "anndata.AnnData",
) -> "anndata.AnnData":
    """Log normalize the gene expression raw counts (per 10k).

    Parameters
    ----------
    data: anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.

    Returns
    -------
    anndata.AnnData
        A data object with normalized data that is ready to be used in further processes.

    Examples
    --------
    >>> data = lognorm_counts(data)
    """

    import numpy as np
    import scanpy as sc

    if "counts" not in data.layers:
        raise ValueError(f"Raw counts matrix not found in layers['counts'].")

    data.X = data.layers["counts"].copy()

    # check for nan in expression data, zero
    if isinstance(data.X, np.ndarray) and np.isnan(data.X).any():
        import warnings

        warnings.warn(
            "NANs detected in counts. NANs will be zeroed before normalization in X.",
            UserWarning,
        )
        data.X = np.nan_to_num(data.X, nan=0.0)

    # log norm
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    del data.uns["log1p"]

    return data


def write_csr_to_tiledb(
    tdb: "tiledb.libtiledb.SparseArrayImpl",
    matrix: "scipy.sparse.csr_matrix",
    value_type: type,
    row_start: int = 0,
    batch_size: int = 25000,
):
    """Write csr_matrix to TileDB.

    Parameters
    ----------
    tdb: tiledb.libtiledb.SparseArrayImpl
        TileDB array.
    arr: numpy.ndarray
        Dense numpy array.
    value_type: type
        The type of the value, typically np.float32.
    row_start: int, default: 0
        The starting row in the TileDB array.
    batch_size: int, default: 100000
        Batch size for the tiles.
    """
    indptrs = matrix.indptr
    indices = matrix.indices
    data = matrix.data

    x = []
    y = []
    vals = []
    for i, indptr in enumerate(indptrs):
        if i != 0 and (i % batch_size == 0 or i == len(indptrs) - 1):
            tdb[x, y] = vals
            x = []
            y = []
            vals = []

        stop = None
        if i != len(indptrs) - 1:
            stop = indptrs[i + 1]

        val_slice = data[slice(indptr, stop)].astype(value_type)
        ind_slice = indices[slice(indptr, stop)]

        x.extend([row_start + i] * len(ind_slice))
        y.extend(ind_slice)
        vals.extend(val_slice)


def optimize_tiledb_array(
    tiledb_array_uri: str,
    steps=100000,
    step_max_frags: int = 10,
    buffer_size: int = 1000000000,  # 1GB
    total_budget: int = 200000000000,  # 200GB
    verbose: bool = True,
):
    """Optimize TileDB Array.

    Parameters
    ----------
    tiledb_array_uri: str
        URI for the TileDB array.
    verbose: bool
        Boolean indicating whether to use verbose printing.
    """

    import tiledb

    if verbose:
        print(f"Optimizing {tiledb_array_uri}")

    frags = tiledb.array_fragments(tiledb_array_uri)
    if verbose:
        print("Fragments before consolidation: {}".format(len(frags)))

    cfg = tiledb.Config()
    cfg["sm.consolidation.steps"] = steps
    cfg["sm.consolidation.step_min_frags"] = 2
    cfg["sm.consolidation.step_max_frags"] = step_max_frags
    cfg["sm.consolidation.buffer_size"] = buffer_size
    cfg["sm.mem.total_budget"] = total_budget
    tiledb.consolidate(tiledb_array_uri, config=cfg)
    tiledb.vacuum(tiledb_array_uri)

    frags = tiledb.array_fragments(tiledb_array_uri)
    if verbose:
        print("Fragments after consolidation: {}".format(len(frags)))


def subset_by_unique_values(
    df: "pandas.DataFrame",
    group_columns: Union[List[str], str],
    value_column: str,
    n: int,
) -> "pandas.DataFrame":
    """Subset a pandas dataframe to only include rows where there are at least
    n unique values from value_column, for each grouping of group_column.

    Parameters
    ----------
    df: "pandas.DataFrame"
        Pandas dataframe.
    group_columns: Union[List[str], str]
        Columns to group by.
    value_column: str
        Column value from which to check the number of instances.
    n: int
        Minimum number of values to be included.

    Returns
    -------
    pandas.DataFrame
        A subsetted dataframe.

    Examples
    --------
    >>> df = subset_by_unique_values(df, "disease", "sample", 10)
    """

    groups = df.groupby(group_columns)[value_column].transform("nunique") >= n

    return df[groups]


def subset_by_frequency(
    df: "pd.DataFrame",
    group_columns: Union[List[str], str],
    n: int,
) -> "pd.DataFrame":
    """Subset the DataFrame to only columns where the group appears at least n times.

    Parameters
    ----------
    df: "pandas.DataFrame"
        Pandas dataframe
    group_columns: Union[List[str], str]
        Columns to group by.
    n: int
        Minimum number of values to be included.

    Returns
    -------
    pandas.DataFrame
        A subsetted dataframe.

    Examples
    --------
    >>> df = subset_by_frequency(df, ["disease", "prediction"], 10)
    """

    freq = df.groupby(group_columns).size()
    hits = freq[freq >= n].index

    return df.set_index(group_columns).loc[hits].reset_index(drop=False)


def categorize_and_sort_by_score(
    df: "pandas.DataFrame",
    name_column: str,
    score_column: str,
    ascending: bool = False,
    topn: Optional[int] = None,
) -> "pandas.DataFrame":
    """Transform column into category, sort, and choose top n

    Parameters
    ----------
    df: "pandas.DataFrame"
        Pandas dataframe.
    name_column: str
        Name of column to sort.
    score_column: str
        Name of score column to sort name_column by.
    ascending: bool
        Sort ascending
    topn: Optional[int], default: None
        Subset to the top n diseases.

    Returns
    -------
    pandas.DataFrame
        A sorted dataframe that is optionally subsetted to top n.

    Examples
    --------
    >>> df = categorize_and_sort_by_score(df, "disease", "Hit Percentage", topn=10)
    """

    mean_scores = (
        df.groupby(name_column)[score_column].mean().sort_values(ascending=ascending)
    )
    df[name_column] = df[name_column].astype("category")
    df[name_column] = df[name_column].cat.set_categories(
        mean_scores.index, ordered=True
    )

    if topn is not None:
        top_values = mean_scores.head(topn).index
        df = df[df[name_column].isin(top_values)]
        # remove unused cats from df
        df[name_column] = df[name_column].cat.remove_unused_categories()

    return df.sort_values(name_column, ascending=ascending)

def title_name(s: str) -> str:
    """Return string if all upper case, otherwise return a title version.

    Examples
    --------
    >>> s = title_name('multiple myeloma') # outputs Multiple Myeloma
    """

    import re

    if s.isupper():
        return s
    elif "'" in s:  # deal with apostrophe case, common in diseases
        return re.sub(
            r"[A-Za-z]+('[A-Za-z]+)?",
            lambda mo: mo.group(0)[0].upper() + mo.group(0)[1:].lower(),
            s,
        )
    else:  # title function where we keep capital letters capitalized
        return re.sub(
            r"[A-Za-z]+('[A-Za-z]+)?",
            lambda mo: mo.group(0)[0].upper() + mo.group(0)[1:],
            s,
        )

