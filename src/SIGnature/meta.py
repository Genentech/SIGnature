from typing import Any, Optional, TypeVar, Union, Tuple

from .utils import (
    subset_by_unique_values,
    subset_by_frequency,
    categorize_and_sort_by_score,
    title_name
)


class Meta:
    """A class for working with cell metadata files and output scores
    
    Parameters
    ----------
    df: pandas.Dataframe
        input dataframe to consider

    Examples
    --------
    >>> meta_df = pd.read_csv(meta_path)
    >>> meta = Meta(meta_df)
    """

    def __init__(self, df: "pandas.DataFrame"):
        self.df = df

    def append(self, df):
        """Append dataframe to current meta object

        Examples
        --------
        >>> meta.append(df)
        """
        import pandas as pd

        self.df = pd.concat([self.df, df])
    
    def columns(self) -> list:
        """Return columns of current dataframe

        Examples
        --------
        >>> columns = meta.columns()
        """
        return self.df.columns.tolist()

    def copy(self):
        """Return copy of current meta object

        Examples
        --------
        >>> new_meta = meta.copy()
        """
        return Meta(self.df.copy())

    def ncell(self) -> int:
        """Returns number of cells in current meta

        Examples
        --------
        >>> ncell = meta.ncell()
        """
        return self.df.shape[0]


    def add_hits(
        self,
        column_name: str,
        mode: str = "percentile",
        cut_val: Union[float, int] = 95.0,
        hit_type: str = "above",
        string_append: str = "__hit",
    ):
        """Add hits above percentile or quantile in a numerical column.

        Parameters
        ----------
        column_name: str
            Name of column used for cut-offs.
        mode: str, default: "percentile"
            Mode in one of "percentile", "quantile", or "value" to calculate cut-offs.
        cut_val: Union[float, int], default: 95.0
            The cut-off value used by percentile or quantile function on the column of interest.
        hit_type: str, default: "above"
            Hit type in one of "above" for hits above cut-off or "below" for hits below.
        string_append: str, default: "__hit"
            String to append to original category name to denote hits.

        Examples
        --------
        >>> meta.add_hits(column_name="GLI2")
        """

        import numpy as np

        assert mode in [
            "percentile",
            "quantile",
            "value",
        ], "Must choose percentile or quantile for mode"
        assert hit_type in ["above", "below"], "Must choose above or below for hit type"
        assert column_name in self.df.columns, "Column must be in dataframe"

        vals = self.df[column_name].values
        if mode == "quantile":
            assert (
                0 <= cut_val <= 1
            ), "For quantile mode, cut-off value must be between 0 and 1"
            cutoff = np.quantile(vals, cut_val)
        elif mode == "percentile":
            assert (
                0 <= cut_val <= 100
            ), "For percentle mode, cut-off value must be between 0 and 100"
            cutoff = np.percentile(vals, cut_val)
        else:
            cutoff = cut_val

        if hit_type == "above":
            self.df[f"{column_name}{string_append}"] = vals > cutoff
        else:
            self.df[f"{column_name}{string_append}"] = vals < cutoff

    def add_scores(self, score_dict: dict, mode: str = "value"):
        """Add attribution scores to metadata. Accounts for .npz sparse files
        (common for attributions) or numpy arrays.

        Parameters
        ----------
        score_dict: dict
            Dictionary where keys are category names and values are either scores or file locations.
        mode: str, default: "value"
            Mode in one of "value" or "file" for how to load data.

        Examples
        --------
        >>> score_dict = {"GLI1": "/data/GLI1_score.npz", "GLI2": "/data/GLI2_score.npy"}
        >>> meta.add_scores_by_file(score_dict, mode='file')
        """

        import numpy as np
        from scipy import sparse

        assert mode in ["value", "file"], "Mode must be value or file"
        if mode == "value":  # for value mode, append values to meta file
            for name, score in score_dict.items():
                if score.shape[0] != self.df.shape[0]:
                    print(
                        f"{name} could not be processed because the values are not the same shape as the current dataframe"
                    )
                    continue

                if isinstance(score, np.ndarray):
                    self.df[name] = score
                elif sparse.isspmatrix(score):
                    self.df[name] = score.toarray()
                else:
                    print(
                        f"{name} could not be processed because value is not a numpy array or a scipy sparse matrix"
                    )
        elif mode == "file":  # for file mode, load the scipy or numpy file and then add
            for name, file in score_dict.items():
                if ".npz" in file:
                    score = sparse.load_npz(file)
                    score = score.toarray()
                elif ".npy" in file:
                    score = np.load(file)
                else:
                    print(f"{file} does not end with .npz or .npy")
                if (
                    score.shape[0] != self.df.shape[0]
                ):  # check that score is same shape as dataframe
                    print(
                        f"{name} could not be processed because the values are not the same shape as the current dataframe"
                    )
                else:
                    self.df[name] = score

    def cat_by_min(
        self,
        column_name: str = "prediction",
        mode: str = "percent",
        cut_val: float = 1.0,
    ) -> list:
        """Get a list of categories with at least X% of a column

        Parameters
        ----------
        column_name: str
            Name of column used for cut-offs.
        mode:
            Mode in "percent" to at percent of all cells or "count" to look at minimum count
        cut_val:
            Cut-off percent or count value to be used.

        Returns
        -------
        list
            A list of valid categories in column.

        Examples
        --------
        >>> cats = meta.cat_by_min(column_name="prediction", mode="count", cut_val=50)
        """

        assert column_name in self.df.columns, "Category must be in meta's columns."
        assert mode in ["percent", "count"], "Mode must be percent or count"

        val_counts = self.df[column_name].value_counts()
        if mode == "percent":
            cut_val = self.ncell() * (cut_val / 100)

        return val_counts[val_counts >= cut_val].index.tolist()

    def samphit_df(
        self,
        cell_min: int = 50,
        samp_min: int = 3,
        samp_groupby: list = ["sample"],
        acats: list = ["tissue", "disease", "study", "sample"],
        dropna: bool = True,
        hit_col: str = "Hit Percentage",
        num_dis: Optional[int] = 15,
    ) -> "pandas.DataFrame":
        """Manipulate dataframe to calculate percentage of hits per sample

        Parameters
        ----------
        cell_min: int
            minimum number of cells per sample to be considered
        samp_min: int
            minimum number of qualifying samples for disease to be considered.
        samp_groupby: list
            groupby to consider for minimum number of cells
        acats: list
            annotation categories that user cares about for plotting
        dropna: bool
            drop all diseases named NA
        hit_col: str
            hit column name to consider
        num_dis: Optional[int]
            number of diseases to include in chart

        Returns
        -------
        pandas.DataFrame
            A sample level dataframe.

        Examples
        --------
        >>> df = meta.samphit_df(num_dis=10)
        """

        # subset to samples that contain cell minimum
        sdf = subset_by_frequency(self.df, samp_groupby, cell_min)
        # subset to diseases that contain at least minimun sample num
        sdf = subset_by_unique_values(sdf, "disease", "sample", samp_min)

        if dropna:
            sdf = sdf[sdf["disease"] != "NA"]

        # calculate hit percentage
        sdf_samps = (
            (sdf.groupby(acats)[hit_col].sum() / sdf.groupby(acats).size())
            .reset_index()
            .set_axis(acats + [hit_col], axis=1)
        )

        sdf_samps["disease"] = [title_name(x) for x in sdf_samps["disease"]]
        sdf_samps = categorize_and_sort_by_score(
            sdf_samps, name_column="disease", score_column=hit_col, topn=num_dis
        )

        return sdf_samps

    def samphit_boxplot(
        df: "pandas.DataFrame",
        title: str = "Hits Across Diseases per Sample",
        hit_label: str = "Hits",
        swarm: bool = False,
        title_fs: Union[int, float] = 16,
        dotsize: Union[int, float] = 3,
        fe: Union[int, float] = 1,
        figsize: Tuple = (6, 4),
        filename: Optional[str] = None,
    ):
        """Plots boxplot and swarmplot for disease
        
        Parameters
        ----------
        df: pandas.DataFrame
            A pandas dataframe that contains sample where each row
            has a sample labeled by its "disease" and "Hit Percentage"
            that indicates what proportion of cells are hits.
        title: str, default: "Hits Across Diseases per Sample"
            Plot title
        hit_label: str, default: "Hits"
            label for what to consider hits
        swarm: bool, default: False
            whether to include a swarmplot on top as well or no
        title_fs: Union[int, float], default: 16
            font size for the title
        dotsize: Union[int, float], default: 3,
            dot size for swarmplot
        fe: Union[int, float], default: 1
            scaling factor for various sizes
        figsize: Tuple, default: (6,4)
            figure size
        filename: Optional[str], default: None
            file name if want to save file
        
        Examples
        --------
        >>> Meta.samphit_boxplot(df=samphit_df)
        """

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(
            data=df, y="disease", x=hit_label, palette="Reds_r", ax=ax, showfliers=False
        )
        if swarm == True:
            sns.swarmplot(
                data=df,
                y="disease",
                x=hit_label,
                ax=ax,
                color="black",
                size=dotsize,
                linewidth=0.5 * fe,
                edgecolor="yellow",
            )
        ax.set_title(title, fontsize=title_fs * fe)

        vals = ax.get_xticks()
        ax.set_xticklabels(["{:,.0%}".format(x) for x in vals], fontsize=8 * fe)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8 * fe, wrap=True)
        ax.set_xlabel(
            "Percentage of Hits per Sample".format(hit_label), fontsize=12 * fe
        )
        ax.set_ylabel("")
        fig.tight_layout(pad=1)

        if filename is not None:
            plt.savefig(filename, dpi=300)

    def subset_hq(
        self,
        cutoff: float = 0.02,
        quality_column: str = "prediction_nn_dist",
        mode: str = "below",
    ):
        """Subset to cells below prediction dist cut-off

        Parameters
        ----------
        cutoff: float, default: 0.02
            Cut-off to use for quality.
        quality_column: str, default: "prediction_nn_dist"
            Column used to score quality metric.
        mode: str, default: "below"
            Mode in "below" for when you want lower and "above" for higher.

        Examples
        --------
        >>> meta.subset_hq()
        """
        assert mode in ["above", "below"], "Mode must be above or below"

        if mode == "below":
            self.df = self.df[self.df[quality_column] <= cutoff]
        else:
            self.df = self.df[self.df[quality_column] >= cutoff]

    def subset_invivo(
        self, column_name: str = "in_vivo", in_vivo_val: Union[bool, str] = True
    ):
        """Subset to in vivo cells using standard SCimilarity columns.

        Examples
        --------
        >>> meta.subset_invivo()
        """

        assert (
            column_name in self.df.columns
        ), f"Column {column_name} must be in dataframe"
        self.df = self.df[self.df[column_name] == in_vivo_val]

    def top_cells(
        self, ncell: int, column_name: str, return_df: bool = True
    ) -> "pandas.DataFrame":
        """Get dataframe including only top cells by category.

        Paramaters
        ----------
        column_name: str
            Name of column to consider.
        ncell: int
            Number of cells to keep.
        return_df: bool, default: True
            Return a dataframe instead of modifying the class attribute.

        Returns
        -------
        pandas.DataFrame
            A dataframe of top cells.

        Examples
        --------
        meta_top100 = meta.top_cells(100, 'GLI2', return_df=True)
        """

        assert column_name in self.df.columns, "Category must be in dataframe"
        if return_df:
            return self.df.nlargest(ncell, column_name)
        else:
            self.df = self.df.nlargest(ncell, column_name)
            return None
