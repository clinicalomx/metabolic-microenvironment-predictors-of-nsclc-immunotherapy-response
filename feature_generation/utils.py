import typer
import pandas as pd

# Help messages
ADATA_HELP = "AnnData object to compute scores for."
GROUPING_HELP = "Column in .obs to compute independent pdensities for. Ideally \
    should be the Image column (or tiles, where there is continuity for \
    distances)."
PAIRS_HELP = "Selection of phenotype paris in phenotype to compute pdensities \
    for."
RADIUS_HELP = "Search radius in image units for identifying cell neighbours \
    (interactions)."
KNN_HELP = "Number of nearest neighbors to use for identifying cell \
    neighbours (interactions)."
X_COORDINATE_HELP = "Column in .obs representing local x value of the cell \
    centroid position."
Y_COORDINATE_HELP = "Column in .obs representing local y value of the cell \
    centroid position."
Z_COORDINATE_HELP = "Column in .obs representing local z value of the cell \
    centroid position."
PHENOTYPE_HELP = "Column in .obs to perform label comparison of pdensity. This \
    is usually the cell type column."
METHOD_HELP = "How to compute pdensity. nodes computes using original \
    implementation. edges computes essentially the beta-index."
INPLACE_HELP = "Annotate the given adata inplace or not."
SPATIAL_SEARCH_METHODS_HELP = "Method to use for finding cell neighbors. \
    Options are: kdtree, balltree, brute, auto."
PERMUTATION_HELP = "Number of permutations to perform for computing pvalues."
PVAL_METHOD_HELP = "Method to use for computing pvalues. Options are: \
    `zscore`, `histocat`."
PVAL_CUTOFF_HELP = "Cutoff for defining a significant pvalue for the \
    permutation tests."

KMEANS_HELP = "Number of a priori clusters to get from kmeans clustering."

# # Constants
# SPATIAL_SEARCH_METHODS = ["kdtree", "balltree", "brute", "auto"]

def adata_annotation():
    """ Annotation for adata argument. """
    return typer.Option("--adata", "-a", help=ADATA_HELP)

def grouping_annotation():
    """ Annotation for grouping argument. """
    return typer.Option("--grouping", "-g", help=GROUPING_HELP)

def pairs_annotation():
    """ Annotation for pairs argument. """
    return typer.Option("--pairs", "-p", help=PAIRS_HELP)

def radius_annotation():
    """ Annotation for radius argument. """
    return typer.Option("--radius", "-r", help=RADIUS_HELP)

def knn_annotation():
    """ Annotation for knn argument. """
    return typer.Option("--knn", "-k", help=KNN_HELP)

def x_coordinate_annotation():
    """ Annotation for x_coordinate argument. """
    return typer.Option("--x_coordinate", "-x", help=X_COORDINATE_HELP)

def y_coordinate_annotation():
    """ Annotation for y_coordinate argument. """
    return typer.Option("--y_coordinate", "-y", help=Y_COORDINATE_HELP)

def z_coordinate_annotation():
    """ Annotation for z_coordinate argument. """
    return typer.Option("--z_coordinate", "-z", help=Z_COORDINATE_HELP)

def phenotype_annotation():
    """ Annotation for phenotype argument. """
    return typer.Option("--phenotype", "-p", help=PHENOTYPE_HELP)

def search_method_annotation():
    """ Annotation for search_method argument. """
    return typer.Option("--search_method", "-s", help=SPATIAL_SEARCH_METHODS_HELP)

def method_annotation():
    """ Annotation for method argument. """
    return typer.Option("--method", "-m", help=METHOD_HELP)

def permutation_annotation():
    """ Annotation for permutation argument. """
    return typer.Option("--permutation", "-p", help=PERMUTATION_HELP)

def pval_method_annotation():
    """ Annotation for pval argument. """
    return typer.Option("--pval", "-p", help=PVAL_METHOD_HELP)

def pval_cutoff_annotation():
    """ Annotation for pval_cutoff argument. """
    return typer.Option("--pval_cutoff", "-p", help=PVAL_CUTOFF_HELP)

def inplace_annotation():
    """ Annotation for inplace argument. """
    return typer.Option("--inplace", "-i", help=INPLACE_HELP)

def kmeans_annotation():
    """ Annotation for kmeans argument. """
    return typer.Option("--kmeans", "-k", help=KMEANS_HELP)

# Helpers functions
def cudf_accelerated_pandas():
    try:
        import cudf.pandas
        cudf.pandas.install()
        return True
    except ImportError:
        return False


class ObsHelper():
    """
    Obs Helper for retrieving metadata with common indexes relative to a 
    specified parent column.
    """
    def __init__(self, adata, base_column):
        self.adata = adata
        self.base_column = base_column
        self._set_groupby_df(base_column)
        self._set_column_relations()
        self._log_column_dtypes()
        
    def _set_groupby_df(self, base_column):
        self.groupby_df = self.adata.obs.groupby(base_column, observed=False)

    def _set_column_relations(self):
        """ Based on cardinality_df, will get keys which have a 1:1 relation. """
        cardinality_df = self.groupby_df.nunique()
        # Columns which core gorup has 1 or less (if there are NaNs) unique values;
        oto = cardinality_df.sum() <= cardinality_df.shape[0]
        self.parallel_keys = cardinality_df.columns[oto]
        self.super_keys = cardinality_df.columns[~oto] # These are probably your cell-level metadata

    def _log_column_dtypes(self):
        """ Log the data types of each key. """
        df = self.adata.obs
        categorical_dtypes = df.select_dtypes(exclude="number").columns
        numerical_dtypes = df.select_dtypes(include="number").columns
        # If the key is numerical AND a super key then its a true numeric which needs aggregation
        self.numerical_keys = pd.Index([x for x in numerical_dtypes if x in self.super_keys])

        # If the key is numerical but a parallel key then it can be treated like a categorical parallel key
        categorical_numerics = pd.Index([x for x in numerical_dtypes if x in self.parallel_keys])
        self.categorical_keys = df.select_dtypes(exclude="number").columns
        self.categorical_keys = self.categorical_keys.append(categorical_numerics)

    def get_metadata_df(self, column, *, skey_handle=None, aggregation_function=None, bins=None):
        groupby_obj = self.groupby_df

        def _get_parallel_key(groupby_obj, column):
            groupby_obj = groupby_obj[column]
            assert all(groupby_obj.nunique() <= 1)
            return groupby_obj.first()
        
        def _get_super_key(groupby_obj, column, skey_handle, aggregation_function, bins):

            # Directive A) Categorical;
            def _get_super_key_categorical(groupby_obj, column, skey_handle):
                # Directive 1: Rows = base, Columns = each category in column, Values = Counts of that category per base.
                if skey_handle in ["category_counts", "category_proportions"]:
                    vals = groupby_obj[column].value_counts().unstack(column)
                    if skey_handle == "category_proportions":
                        vals = vals.div(vals.sum(axis=1), axis=0)
                    return vals
                else:
                    raise ValueError("Unsupported skey handle for categorical superkey column")

            # Directive B) Numerical
            def _get_super_key_numerical(groupby_obj, column, skey_handle, aggregation_function, bins):
                # Sub-Directive B1) Numerical -> Categorical; Binning Agg -> Parsed to Directive A
                def _bin_numerics(groupby_obj, column, bins): # define bins as a list of nums defining boundaries; i.e. [-np.inf, -50, 0, 50, np.inf]
                    assert bins is not None
                    def _bin_and_count(groupby_obj, column, bins):
                        # Apply binning
                        binned = pd.cut(groupby_obj[column], bins=bins)
                        counts = binned.value_counts().reindex(pd.IntervalIndex.from_breaks(bins, closed='right'))
                        return counts
                    return groupby_obj.apply(_bin_and_count, column=column, bins=bins)

                # Sub-Directive B2) Numerical -> Summary per base. (i.e. mean dist {column} per unique_core {base})
                def _summarise_numerics(groupby_obj, column, aggregation_function):

                    def _get_aggregation_function(aggregation_function):
                        # Parse common aggregation functions which are str to pd.core.GroupBy callables
                        match aggregation_function: # Pass
                            case "sum":
                                return pd.core.groupby.DataFrameGroupBy.sum
                            case "max":
                                return pd.core.groupby.DataFrameGroupBy.max
                            case "min":
                                return pd.core.groupby.DataFrameGroupBy.min
                            case "first":
                                return pd.core.groupby.DataFrameGroupBy.first
                            case "last":
                                return pd.core.groupby.DataFrameGroupBy.last
                            case "mean":
                                return pd.core.groupby.DataFrameGroupBy.mean
                            case "median":
                                return pd.core.groupby.DataFrameGroupBy.median
                            case _:
                                raise ValueError(
                                    f"`aggregation_function` not supported or none provided. \n" \
                                    "Valid aggregation functions: \n\t" \
                                    "sum, max, min, first, last, mean, median."
                                    )
                    
                    agg_func = _get_aggregation_function(aggregation_function)
                    return agg_func(groupby_obj[column])
                
                # Sub-Directive B3) Numerical Widened -> Restricted to annotation boxplots/scatterplots etc.
                def _widen_numerics(groupby_obj, column):
                    grouped = groupby_obj[column].apply(list)
                    return pd.DataFrame(grouped.tolist(), index=grouped.index)

                # Handle numerical sub-directives
                if skey_handle == "summarise":
                    return _summarise_numerics(groupby_obj, column, aggregation_function)
                elif skey_handle == "bin":
                    return _bin_numerics(groupby_obj, column, bins)
                elif skey_handle == "widen":
                    return _widen_numerics(groupby_obj, column)
                else:
                    raise ValueError("Invalid skey_handling method for numerics.")                
                
            ## Apply appropriate directives
            if isinstance(column, list):
                if all([c for c in column if c in self.categorical_keys]):
                    return _get_super_key_categorical(groupby_obj, column, skey_handle)
                else: # theres a numeric; 
                    raise NotImplementedError("Mixed key cardinalities/dtypes not implemneted yet")
            else:
                if column in self.categorical_keys:
                    return _get_super_key_categorical(groupby_obj, column, skey_handle)
                else:
                    return _get_super_key_numerical(groupby_obj, column, skey_handle, aggregation_function, bins)
            
        # Parallel Keys
        if isinstance(column, list):
            # Assert for now that both at parallel keys
            if all([c for c in column if c in self.parallel_keys]):
                result = _get_super_key(groupby_obj, column, skey_handle, aggregation_function, bins) # INFS theory; if multiple primary keys-> it follows to become a superkey;
        else:
            if column in self.parallel_keys:
                result = _get_parallel_key(groupby_obj, column)
            else:
                result = _get_super_key(groupby_obj, column, skey_handle, aggregation_function, bins)

            if isinstance(result, pd.Series):
                result = pd.DataFrame(result)
        
        return result

    def get_metadata_column(self, metadata_df):
        # Mainly for getting the appropriate column metadata from a given metadata_df, with some aggregation usually;
        return pd.DataFrame(
            metadata_df.columns.to_list(), columns=[x+"_col" for x in metadata_df.columns.names], index=metadata_df.columns)