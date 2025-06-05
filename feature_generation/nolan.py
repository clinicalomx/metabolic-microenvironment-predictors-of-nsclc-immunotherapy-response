""" Package for performing neighborhood analyses. """
from networkx import adjacency_graph

""" Imports """
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns
from typing import List, Tuple
from typing_extensions import Annotated
from anndata import AnnData
import typer
import utils
from scipy.stats import ttest_ind
import statsmodels.api as sm
from utils import ObsHelper
from collections import defaultdict, deque
from scipy.spatial import Delaunay, delaunay_plot_2d

app_nolan = typer.Typer()

# TODO: remove
from time import time
def timer_func(func): 
    # This function shows the execution time of  
    # the function object passed 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func 


NOLAN_KEY = "nolan"
CN_KEY = "cellular_neighborhoods"
NE_KEY = "neighborhoods_enrichment"

def _get_necessary_data(
        adata: AnnData,
        x_coordinate: str,
        y_coordinate: str,
        z_coordinate: str,
        phenotype: str,
        grouping: str
    ):
    """ Get the coordinate data from the AnnData object. """
    if z_coordinate is not None:
        return adata.obs[[x_coordinate, y_coordinate, z_coordinate, phenotype, grouping]]
    else:
        return adata.obs[[x_coordinate, y_coordinate, phenotype, grouping]]

def _get_neighborhoods_from_job(
        job, 
        region_groupby, 
        knn, 
        x_coordinate: str,
        y_coordinate: str,
        z_coordinate: str):
        """ For a given job (i.e. a given region/image in the dataset), return the indices of the nearest neighbors, for each cell in that job.
        Called by process_jobs.
            
            Params:
                job (str): Metadata containing start time, index of reigon, region name, indices of reigon in original dataframe.
                n_neighbors (str): Number of neighbors to find for each cell. 
            
            Returns:
                neighbors (numpy.ndarray): Array of indices where each row corresponds to a cell, and values correspond to indices of the nearest neighbours found for that cell. Sorted.
        """
        # Unpack job metadata file
        _, region, indices = job
        region = region_groupby.get_group(region)

        if z_coordinate is None:
            coords = [x_coordinate, y_coordinate]
        else:
            coords = [x_coordinate, y_coordinate, z_coordinate]

        X_data = region.loc[indices][coords].values
        # Perform sklearn unsupervised nearest neighbors learning, on the x y coordinate values
        # Essentially does euclidean metric; but technically Minkowski, with p=2
        knn += 1 # Account for self-computation in sklearn implementation
        neighbors = NearestNeighbors(n_neighbors=knn).fit(X_data)
        # Unpack results
        distances, indices = neighbors.kneighbors(X_data)
        distances = distances[:, 1:] # Remove self-computation
        indices = indices[:, 1:] # Remove self-computation
        sorted_neighbors = _sort_neighbors(region, distances, indices)
        return sorted_neighbors.astype(np.int32)

def _sort_neighbors(region, distances, indices):
    """ Processes the two outputs of sklearn NearestNeighbors to sort indices of nearest neighbors. """
    # Sort neighbors
    args = distances.argsort(axis = 1)
    add = np.arange(indices.shape[0])*indices.shape[1]
    sorted_indices = indices.flatten()[args+add[:,None]]
    neighbors = region.index.values[sorted_indices]
    return neighbors

@timer_func
@app_nolan.command()
def cellular_neighborhoods(
        adata: Annotated[AnnData, utils.adata_annotation()],
        grouping: Annotated[str, utils.grouping_annotation()],
        x_coordinate: Annotated[str, utils.x_coordinate_annotation()] = "x",
        y_coordinate: Annotated[str, utils.y_coordinate_annotation()] = "y",
        z_coordinate: Annotated[str, utils.z_coordinate_annotation()] = None, # TODO: z-coordinate support?
        knn: Annotated[int, utils.knn_annotation()] = 10,
        kmeans: Annotated[int, utils.kmeans_annotation()] = 10,
        phenotype: Annotated[str, utils.phenotype_annotation()] = None,
        inplace: Annotated[bool, utils.inplace_annotation()] = True,
        sortByPhenotype = None,
        sortByPhenotypeSelection = None,
    ):
    # 1) Generate jobs
    # Extract coordinate, phenotype, and grouping data
    data = _get_necessary_data(
        adata, x_coordinate, y_coordinate, z_coordinate, grouping, phenotype)

    index_map = dict(zip(list(data.index), list(data.reset_index().index)))
    data = data.reset_index()
    
    # One-hot encode data
    data = pd.concat([data, pd.get_dummies(data[phenotype])], axis=1) # Dummifie the cluster column into one-hot encoding

    # Transpose/Flatten phenotype data
    phenotypes = data[phenotype].unique()
    data_summed_phenotype = data[phenotypes].values # Tranposed + One-hot encoded phenotypes

    # Groupby object, groupby regions
    region_groupby = data.groupby(grouping, observed=False)

    # All the different regions
    regions = list(data[grouping].unique())

    # 1) Generate jobs. Indices are the dataframe indices for the given tissue/region
    jobs = [
        (regions.index(t),t,a) for t, indices in region_groupby.groups.items() \
        for a in np.array_split(indices,1)
        ]
    
    # 2) Process jobs
    processed_neighbors = list()
    for job in jobs:
        processed_neighbors.append(
            _get_neighborhoods_from_job(
                job, 
                region_groupby, 
                knn, 
                x_coordinate, 
                y_coordinate, 
                z_coordinate
                )
            )

    # 3) Annotate jobs
    out_dict = {}
    for neighbors, job in zip(processed_neighbors, jobs):
        chunk = np.arange(len(neighbors))
        region_name = job[1]
        indices = job[2]
        within = neighbors[chunk, :knn] #query up to knn neighbors
        window = data_summed_phenotype[within.flatten()]
        window = window.reshape(len(chunk), knn, len(phenotypes)).sum(axis = 1)
        out_dict[region_name] = (window.astype(np.float16), indices)

    region_dfs = [
        pd.DataFrame(
            out_dict[region_name][0],
            index = out_dict[region_name][1].astype(int),
            columns = phenotypes) 
        for region_name in regions  
        ]
    
    window = pd.concat(region_dfs, axis=0)
    window = window.loc[data.index.values]
    if z_coordinate is None:
        keep_cols = [x_coordinate, y_coordinate, phenotype]
    else:
        keep_cols = [x_coordinate, y_coordinate, z_coordinate, phenotype]
    window = pd.concat([data[keep_cols], window], axis=1) # Except grouping

    # Then perform knn on window
    kmeans_neighborhoods = MiniBatchKMeans(n_clusters=kmeans, n_init=knn, random_state=0, batch_size = 2560) #Changed by Aaron
    X_data = window[phenotypes]#.values
    kmeans_neighborhoods_labels = kmeans_neighborhoods.fit_predict(X_data.values)

    def relabelUsingTumourFractions(_labels, _phenotypes, _sortby):
        fracsByGroup, indexByGroup = [], []
        for group in set(_labels):
            fracs = np.sum(_phenotypes[_labels == group] == _sortby) / len(_phenotypes[_labels == group])
            fracsByGroup.append(fracs)
            indexByGroup.append(group)
        sortedFracsByGroup, sortedIndexByGroup = zip(*sorted(zip(fracsByGroup, indexByGroup)))
        mapping = {i: ind for i, ind in enumerate(sortedIndexByGroup)}
        _labels = np.array([mapping[lab] for lab in _labels])
        return _labels

    kmeans_neighborhoods_labels = relabelUsingTumourFractions(kmeans_neighborhoods_labels, adata.obs[sortByPhenotype], sortByPhenotypeSelection)
    
    # Compute enrichment scores;
    phenotype_distance_to_CN_centroids = kmeans_neighborhoods.cluster_centers_
    phenotype_frequencies = data_summed_phenotype.mean(axis=0) # Frequency of cell type across entire dataset. == Cells that are phenotype X / Total Cells
    num = phenotype_distance_to_CN_centroids + phenotype_frequencies # n neighbors rows, phenotype cols
    norm = (phenotype_distance_to_CN_centroids + phenotype_frequencies).sum(axis=1, keepdims=True) 
    score = np.log2(num/norm/phenotype_frequencies)
    score_df = pd.DataFrame(score, columns=phenotypes)

    phenotypesToSort = np.array(adata.obs[sortByPhenotype])

    results = {
        "labels": kmeans_neighborhoods_labels,
        "inertia": kmeans_neighborhoods.inertia_,
        "enrichment_matrix": score_df
    }
    # Annotate assignments back to AnnData using a tree-like structure
    # with a nested dicts format; indexed by the data it shows
    # Splits in nodes by different parameters
    if inplace:
        adata.obsm[f"nolan_{CN_KEY}_input_X"] = X_data.values # Save the X matrix used for kmeans

        if NOLAN_KEY not in adata.uns:
            adata.uns[NOLAN_KEY] = dict()

        if CN_KEY not in adata.uns[NOLAN_KEY]:
            adata.uns[NOLAN_KEY][CN_KEY] = dict()

        uns_node = adata.uns[NOLAN_KEY][CN_KEY]
        
        for k, v in results.items():
            node = _annotate_tree(
                start_node=uns_node,
                node_label_A=k,
                node_label_B=(grouping, phenotype),
                data=dict())
            
            _annotate_tree(
                start_node=node,
                node_label_A=(knn, kmeans),
                data=v)

    else:
        return results

def _annotate_tree(
        start_node,
        data,
        node_label_A=None,
        node_label_B=None,
    ):
    """ Creates or annotates a leaf with forks from a start_node, then annotates
        the tuple-keyed node with data. Non-recursive to enforce explicit node
        traversals.

        If node_label_A and node_label_B is supplied:

                        start_node
                        ||      ||
                    node_label_A  etc
                    ||        ||
               node_label_B  etc
                    ||
                   data

        If node_label_A is supplied but not node_label_B, annotates
        data.
                        start_node
                        ||       ||
                   node_label_A  etc
                        ||
                       data

        Vice-versa if node_label_A is not supplied:
                        start_node
                        ||       ||
                  node_label_B  etc
                        ||
                       data
        
        If no node_labels supplied it will annotate the start_node with data 
        directly:
                        start_node
                        ||      ||
                       data     etc

        If data is a dict, then the node reference at data is returned.
    """
    node = start_node
    
    # Keep track of parent and key for updating in place
    parent = None
    key = None
    
    if node_label_A:
        if node_label_A not in node:
            node[node_label_A] = dict()
        parent, key = node, node_label_A
        node = node[node_label_A]  # Terminate at node_label
        
        if node_label_B:
            if node_label_B not in node:
                node[node_label_B] = dict()
            parent, key = node, node_label_B
            node = node[node_label_B]  # Terminate at node_label -> p1/p2
        
    else:
        if node_label_B:  # no node_label, but p1 and p2
            if node_label_B not in node:
                node[node_label_B] = dict()
            parent, key = node, node_label_B
            node = node[node_label_B]
    
    # Update the parent node's key with data
    if parent is not None:
        if isinstance(data, dict):
            parent[key].update(data)
        else:
            parent[key] = data
    else:
        if isinstance(data, dict):
            start_node.update(data)
        else:
            start_node['value'] = data

    if isinstance(node, dict):
        return node

def _check_cn_run_exists_in_adata_od(
        adataod,
        grouping,
        phenotype,
        k,
        n,
        label
    ):
    if NOLAN_KEY not in adataod:
        raise KeyError("Run cellular_neighborhoods first")

    str_builder = f"No cellular_neighborhoods data for {label} found"
    if label not in adataod[NOLAN_KEY][CN_KEY]:
        raise KeyError(str_builder)

    if (grouping, phenotype) not in adataod[NOLAN_KEY][CN_KEY][label]:
        str_builder += f" with grouping = {grouping}, phenotype = {phenotype}"
        raise KeyError(str_builder)
    
    if (k, n) not in adataod[NOLAN_KEY][CN_KEY][label][(grouping, phenotype)]:
        str_builder += f" for K = {k}, number of neighborhoods = {n}"
        raise KeyError(str_builder)

#TODO; below can be single function + enums
def _get_cellular_neighborhoods_labels(
        adata,
        grouping,
        phenotype,
        k, # For Knn
        n # For kmeans ncluster or n neighborhoods
    ):
    _check_cn_run_exists_in_adata_od(
        adata.uns, grouping, phenotype, k, n, "labels")
    return adata.uns[NOLAN_KEY][CN_KEY]["labels"][(grouping, phenotype)][(k, n)]

def _get_cellular_neighborhoods_inertia(
        adata,
        grouping,
        phenotype,
        k, # For Knn
        n # For kmeans ncluster or n neighborhoods
    ):
    _check_cn_run_exists_in_adata_od(
        adata.uns, grouping, phenotype, k, n, "inertia")
    return adata.uns[NOLAN_KEY][CN_KEY]["inertia"][(grouping, phenotype)][(k, n)]

def _get_cellular_neighborhoods_enrichment_matrix(
        adata,
        grouping,
        phenotype,
        k, # For Knn
        n # For kmeans ncluster or n neighborhoods
    ):
    _check_cn_run_exists_in_adata_od(
        adata.uns, grouping, phenotype, k, n, "enrichment_matrix")
    return adata.uns[NOLAN_KEY][CN_KEY]["enrichment_matrix"][(grouping, phenotype)][(k, n)]

## Below with regards to endog variables + neighborhoods

def _normalise_log2p(X, pseudocount=1e-3):
    """ Given a df, re-normalise dfm apply log2p with pseudocount. """
    X = X.div(X.sum(axis=1), axis=0) # Normalise rows to add to 1
    return X.map(lambda x: np.log2(x + pseudocount)) # Apply log2p transformations to every value

def _build_design_matrix(Y_cond, X_cond, G, neighborhood, ct):
    """ Builds a design matrix based on multiple conditional probabilities, 
        given a `neighborhood` and `ct`,
        
        Y_cond -> Y(sample, n, ct | n = neighborhood, ct = ct)
        X_cond -> X(ct | ct = ct)
        G -> G(s | s = s)

        Return dataframe of columns: [Y_cond, X_cond, G]
        with rows as samples with non-na values for all columns (rows or sample
        with missing values are omitted from design matrix).

        """
    Y_sub = Y_cond.xs(neighborhood, level=0)[ct]
    Y_sub_remove_nan = Y_sub.dropna() # --> Drop Nas (don't include samples where we don't have proportions for a given CT in a given neighborhood)
    Y_sub_remove_nan.name = f"Y(sample, ct, n | ct = {ct}, n = {neighborhood})"

    common_index = Y_sub_remove_nan.index
    # Get X(ct = CT), which are the transformed CT frequencies per sample
    X_sub = X_cond.loc[common_index, ct]
    X_sub.name = f"X(sample, ct | ct = {ct})"

    # Get G(sample)
    G_sub = G.loc[common_index]
    # Get F(sample) --> all are 1..? NOTE: this maybe redundant then
    F_sub = pd.Series(np.ones(len(G_sub)))
    F_sub.index = common_index
    F_sub.name = "F"

    # contrsuct design matrix for logging
    design_df = pd.concat([Y_sub_remove_nan, X_sub, G_sub, F_sub], axis=1)

    return design_df

def _consolidate_statistics(results, rownames, colnames):
    """ Given the output results from the linear model loops, organise into a matrix:
        Neighborhoods as rows,
        Phenotypes as columns,
        values as p-values / coefficients. 
        
        All are respective to 1/0
        """
    index = pd.MultiIndex.from_tuples(
        results.keys(), 
        names=[colnames, rownames]
    )
    df = pd.DataFrame(list(results.values()), index=index).unstack(level=0)
    df.columns = df.columns.droplevel(0)
    return df

def _check_binary_response(adata, response):
    if isinstance(adata.obs[response]):
        pass
    pass

def _factor_proportions_by_response(
        adata,
        grouping,
        factor, # Neighborhood key
        response, # response key
        *,
        inplace=True):
    """ Computes the proportions (or factors) per sample (grouping) of a given
        factor, between response. Used for neighborhood proportions between
        responses. """
    if pd.api.types.is_numeric_dtype(adata.obs[factor]):
        # create a copy and cast to category
        adata = adata.copy()
        adata.obs[factor] = adata.obs[factor].astype("category")
    
    sample_helper = ObsHelper(adata, grouping)
    outer_factor_proportions_by_sample = sample_helper.get_metadata_df(
        factor, skey_handle="category_proportions")
    
    sample_to_inner_factor = sample_helper.get_metadata_df(response) # get the mapping from sample to inner fac

    proportions_master = pd.concat(
        [sample_to_inner_factor, outer_factor_proportions_by_sample], 
        axis=1)
    
    if inplace:
        adata.uns[f"{factor}_proportions_by_{grouping}_between_{response}"]
    else:
        return proportions_master #

def _test_factor_proportions_by_response(
    adata,
    grouping,
    factor, # Neighborhood key
    response, # response key
    *,
    inplace=True):
    
    proportions_master = _factor_proportions_by_response(
        adata, grouping, factor, response, inplace=False)
    
    ttest_results = []
    unique_factors = adata.obs[factor].unique()
    for n in unique_factors:
        group0 = proportions_master[proportions_master[response] == 0.0][n]
        group1 = proportions_master[proportions_master[response] == 1.0][n]
        t_stat, p_value = ttest_ind(group0, group1, equal_var=False) # Welch's t-test
        ttest_results.append((n, t_stat, p_value))
    ttest_results_df = pd.DataFrame(
        ttest_results, columns=['base_celltypes', 't_stat', 'p_value'])
    
    null_hypothesis, adjusted_pvalues, _, _ = \
        sm.stats.multipletests(ttest_results_df["p_value"].values)
    ttest_results_df["adjusted_pvalues"] = adjusted_pvalues
    ttest_results_df["null_hypothesis"] = null_hypothesis

    if inplace:
        adata.uns[f"test_{factor}_proportions_by_{grouping}_between_{response}"]
    else:
        return ttest_results_df

def dom_factor_proportions_by_response(
    adata,
    grouping,
    factor, # Neighborhood key
    response, # response key
    *,
    inplace=True): 

    proportions_master = _factor_proportions_by_response(
        adata, grouping, factor, response, inplace=False
    )

    means = proportions_master.groupby(response).mean().T
    stds = proportions_master.groupby(response).std().T
    n = proportions_master.groupby(response).count().T
    ci = 1.96 * (stds / np.sqrt(n))
    mean_diff = means[1.0] - means[0.0]
    ci_diff = np.sqrt(ci[1.0]**2 + ci[0.0]**2)

    if inplace:
        adata.uns[f"mean_difference_of_{factor}_proportions_by_{response}"]
    
    else: 
        return mean_diff, ci_diff

def neighborhood_enrichment(
    adata,
    grouping, # The column in .obs denoting replicates; usually tma cores or patients
    phenotype, # This should the column by which neighborhood was computed on 
    knn,
    kmeans,
    response, # The metadata to check enrichment of neighborhoods between
    *,
    pseudocount=1e-3, # Pseudocount for log2p transformations performed on the data
    inplace=True
    ):
    neighborhood_labels = _get_cellular_neighborhoods_labels(
        adata, grouping, phenotype, knn, kmeans)

    local_nhood_name = "_temp_neighborhood_col"
    adata.obs[local_nhood_name] = neighborhood_labels

    # Categories to hold
    unique_celltypes = adata.obs[phenotype].unique()
    unique_neighborhoods = adata.obs[local_nhood_name].unique()

    # Memory intensive for now but organised;
    agg_helper = utils.ObsHelper(adata, [local_nhood_name, grouping])
    sample_helper = utils.ObsHelper(adata, grouping)

    # Y response; map samples to their response (1:1 or N:1) cardinality enforced with obshelper
    sample_to_response = sample_helper.get_metadata_df(response)

    # Main computations
    # Computes Pnc 2d matrix
    # Neighborhoods in rows, Cell types as columns
    # Values are proportion of a cell type composing a given neighborhood
    # i.e.) 0.1 -> 10% of neighborhood Z is composed of cell type A 
    ct_props_by_sample = sample_helper.get_metadata_df(
        phenotype,
        skey_handle="category_proportions") # already drops na

    # Computes Psnc 3d tensor
    # samples as z (outer row with pandas), neighborhoods as rows, cts in columns
    ct_props_by_neighborhood_and_sample = agg_helper.get_metadata_df(
        phenotype,
        skey_handle="category_proportions")
    
    X_sample_log2 = _normalise_log2p(ct_props_by_sample, pseudocount)
    X_sample_neighborhood_log2 = ct_props_by_neighborhood_and_sample.groupby(
        level=0, 
        group_keys=False).apply(_normalise_log2p) # -> Equivalent to X_cond_nb. Keep Nans

    # Design matrix and linear model
    design_matrices = {}
    p_values = {}
    coefficients = {}
    t_values = {}

    for ct in unique_celltypes:
        for n in unique_neighborhoods:
            design_df = _build_design_matrix(
                X_sample_neighborhood_log2, 
                X_sample_log2, 
                sample_to_response, 
                n, 
                ct)
            
            Y = design_df.iloc[:, 0]
            X = design_df.iloc[:, 1:]
            regress = sm.OLS(Y, X).fit()
            p_values[(ct, n)] = regress.pvalues[response]
            coefficients[(ct, n)] = regress.params[response] 
            t_values[(ct, n)] = regress.tvalues[response]
            design_matrices[(ct, n)] = design_df
    
    p_values_df = _consolidate_statistics(
        p_values, 
        local_nhood_name, 
        phenotype
    )

    coefficients_df = _consolidate_statistics(
        coefficients,
        local_nhood_name,
        phenotype
    )

    t_values_df = _consolidate_statistics(
        t_values,
        local_nhood_name,
        phenotype
    )

    # Multiple hypotheses correspond to the different CTs across different Ns
    # n_hypotheses = n_cts * n_ns
    # Can use sm wrapper to correct p_values
    # _ are a_sidak, a_bonferonni (alpha values)
    null_hypothesis, adjusted_pvalues, _, _ = \
        sm.stats.multipletests(
            p_values_df.values.flatten(), 
            method="bonferroni")

    null_hypothesis_df = pd.DataFrame(
        null_hypothesis.reshape(p_values_df.shape))
    null_hypothesis_df.columns = p_values_df.columns
    null_hypothesis_df.index = p_values_df.index

    adjusted_pvalues_df = pd.DataFrame(
        adjusted_pvalues.reshape(p_values_df.shape))
    adjusted_pvalues_df.columns = p_values_df.columns
    adjusted_pvalues_df.index = p_values_df.index

    results = {
        "p_values": p_values_df,
        "adjusted_p_values": adjusted_pvalues_df,
        "reject_null_hypothesis": null_hypothesis_df,
        "coefficients": coefficients_df,
        "t_values": t_values_df
    }
    # NOTE; delete temp col
    adata.obs = adata.obs.drop(local_nhood_name, axis=1)
    
    if inplace:
        if NOLAN_KEY not in adata.uns:
            adata.uns[NOLAN_KEY] = dict()
        
        if NE_KEY not in adata.uns[NOLAN_KEY]:
            adata.uns[NOLAN_KEY][NE_KEY] = dict()
        
        uns_node = adata.uns[NOLAN_KEY][NE_KEY]

        for k, v in results.items():
            node = _annotate_tree(
                start_node=uns_node,
                node_label_A=k,
                node_label_B=(grouping, phenotype),
                data=dict())
            
            _annotate_tree(
                start_node=node,
                node_label_A=(knn, kmeans),
                data=v)
        
    else:
        return results

def label_subclusters(points, labels):
    """ Label contiguous subclusters based on Delaunay triangulation and label matching.

        Args:
        - points: List of 2D points [(x1, y1), (x2, y2), ...]
        - labels: List of class labels [c1, c2, ...]

        Returns:
        - subclusters: List of tuples (subcluster_id, subcluster_points)
    """
    adjacency_graph = build_delaunay_graph(points, labels)
    visited = set()  # Track visited points
    subclusters = []  # List to store each subcluster
    cluster_id = 0  # ID for subclusters

    for i in range(len(points)):
        if i not in visited:
            # Perform BFS to get all connected points (same subcluster)
            subcluster = bfs_traverse(adjacency_graph, i, visited)
            if subcluster:
                subclusters.append((cluster_id, subcluster))  # Store subcluster with ID
                cluster_id += 1

    return subclusters


def bfs_traverse(adjacency_graph, start, visited):
    """ Perform BFS to find all points in the same subcluster.

        Args:
        - adjacency_graph: The adjacency graph
        - start: Starting point index for BFS
        - visited: Set of already visited nodes

        Returns:
        - subcluster: List of point indices in the same subcluster
    """
    queue = deque([start])
    visited.add(start)
    subcluster = [start]

    while queue:
        node = queue.popleft()
        for neighbor in adjacency_graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                subcluster.append(neighbor)

    return subcluster


def build_delaunay_graph(points, labels, plot=False):
    """ Build an adjacency graph based on Delaunay triangulation and label matching.

        Args:
        - points: List of 2D points [(x1, y1), (x2, y2), ...]
        - labels: List of class labels corresponding to points [c1, c2, ...]

        Returns:
        - adjacency_graph: Graph represented as an adjacency list (dict)
    """
    if len(points) < 3:
        adjacency_graph = {}
        for i in range(len(labels)):
            for j in range(len(labels)):
                adjacency_graph[i] = [j]
                adjacency_graph[j] = [i]
                return adjacency_graph

    points = np.array(points)
    tri = Delaunay(points)
    delaunay_plot_2d(tri)

    adjacency_graph = defaultdict(list)

    # Iterate over the simplices (triangles) in the Delaunay triangulation
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                p1, p2 = simplex[i], simplex[j]

                # Add an edge if the labels are the same
                if labels[p1] == labels[p2]:
                    adjacency_graph[p1].append(p2)
                    adjacency_graph[p2].append(p1)

    return adjacency_graph


def returnProj(xs_, ys_, xcs_, ycs_):
    A_ = np.array([xcs_[0], ycs_[0]])
    B_ = np.array([xcs_[1], ycs_[1]])
    C_ = np.array([xcs_[2], ycs_[2]])
    Ab_, Bb_, Cb_ = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, 1.0])

    projA_ = []
    for x_, y_ in zip(xs_[0], ys_[0]):
        lambda_A, lambda_B, lambda_C = barycentric_coords(np.array([x_, y_]), A_, B_, C_)
        P_projected = lambda_A * Ab_ + lambda_B * Bb_ + lambda_C * Cb_
        if lambda_A + lambda_B + lambda_C <= 1:
            projA_.append(P_projected)

    projB_ = []
    for x_, y_ in zip(xs_[1], ys_[1]):
        lambda_A, lambda_B, lambda_C = barycentric_coords(np.array([x_, y_]), A_, B_, C_)
        P_projected = lambda_A * Ab_ + lambda_B * Bb_ + lambda_C * Cb_
        if lambda_A + lambda_B + lambda_C <= 1:
            projB_.append(P_projected)

    projC_ = []
    for x_, y_ in zip(xs_[2], ys_[2]):
        lambda_A, lambda_B, lambda_C = barycentric_coords(np.array([x_, y_]), A_, B_, C_)
        P_projected = lambda_A * Ab_ + lambda_B * Bb_ + lambda_C * Cb_
        if lambda_A + lambda_B + lambda_C <= 1:
            projC_.append(P_projected)

    return np.array(projA_), np.array(projB_), np.array(projC_)


# Function to compute Barycentric coordinates
def barycentric_coords(P, A, B, C):
    # Calculate the total area of the triangle ABC
    area_ABC = triangle_area(A, B, C)

    # Calculate the areas of the sub-triangles
    area_PBC = triangle_area(P, B, C)
    area_PAC = triangle_area(P, A, C)
    area_PAB = triangle_area(P, A, B)

    # Barycentric coordinates
    if area_ABC > 0.0:
        lambda_A = area_PBC / area_ABC
        lambda_B = area_PAC / area_ABC
        lambda_C = area_PAB / area_ABC
    else:
        return 0.0, 0.0, 0.0

    return lambda_A, lambda_B, lambda_C

def triangle_area(A, B, C):
    return 0.5 * abs(A[0]*(B[1] - C[1]) + B[0]*(C[1] - A[1]) + C[0]*(A[1] - B[1]))
