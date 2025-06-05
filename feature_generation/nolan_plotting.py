import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from anndata import AnnData
from utils import ObsHelper
import re

def _get_enrichment_matrix_in_adata(adata: AnnData, knn=None, kmeans=None):
    match = False
    if knn is not None:
        if kmeans is not None:
            match = f"nolan_CN_enrichment_matrix_Km{kmeans}_Knn{knn}"
        else:
            raise ValueError("Specify the Kmeans")
    else:
        if kmeans is not None:
            raise ValueError("Specify the KNN")
        else:
            pattern = r"^nolan_CN_enrichment_matrix_"
            keys = adata.uns.keys()
            for key in keys:
                if re.match(pattern, key): # Here it is
                    match = key
                    print(f"Got: {match}")
    
    if match:
        return adata.uns[match]

    else:
        return match

def _get_nolan_neighborhood_key_in_adata(adata: AnnData, knn=None, kmeans=None):
    match = False
    if knn is not None:
        if kmeans is not None:
            match = f"nolan_cellular_neighborhood_labels_Km{kmeans}_Knn{knn}"
        else:
            raise ValueError("Specify the Kmeans")
    else:
        if kmeans is not None:
            raise ValueError("Specify the KNN")
        else:
            pattern = r"^nolan_cellular_neighborhood_"
            keys = adata.obs.keys()
            for key in keys:
                if re.match(pattern, key): # Here it is
                    match = key
                    print(f"Got: {match}")
    
    if match:
        return adata.obs[match]
    
    else:
        return False

def plot_enrichment_scores(
    adata: AnnData,
    transpose=False,
    knn=None,
    kmeans=None,
    **kwargs
    ):
    """ Kwargs parsed to sns.clustermap"""
    matrix = _get_enrichment_matrix_in_adata(adata, knn, kmeans)

    if isinstance(matrix, bool) and not matrix:
        raise ValueError("No Nolan enrichment matrix found in adata.uns")

    if transpose:
        matrix = matrix.T

    cmg = sns.clustermap(
        matrix,
        **kwargs
        )
    
    return cmg

def plot_factor_frequencies(
    adata: AnnData,
    grouping: str, # This represents single points or samples (ideally single Images/ROI)
    factor: str, # This represents the X-axis; categorical/discrete -> Neighborhoods
    regressor: str, # This represents the nested categories, usually a response / regressor
    figsize=(15,8),
    alpha=0.15,
    dodge=0.3,
    **kwargs,
    ):
    """ General implementaiton of a strip + point plot for plotting frequencies"""
    # Get the data
    if pd.api.types.is_numeric_dtype(adata.obs[outer_factor]):
        # create a copy and cast to category
        adata = adata.copy()
        adata.obs[outer_factor] = adata.obs[outer_factor].astype("category")
    
    sample_helper = ObsHelper(adata, grouping)
    outer_factor_proportions_by_sample = sample_helper.get_metadata_df(
        outer_factor, skey_handle="category_proportions")
    
    sample_to_inner_factor = sample_helper.get_metadata_df(inner_factor) # get the mapping from sample to inner fac

    proportions_master = pd.concat(
        [sample_to_inner_factor, outer_factor_proportions_by_sample], 
        axis=1)
    
    melted = proportions_master.reset_index().melt(
        id_vars = [grouping, inner_factor],
        var_name = f"{outer_factor}",
        value_name = "Proportion"
    )

    fig, ax = plt.subplots(figsize=figsize) # TODO: param
    sns.stripplot(
        data=melted,
        x=outer_factor,
        y="Proportion",
        hue=inner_factor,
        ax=ax,
        alpha=alpha,
        dodge=dodge,
        legend=False,
        **kwargs
        )
    sns.pointplot(
        data=melted,
        x=outer_factor,
        y="Proportion",
        hue=inner_factor,
        ax=ax,
        dodge=0.3,
        linestyle="none",
        **kwargs
        )
    
    return fig, ax

def plot_neighborhood_frequencies(
        adata: AnnData, 
        grouping: str,
        inner_factor: str,
        knn=None,
        kmeans=None):
    neighborhood_key = _get_nolan_neighborhood_key_in_adata(adata, knn, kmeans)

    if isinstance(neighborhood_key, bool) and not neighborhood_key:
        raise ValueError("No Nolan neighborhood labels found in adata.obs")
    
    return plot_factor_frequencies(
        adata, grouping, neighborhood_key, inner_factor)


def plot_clusters(
        adata: AnnData,
    ):
    
    fig, ax = plt.subplots(ncols=2, figsize=(10, 8))
    ax[0].scatter(adata)
        
    return fig, ax