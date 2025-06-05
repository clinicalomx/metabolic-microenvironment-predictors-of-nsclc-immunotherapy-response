import gc
from nolan import *
import scanpy as sc
import nolan_plotting
import matplotlib.pyplot as plt

class ClusterPointsNolan:
    def __init__(self, adata, n, k,
                 grouping='Tissue_Subsample_ID',
                 phenotype='tumournontumour',
                 imageName=None
                 ):
        self.n = n #Number of clusters
        self.k = k #Number of neighbours to consider in Nolan clustering
        self.adata = adata
        self.grouping = grouping
        self.phenotype = phenotype
        self.imageName = imageName

        self.inertia = None
        self.labels = None
        self.enrichment_matrix = None

    def cluster(self):
        cellular_neighborhoods(
            self.adata,
            #self.adata[~np.isnan(self.adata.obs.immune_func)],
            grouping=self.grouping,
            # Column in adata.obs that defines different spatial contexts (i.e. different images or cores). Usually image, or parent column.
            x_coordinate="x",  # Column in adata.obs that defines your cell X coordinate
            y_coordinate="y",  # Above for y
            knn=self.n,  # How many surrounding cells to take into account (window size)
            kmeans=self.k,  # How many neighborhoods to compute
            phenotype=self.phenotype,  # Column in adata.obs that defines your cell types of phenotypes of interest
            inplace=True
        )  # Annotate and attach the result to the adata directly

        #self.adata.uns[f"nolan_cellular_neighborhood_inertia_Km{self.k}_Knn{self.n}"] =
        #self.adata.obs[f"nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}"] =
        #self.adata.uns[f"nolan_CN_enrichment_matrix_Km{self.k}_Knn{self.n}"] =
        self.inertia = self.adata.uns['nolan']['cellular_neighborhoods']['inertia']\
            [(self.grouping, self.phenotype)][(self.n, self.k)]
        self.labels = self.adata.uns['nolan']['cellular_neighborhoods']['labels']\
            [(self.grouping, self.phenotype)][(self.n, self.k)]
        self.enrichment_matrix = self.adata.uns['nolan']['cellular_neighborhoods']\
            ['enrichment_matrix'][(self.grouping, self.phenotype)][(self.n, self.k)]

    def printClusterInfo(self):
        clusterlabels = list(set(self.adata.obs[f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}']))
        print("Cluster labels: ", set(self.adata.obs[f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}']))
        tumournontumour, celltype = [], []
        for label in clusterlabels:
            tumournontumour.append(self.adata[self.adata.obs[f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}']\
                                 .isin([label])].obs['tumournontumour'].value_counts())
            celltype.append(self.adata[self.adata.obs[f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}']\
                                 .isin([label])].obs['immune_func'].value_counts())
        print("Cluster type tumour: ", tumournontumour)
        print("Cluster type immune: ", celltype)

    def returnModeOfClusterCellTypes(self, label, toExclude=['Vessels', 'Other']):
        vals = self.adata[self.adata.obs[f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}']\
                                 .isin([label])].obs['immune_func']
        vals = vals[~vals.isin(toExclude)]
        return vals.mode()

    def plot_clustering(self):

        sc.set_figure_params(format='png', dpi=80, dpi_save=300, figsize=(30, 30))
        fig = sc.pl.spatial(self.adata,
                      color=f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}',
                      spot_size=12, color_map="Paired",
                      layer='log1p_normalised', vmin='p10', vmax='p90',
                      size=1.5, alpha_img=0, return_fig=True, show=False)
        gc.collect()
        plt.savefig(r'nolan_'+self.imageName+f"k{self.k}n{self.n}"+'.png')
        plt.close()

    def plot_enrichment_scores(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        nolan_plotting.plot_enrichment_scores(
            self.adata,
            knn=self.k,
            kmeans=self.n,
            transpose=True,  # Swap x and y axis
            cmap="bwr",  # blue, white, red cmap. Best cmap for negative to positive scaling.
            vmin=-3,  # Enforce data to fal between -3 and 3
            vmax=3,
            figsize=(7, 5)  # aspect ratio
        )
        plt.savefig(r'enrichment_'+self.imageName+f"k{self.k}n{self.n}"+'.png')
        plt.close()

'''
import scipy
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, OPTICS
class ClusterPoints:
    def __init__(self, clusteringType, X, Y, plotFileName=None, numClusters=10):
        clusteringTypes = ['kmeans', 'dbscan', 'optics']
        self.clusteringType = clusteringType
        assert self.clusteringType in clusteringTypes
        self.numClusters = numClusters
        self.clusterObject = None
        self.X = X
        self.Y = Y
        self.minX, self.maxX = self.X.min(), self.X.max()
        self.minY, self.maxY = self.Y.min(), self.Y.max()
        self.labels = None
        self.plotFileName = plotFileName

        self.cluster()
        self.plot()

    def cluster(self):
        if self.clusteringType == 'kmeans':
            self.clusterObject = (KMeans(n_clusters=self.numClusters, random_state=0,
                                        n_init=50, max_iter=30000)
                                  .fit(np.vstack([self.X, self.Y]).T))
            self.labels = self.clusterObject.labels_
        elif self.clusteringType == 'dbscan':
            self.clusterObject = DBSCAN(eps=50, min_samples=10) \
                                  .fit(np.vstack([self.X, self.Y]).T)
            self.labels = self.clusterObject.labels_
        elif self.clusteringType == 'optics':
            self.labels = self.clusterObject =  OPTICS(
                min_samples=100,
                xi=50,
                min_cluster_size=100,
            ).fit(np.vstack([self.X, self.Y]).T)
        if hasattr(self.clusterObject, "labels_"):
            self.labels = self.clusterObject.labels_.astype(int)
        else:
            self.labels = self.clusterObject.predict(np.vstack([self.X, self.Y]).T)
        #self.labels = self.clusterObject.predict(np.vstack([self.X, self.Y]).T)

    def predict(self, X, Y):
        if self.clusteringType == 'kmeans':
            return self.clusterObject.predict(np.vstack([X, Y]).T)
        elif self.clusteringType == 'dbscan':
            return -1

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        X, Y = np.arange(self.minX, self.maxX, 1), np.arange(self.minY, self.maxY, 1)
        plt.scatter(self.Y, self.X, c=self.labels, s=2)
        if self.plotFileName:
            plt.savefig('clustering_'+self.plotFileName+'.png')
        plt.close()

class ClusterClinicalData:
    def __init__(self, adata, clusteringType='dbscan', target_reference_channel='tumournontumour',
                 target='tumour', reference='nontumour', plotFileName=None):
        self.clusteringType = clusteringType
        self.adata = adata

        if plotFileName:
            self.plotFileName = plotFileName
        else:
            self.plotFileName = clusteringType + '_' + target_reference_channel

        self.clusterPointsReference = ClusterPoints(self.clusteringType, self.adata[self.adata.obs[target_reference_channel].isin([reference])].obs.x,
                                           self.adata[self.adata.obs[target_reference_channel].isin([reference])].obs.y,
                                                    plotFileName=target+'_'+list(set(self.adata.obs['Tissue_Subsample_ID']))[0], numClusters=20)
        self.clusterPointsTarget = ClusterPoints(self.clusteringType, self.adata[self.adata.obs[target_reference_channel].isin([target])].obs.x,
                                           self.adata[self.adata.obs[target_reference_channel].isin([target])].obs.y,
                                                 plotFileName=reference+'_'+list(set(self.adata.obs['Tissue_Subsample_ID']))[0], numClusters=20)

    def plot(self):
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        ax[0].scatter(self.clusterPointsReference.X, self.clusterPointsReference.Y,
                      s=1, c=self.clusterPointsReference.labels)
        ax[1].scatter(self.clusterPointsTarget.X, self.clusterPointsTarget.Y,
                      s=1, c=self.clusterPointsTarget.labels)
        ax[2].scatter(self.clusterPointsReference.X, self.clusterPointsReference.Y,
                      s=1, c=self.clusterPointsReference.labels, cmap=cm.Reds)
        ax[2].scatter(self.clusterPointsTarget.X, self.clusterPointsTarget.Y,
                      s=1, c=self.clusterPointsTarget.labels, cmap=cm.Blues)
        plt.savefig(self.plotFileName+'.png')
        plt.close()'''