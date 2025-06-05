import gc, os
import itertools
from nolan import *
import scanpy as sc
import nolan_plotting
import matplotlib.pyplot as plt
import squidpy as sq


class NolanCNsAndSubClustering:
    def __init__(self, adata, n, k,
                 grouping='Tissue_Subsample_ID',
                 phenotype='tumournontumour',
                 imageName=None,
                 baseFolder=None,
                 patient=None
                 ):
        self.n = n  # Number of clusters
        self.k = k  # Number of neighbours to consider in Nolan clustering
        self.adata = adata
        self.grouping = grouping
        self.phenotype = phenotype
        #self.imageName = imageName
        #self.patient = patient

        self.inertia = None
        self.labels = None
        self.enrichment_matrix = None
        self.clusterlabels = {}
        self.baseCNVariableName = None
        self.baseSubclusterVariableName = None
        self.baseFolder = baseFolder

        #if not os.path.exists(self.baseFolder):
        #    os.makedirs(self.baseFolder)

    # Cluster (cells -> CNs)
    # Takes the x, y positions of the cells and applies Nolan clustering to produce CNs
    #
    def cluster(self, updateCNs=True):
        cellular_neighborhoods(
            self.adata,
            grouping=self.grouping,
            # Column in adata.obs that defines different spatial contexts (i.e. different images or cores). Usually image, or parent column.
            x_coordinate="x",  # Column in adata.obs that defines your cell X coordinate
            y_coordinate="y",  # Above for y
            knn=self.n,  # How many surrounding cells to take into account (window size)
            kmeans=self.k,  # How many neighborhoods to compute
            phenotype=self.phenotype,  # Column in adata.obs that defines your cell types of phenotypes of interest
            inplace=True,
            sortByPhenotype='celltypes',
            sortByPhenotypeSelection='tumorcells',
        )  # Annotate and attach the result to the adata directly
        # self.adata.uns[f"nolan_cellular_neighborhood_inertia_Km{self.k}_Knn{self.n}"] =
        # self.adata.obs[f"nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}"] =
        # self.adata.uns[f"nolan_CN_enrichment_matrix_Km{self.k}_Knn{self.n}"] =

        self.inertia = self.adata.uns['nolan']['cellular_neighborhoods']['inertia'] \
            [(self.grouping, self.phenotype)][(self.n, self.k)]
        self.labels = self.adata.uns['nolan']['cellular_neighborhoods']['labels'] \
            [(self.grouping, self.phenotype)][(self.n, self.k)]
        self.enrichment_matrix = self.adata.uns['nolan']['cellular_neighborhoods'] \
            ['enrichment_matrix'][(self.grouping, self.phenotype)][(self.n, self.k)]

        self.adata.uns[f'nolan_CN_enrichment_matrix_Km{self.k}_Knn{self.n}'] = self.enrichment_matrix
        self.adata.obsm['spatial'] = np.array(np.stack([self.adata.obs.x, self.adata.obs.y]).T)

        if updateCNs:
            self.updateWithCNIndex()

        #self.plot_enrichment_scores()
        #self.plot_clustering()


    def printClusterInfo(self):
        clusterlabels = list(set(self.adata.obs[f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}']))
        print("Cluster labels: ", set(self.adata.obs[f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}']))
        tumournontumour, celltype = [], []
        for label in clusterlabels:
            tumournontumour.append(
                self.adata[self.adata.obs[f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}'] \
                .isin([label])].obs['tumournontumour'].value_counts())
            celltype.append(self.adata[self.adata.obs[f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}'] \
                            .isin([label])].obs['immune_func'].value_counts())
        print("Cluster type tumour: ", tumournontumour)
        print("Cluster type immune: ", celltype)

    def returnModeOfClusterCellTypes(self, label, toExclude=['Vessels', 'Other']):
        vals = self.adata[self.adata.obs[f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}'] \
            .isin([label])].obs['immune_func']
        vals = vals[~vals.isin(toExclude)]
        return vals.mode()

    # Updates the overall data structure and the image data structure with the CN labels, inertia and enrichment matrix
    def updateWithCNIndex(self, tissueIdentifier='Tissue_Subsample_ID'):
        self.adata.obs \
           [f"nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}"] = self.labels
        self.adata.obs[f"nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}"] \
            = self.labels
        self.adata.uns[f"nolan_cellular_neighborhood_inertia_Km{self.k}_Knn{self.n}"] \
            = self.inertia
        self.adata.uns[f"nolan_CN_enrichment_matrix_Km{self.k}_Knn{self.n}"] \
            = self.enrichment_matrix
        return


    # Takes a cluster object and assigns the cell types to the cluster object using various selection criteria
    # TODO: make a cluster object and make it generic to the cluster method
    def assignCellTypesToCNs(self, phenotype='baseCTtumour', selection='mode'):
        assert selection in ['mode', 'top3']

        if not phenotype in self.adata.obs.keys():
            print(f"{phenotype} not in keys!")
            print(self.adata.obs.keys())
            return Exception
        clusterTypeNaming = f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}'
        self.clusterlabels[clusterTypeNaming] = list(set(self.adata.obs[clusterTypeNaming]))

        cluster_type_dict = {}
        if selection == 'mode':
            cluster_type_dict = {label: str(self.adata[self.adata.obs[clusterTypeNaming].isin([label])] \
                .obs[phenotype].mode().values[0]) for label in self.clusterlabels[clusterTypeNaming]
                if len(self.adata[self.adata.obs[clusterTypeNaming].isin([label])].obs[phenotype]) > 0}
        elif selection == 'top3':
            cluster_type_dict = {label: self.adata[self.adata.obs[clusterTypeNaming].isin([label])] \
                .obs[phenotype].value_counts().values[:2] for label in self.clusterlabels[clusterTypeNaming]
                if len(self.adata[self.adata.obs[clusterTypeNaming].isin([label])].obs[phenotype]) > 0}

        self.adata.obs[clusterTypeNaming.replace('labels', phenotype)] = self.adata.obs \
            [clusterTypeNaming].map(cluster_type_dict)#.astype('category')

        return

    def subclusterNolanCNs(self, adata):
        subclusters = label_subclusters([[x_, y_] for x_, y_ in zip(np.array(adata.obs.x),
                            np.array(adata.obs.y))], np.array(adata.obs[
                            f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}']))

        cell_cluster_index   = np.nan*np.empty(len(adata.obs.x), dtype=int)
        cell_cluster_centerX = np.nan*np.empty(len(adata.obs.x), dtype=float)
        cell_cluster_centerY = np.nan*np.empty(len(adata.obs.x), dtype=float)
        cell_cluster_types   = ['' for i in range(len(adata.obs.x))]

        for clusterIndex, (referenceCellIndex, targetCellIndex) in enumerate(subclusters):
            meanx = adata.obs.x[
                adata.obs.reset_index().index.astype(int).isin(targetCellIndex)].mean()
            meany = adata.obs.y[
                adata.obs.reset_index().index.astype(int).isin(targetCellIndex)].mean()
            modetype = np.array(adata.obs['celltypes']
                                [adata.obs.reset_index().index.astype(int).isin(targetCellIndex)].mode())

            modetype = modetype[0]

            for i in range(len(targetCellIndex)):
                cell_cluster_index[
                    targetCellIndex[i]] = clusterIndex  # clusterIndex+100000#referenceCellIndex
                cell_cluster_centerX[targetCellIndex[i]] = meanx
                cell_cluster_centerY[targetCellIndex[i]] = meany
                cell_cluster_types  [targetCellIndex[i]] = modetype

        adata.obs[f'nolan_cellular_neighborhood_subclusterindex_Km{self.k}_Knn{self.n}'] = cell_cluster_index
        adata.obs[f'nolan_cellular_neighborhood_subclustermeanX_Km{self.k}_Knn{self.n}'] = cell_cluster_centerX
        adata.obs[f'nolan_cellular_neighborhood_subclustermeanY_Km{self.k}_Knn{self.n}'] = cell_cluster_centerY
        adata.obs[f'nolan_cellular_neighborhood_subclustertype_Km{ self.k}_Knn{self.n}'] = cell_cluster_types

        return adata, f'nolan_cellular_neighborhood_subclusterindex_Km{self.k}_Knn{self.n}'

    #TODO: figure out what I wanted with clusterDelauneyNeighbors
    def fillNolanTriangles(self, clusterDelauneyNeighbors,
                           uniqueDelauneyNeighbors, nolan_barycentric_triangles=None, groupby=None):
        subclusterid = f'nolan_cellular_neighborhood_subclusterindex_Km{self.k}_Knn{self.n}'
        #clusterIds   = [a[subclusterid].iloc[0] for _, a in self.imageAData.obs.groupby(by=subclusterid)]
        clusterPosX  = [a.x.mean()      for _, a in self.adata.obs.groupby(by=subclusterid)]
        clusterPosY  = [a.y.mean()      for _, a in self.adata.obs.groupby(by=subclusterid)]
        clusterTypes = [a[f'nolan_cellular_neighborhood_subclustertype_Km{self.k}_Knn{self.n}'] \
                        .mode().iloc[0] for _, a in self.adata.obs.groupby(by=subclusterid)]

        # Get all triples of cell types
        cell_triples = list(itertools.combinations(sorted(list(set(clusterTypes))), 3))
        if not nolan_barycentric_triangles:
            nolan_barycentric_triangles = {triple: [[], [], [], []] for triple in cell_triples}#[[], [], [], []]

        for i in range(len(uniqueDelauneyNeighbors)):

            type1  = clusterTypes[uniqueDelauneyNeighbors[i][0]]
            type2  = clusterTypes[uniqueDelauneyNeighbors[i][1]]
            type3  = clusterTypes[uniqueDelauneyNeighbors[i][2]]
            if type1 == type2 or type2 == type3 or type1 == type3:
                continue
            if not tuple(sorted([type1, type2, type3])) in nolan_barycentric_triangles.keys():
                nolan_barycentric_triangles[tuple(sorted([type1, type2, type3]))] = [[], [], [], []] #[[], [], [], []]

            x1, y1 = self.adata[self.adata.obs[subclusterid] == uniqueDelauneyNeighbors[i][0]].obs.x, \
                     self.adata[self.adata.obs[subclusterid] == uniqueDelauneyNeighbors[i][0]].obs.y
            x2, y2 = self.adata[self.adata.obs[subclusterid] == uniqueDelauneyNeighbors[i][1]].obs.x, \
                     self.adata[self.adata.obs[subclusterid] == uniqueDelauneyNeighbors[i][1]].obs.y
            x3, y3 = self.adata[self.adata.obs[subclusterid] == uniqueDelauneyNeighbors[i][2]].obs.x, \
                     self.adata[self.adata.obs[subclusterid] == uniqueDelauneyNeighbors[i][2]].obs.y

            ind1 = np.where(type1 == np.array(sorted([type1, type2, type3])))[0][0]#2
            ind2 = np.where(type2 == np.array(sorted([type1, type2, type3])))[0][0]#0
            ind3 = np.where(type3 == np.array(sorted([type1, type2, type3])))[0][0]#1

            pA, pB, pC = returnProj([x1, x2, x3], [y1, y2, y3],
                                [clusterPosX[uniqueDelauneyNeighbors[i][0]],
                                      clusterPosX[uniqueDelauneyNeighbors[i][1]],
                                      clusterPosX[uniqueDelauneyNeighbors[i][2]]],
                                [clusterPosY[uniqueDelauneyNeighbors[i][0]],
                                      clusterPosY[uniqueDelauneyNeighbors[i][1]],
                                      clusterPosY[uniqueDelauneyNeighbors[i][2]]],
                                   )
            if pA.shape[0] > 0 and pB.shape[0] > 0 and pC.shape[0] > 0:
                nolan_barycentric_triangles[tuple(sorted([type1, type2, type3]))][ind1].extend(list(pA))
                nolan_barycentric_triangles[tuple(sorted([type1, type2, type3]))][ind2].extend(list(pB))
                nolan_barycentric_triangles[tuple(sorted([type1, type2, type3]))][ind3].extend(list(pC))
                if groupby:
                    nolan_barycentric_triangles[tuple(sorted([type1, type2, type3]))][3] = np.append(
                        nolan_barycentric_triangles[tuple(sorted([type1, type2, type3]))][3], self.adata.obs[groupby].iloc[0])

        print("NolanCNsAndSubclustering.fillNolanTriangles: ", nolan_barycentric_triangles)

        return nolan_barycentric_triangles

    def plotNolanTriangles(self, nolan_barycentric_triangles, groupby=[['PD'], ['CR', 'PR', 'NE', 'SD']]):
        for (type1, type2, type3) in nolan_barycentric_triangles.keys():
            print(nolan_barycentric_triangles[(type1, type2, type3)][3])
            if len(nolan_barycentric_triangles[(type1, type2, type3)][3]) < 50:
                continue

            fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
            if nolan_barycentric_triangles[(type1, type2, type3)][3][0] in groupby[0]:
                ax[0, 0].scatter(np.array(nolan_barycentric_triangles[(type1, type2, type3)][0])[:, 0],
                                 np.array(nolan_barycentric_triangles[(type1, type2, type3)][0])[:, 1], c='r', s=0.5)
                ax[0, 0].scatter(np.array(nolan_barycentric_triangles[(type1, type2, type3)][1])[:, 0],
                                 np.array(nolan_barycentric_triangles[(type1, type2, type3)][1])[:, 1], c='g', s=0.5)
                ax[0, 0].scatter(np.array(nolan_barycentric_triangles[(type1, type2, type3)][2])[:, 0],
                                 np.array(nolan_barycentric_triangles[(type1, type2, type3)][2])[:, 1], c='b', s=0.5)
                ax[0, 0].text(0.55, 0.95, type3, ha='left', wrap=True)
                ax[0, 0].text(0.0, -0.04, type1, ha='left', wrap=True)
                ax[0, 0].text(0.8, -0.04, type2, ha='left', wrap=True)
                if len(np.array(nolan_barycentric_triangles[(type1, type2, type3)][0])) > 0:
                    sns.kdeplot(x=np.array(nolan_barycentric_triangles[(type1, type2, type3)][0])[:, 0],
                                y=np.array(nolan_barycentric_triangles[(type1, type2, type3)][0])[:, 1], c='r', ax=ax[0, 1],
                                levels=[0.1, 0.3, 0.6, 0.9], cut=0)
                if len(np.array(nolan_barycentric_triangles[(type1, type2, type3)][1])) > 0:
                    sns.kdeplot(x=np.array(nolan_barycentric_triangles[(type1, type2, type3)][1])[:, 0],
                                y=np.array(nolan_barycentric_triangles[(type1, type2, type3)][1])[:, 1], c='g', ax=ax[0, 1],
                                levels=[0.1, 0.3, 0.6, 0.9], cut=0)
                if len(np.array(nolan_barycentric_triangles[(type1, type2, type3)][2])) > 0:
                    sns.kdeplot(x=np.array(nolan_barycentric_triangles[(type1, type2, type3)][2])[:, 0],
                                y=np.array(nolan_barycentric_triangles[(type1, type2, type3)][2])[:, 1], c='b', ax=ax[0, 1],
                                levels=[0.1, 0.3, 0.6, 0.9], cut=0)

            elif nolan_barycentric_triangles[(type1, type2, type3)][3][0] in groupby[1]:
                print("np.array(nolan_barycentric_triangles[(type1, type2, type3)][0]): ",
                      np.array(nolan_barycentric_triangles[(type1, type2, type3)][0]))
                ax[1, 0].scatter(np.array(nolan_barycentric_triangles[(type1, type2, type3)][0])[:, 0],
                                 np.array(nolan_barycentric_triangles[(type1, type2, type3)][0])[:, 1], c='r', s=0.5)
                ax[1, 0].scatter(np.array(nolan_barycentric_triangles[(type1, type2, type3)][1])[:, 0],
                                 np.array(nolan_barycentric_triangles[(type1, type2, type3)][1])[:, 1], c='g', s=0.5)
                ax[1, 0].scatter(np.array(nolan_barycentric_triangles[(type1, type2, type3)][2])[:, 0],
                                 np.array(nolan_barycentric_triangles[(type1, type2, type3)][2])[:, 1], c='b', s=0.5)
                ax[1, 0].text(0.55, 0.95, type3, ha='left', wrap=True)
                ax[1, 0].text(0.0, -0.04, type1, ha='left', wrap=True)
                ax[1, 0].text(0.8, -0.04, type2, ha='left', wrap=True)
                if len(np.array(nolan_barycentric_triangles[(type1, type2, type3)][0])) > 0:
                    sns.kdeplot(x=np.array(nolan_barycentric_triangles[(type1, type2, type3)][0])[:, 0],
                                y=np.array(nolan_barycentric_triangles[(type1, type2, type3)][0])[:, 1], c='r', ax=ax[1, 1],
                                levels=[0.1, 0.3, 0.6, 0.9], cut=0)
                if len(np.array(nolan_barycentric_triangles[(type1, type2, type3)][1])) > 0:
                    sns.kdeplot(x=np.array(nolan_barycentric_triangles[(type1, type2, type3)][1])[:, 0],
                                y=np.array(nolan_barycentric_triangles[(type1, type2, type3)][1])[:, 1], c='g', ax=ax[1, 1],
                                levels=[0.1, 0.3, 0.6, 0.9], cut=0)
                if len(np.array(nolan_barycentric_triangles[(type1, type2, type3)][2])) > 0:
                    sns.kdeplot(x=np.array(nolan_barycentric_triangles[(type1, type2, type3)][2])[:, 0],
                                y=np.array(nolan_barycentric_triangles[(type1, type2, type3)][2])[:, 1], c='b', ax=ax[1, 1],
                                levels=[0.1, 0.3, 0.6, 0.9], cut=0)
            ax[0, 0].set_title("Responded > 0")
            ax[1, 0].set_title("Responded == 0")
            # plt.savefig(os.path.join(r'triangles', type1+'_'+type2+'_'+type3+'.svg'))
            plt.show()
        return

    @staticmethod
    def getClusterNeighbours(adata, clusterLabel):
        #subclusterid = f'nolan_cellular_neighborhood_subclusterindex_Km{self.k}_Knn{self.n}'
        clusterIds   = [a[clusterLabel].iloc[0] for _, a in adata.obs.groupby(by=clusterLabel)]
        clusterPosX  = [a.x.mean()      for _, a in adata.obs.groupby(by=clusterLabel)]
        clusterPosY  = [a.y.mean()      for _, a in adata.obs.groupby(by=clusterLabel)]
        #clusterTypes = [a[f'nolan_cellular_neighborhood_subclustertype_Km{self.k}_Knn{self.n}'] \
        #                .iloc[0] for _, a in self.adata.obs.groupby(by=subclusterid)]
        clusterNeighbors = [[] for _, a in adata.obs.groupby(by=clusterLabel)]

        agraph = build_delaunay_graph(np.array([[x_, y_] for x_, y_ in zip(clusterPosX, clusterPosY)]),
            np.ones(len(clusterPosX)))

        try:
            for i, ii in agraph.items():
                clusterNeighbors[i].extend(ii)
        except:
            raise(Exception(""))

        seen = set()
        triples, uniquetriples = [], []
        clusterwise_triples = [[] for _ in range(len(clusterPosX))]
        for i in range(len(clusterNeighbors)):

            firstLevelNeighbors = clusterNeighbors[i]
            for firstL in firstLevelNeighbors:
                for secondL in clusterNeighbors[firstL]:
                    if i in clusterNeighbors[secondL] and [i, firstL, secondL] not in triples and \
                            [i, secondL, firstL] not in triples:
                        clusterwise_triples[i].append([firstL, secondL])
                    if i in clusterNeighbors[secondL] and frozenset([i, firstL, secondL]) not in seen:
                        uniquetriples.append([i, firstL, secondL])
                        seen.add(frozenset(uniquetriples[-1]))

        return np.unique([a for b in clusterwise_triples for a in b], axis=0), uniquetriples, clusterIds

    '''def plotClusterCNsAndSubClusters(self, patient, adata, location='ytma_analysis_figures'):
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10,10))
        sq.pl.spatial_scatter(adata, shape=None, color= \
            f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}',
                              cmap='tab20', size=8, ax=ax[0,0])
        sq.pl.spatial_scatter(adata, shape=None, color='new_cell_types',
                              cmap='tab20', size=8, ax=ax[0,1])
        sq.pl.spatial_scatter(adata, shape=None, color= \
            f'nolan_cellular_neighborhood_subclusterindex_Km{self.k}_Knn{self.n}',
                              cmap='tab20', size=8, ax=ax[1,0])
        sq.pl.spatial_scatter(adata, shape=None, color= \
            f'nolan_cellular_neighborhood_subclustertype_Km{self.k}_Knn{self.n}',
                              cmap='tab20', size=8, ax=ax[1,1])
        plt.savefig(f"{location}/{patient}_clusters.svg")
        plt.close()
        return'''

    def plot_clustering(self, patient, adata):
        sc.set_figure_params(format='png', dpi=80, dpi_save=300, figsize=(30, 30))
        #fig = sc.pl.spatial(self.adata,
        #                    color=f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}',
        #                    spot_size=12, color_map="Paired",
        #                    layer='log1p_normalised', vmin='p10', vmax='p90',
        #                    size=1.5, alpha_img=0, return_fig=True, show=False)
        #gc.collect()
        #plt.savefig(f'{self.baseFolder}/{self.patient}_nolan_' + self.imageName + f"k{self.k}n{self.n}" + '.png')
        #plt.close()

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(30,20))
        sq.pl.spatial_scatter(adata, shape=None, color=f'nolan_cellular_neighborhood_labels_Km{self.k}_Knn{self.n}',
                              layer='log1p_normalised', size=19, ax=ax[0,0])
        sq.pl.spatial_scatter(adata, shape=None, color='cell_types',
                              layer='log1p_normalised', size=19, ax=ax[0,1])
        sq.pl.spatial_scatter(adata, shape=None, color=f'Immune_func',
                              layer='log1p_normalised', size=19, ax=ax[1,0])
        sq.pl.spatial_scatter(adata, shape=None, color=f'Immune_meta',
                              layer='log1p_normalised', size=19, ax=ax[1,1])
        #gc.collect()
        plt.savefig(f'{self.baseFolder}/{patient}_nolan_k{self.k}n{self.n}' + '.png')
        plt.close()

    def plot_enrichment_scores(self, patient, adata):
        fig, ax = plt.subplots(figsize=(10, 8))
        nolan_plotting.plot_enrichment_scores(
            adata,
            knn=self.n,
            kmeans=self.k,
            transpose=True,  # Swap x and y axis
            cmap="bwr",  # blue, white, red cmap. Best cmap for negative to positive scaling.
            vmin=-3,  # Enforce data to fal between -3 and 3
            vmax=3,
            figsize=(7, 5)  # aspect ratio
        )
        plt.savefig(f'{self.baseFolder}/{patient}_enrichment_' + f"k{self.k}n{self.n}" + '.png')
        plt.close()