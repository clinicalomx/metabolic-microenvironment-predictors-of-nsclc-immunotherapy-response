from PlotRipleyFunctions import *
from ReadClinicalData import *
from ComputeNeighbourhoodCounts import *
#from PerlinProcessSim import *
from NolanCNsAndSubclustering import *
from PatientAnnDataEP import *
from Spatial2DDensityCorrelation import *
from ComputeGraphMetrics import *
import warnings
import anndata, pickle
warnings.simplefilter('ignore', category=anndata.ImplicitModificationWarning)
from helpers import computeConcaveHull

def analyze(patient_list):
    radii = np.arange(5, 200, 20)
    gcross_radii = np.arange(5, 200, 20)
    k, n = 3, 20 # Nolan cluster count and neighbor clustering count
    CNGroupLabels = list(range(k))
    patientcolumnname = 'patientid'

    #r'/mnt/d/YTMA_NSCLS/nsclc_adata_features.h5ad'
    #r'D:\YTMA_NSCLS\nsclc_adata_features.h5ad'
    #hnscc_adata_features
    #clinicalData = ReadClinicalData(file_location=r'C:\Users\AaronKilgallon\NSCLC_HNCC_Analysis\Aaron_Analysis\hnscc_adata_features.h5ad', patients='patient_id')
    clinicalData = ReadClinicalData(file_location=r'C:\Users\AaronKilgallon\YTMA_ANALYSIS_CHAIN\AnnDataFiles\2-output_nsclc_ytma_features_preandpost.h5ad', patients=patientcolumnname)

    clinicalData.refactorMetaAndImmune()

    print(clinicalData.adata.obs.keys())

    # If the samples don't need to be separated, each tissue
    clinicalData.adata.obs[patientcolumnname]   = clinicalData.adata.obs[patientcolumnname].astype(str)
    clinicalData.adata.obs['Tissue_Subsample_ID'] = clinicalData.adata.obs[patientcolumnname].astype(str)

    print("clinicalData.adata.obs['patientcolumnname']: ", clinicalData.adata.obs[patientcolumnname])

    listOfCellTypes = list(set(clinicalData.adata.obs.celltypes))

    clinicalOutcomes = clinicalData.returnPatientsAndOutcomes(outcomeLabel='OR', timeLabel='os.T', eventLabel='os.E',
                                                              groupby=[['PD'], ['CR', 'PR', 'NE', 'SD']], labels=['NoResponse', 'Response'])

    # metricsDict: A user-provided driver of the metrics to compute
    # {Label: [MetricType(KRipleyCross, GCross, etc.), Reference Phenotype (cell_types, immune_func, etc.),
    #          Grouping Label (nolan_cellular_neighborhood_labels_Km{k}_Knn{n}, etc.),
    #          Neighbour Label (subclusterDelauneyNeighbors or CNDelauneyNeighbors or None)
    #          Edges Label (nolan_cellular_neighborhood_edges_Km{k}_Knn{n}, etc.)
    #          Dimensionality of metric (GxGxCxCxR, GxCxC, GxCxR, etc., for group G, for phenotypes C, and radii R)
    #         ]}


    #TODO: StoredTYpes are calculated internally
    metricsDict = {
        'GCross_k3CNsxCellsxCells_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", 'celltypes', 'celltypes'],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},
        'GCross_k3CNsxImmunefuncxImmunefunc_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "Immune_func", "Immune_func"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_k3CNsxImmunefuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "Immune_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_k3CNsxImmunemetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "Immune_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_k3CNsxImmunemetaxImmunemeta_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "Immune_meta", "Immune_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},
        'GCross_k3CNsxTumourfuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "tumour_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_k3CNsxTumourmetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "tumour_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},
        'GCross_k3CNsxFuncmetapathxFuncmetapath_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "cell_types_immune_metapath", "cell_types_immune_metapath"],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},


        'GCross_k2CNsxCellsxCells_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", 'celltypes', 'celltypes'],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},
        'GCross_k2CNsxImmunefuncxImmunefunc_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", "Immune_func", "Immune_func"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_k2CNsxImmunefuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", "Immune_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_k2CNsxImmunemetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", "Immune_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_k2CNsxImmunemetaxImmunemeta_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", "Immune_meta", "Immune_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},
        'GCross_k2CNsxTumourfuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", "tumour_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_k2CNsxTumourmetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", "tumour_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},
        'GCross_k2CNsxFuncmetapathxFuncmetapath_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", "cell_types_immune_metapath", "cell_types_immune_metapath"],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},


        'GCross_MPCNsxCellsxCells_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", 'celltypes', 'celltypes'],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},
        'GCross_MPCNsxImmunefuncxImmunefunc_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "Immune_func", "Immune_func"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_MPCNsxImmunefuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "Immune_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_MPCNsxImmunemetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "Immune_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_MPCNsxImmunemetaxImmunemeta_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "Immune_meta", "Immune_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},
        'GCross_MPCNsxTumourfuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "tumour_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxCxR'},
        'GCross_MPCNsxTumourmetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "tumour_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},
        'GCross_MPCNsxFuncmetapathxFuncmetapath_NoNeighs_AllCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "cell_types_immune_metapath", "cell_types_immune_metapath"],
            'radii': radii,
            'neighbors': None,
            'edges': None,  # f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
            'storedType': 'GxCxCxR'},


        'GCross_k3CNsxCellsxCells_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "nb_tumournontumour_20_3", 'celltypes', 'celltypes'],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_20_3_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k3CNsxImmunefuncxImmunefunc_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "nb_tumournontumour_20_3", "Immune_func", "Immune_func"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_20_3_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k3CNsxImmunefuncxTumourfunc_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "nb_tumournontumour_20_3", "Immune_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_20_3_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k3CNsxImmunemetaxTumourmeta_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "nb_tumournontumour_20_3", "Immune_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_20_3_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k3CNsxImmunemetaxImmunemeta_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "nb_tumournontumour_20_3", "Immune_meta", "Immune_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_20_3_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k3CNsxTumourfuncxTumourfunc_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "nb_tumournontumour_20_3", "tumour_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_20_3_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k3CNsxTumourmetaxTumourmeta_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "nb_tumournontumour_20_3", "tumour_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_20_3_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k3CNsxFuncmetapathxFuncmetapath_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_20_3", "nb_tumournontumour_20_3", "cell_types_immune_metapath", "cell_types_immune_metapath"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_20_3_edges",
            'storedType': 'GxGxCxCxR'},


        'GCross_k2CNsxCellsxCells_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", f"nb_tumournontumour_50_2", 'celltypes', 'celltypes'],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_50_2_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k2CNsxImmunefuncxImmunefunc_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", f"nb_tumournontumour_50_2", "Immune_func", "Immune_func"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_50_2_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k2CNsxImmunefuncxTumourfunc_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", f"nb_tumournontumour_50_2", "Immune_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_50_2_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k2CNsxImmunemetaxTumourmeta_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", f"nb_tumournontumour_50_2", "Immune_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_50_2_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k2CNsxImmunemetaxImmunemeta_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", f"nb_tumournontumour_50_2", "Immune_meta", "Immune_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_50_2_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k2CNsxTumourfuncxTumourfunc_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", f"nb_tumournontumour_50_2", "tumour_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_50_2_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k2CNsxTumourmetaxTumourmeta_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", f"nb_tumournontumour_50_2", "tumour_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_50_2_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_k2CNsxFuncmetapathxFuncmetapath_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"nb_tumournontumour_50_2", f"nb_tumournontumour_50_2", "cell_types_immune_metapath", "cell_types_immune_metapath"],
            'radii': radii,
            'neighbors': None,
            'edges': f"nb_tumournontumour_50_2_edges",
            'storedType': 'GxGxCxCxR'},

        'GCross_MPCNsxCellsxCells_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "MetaPathNeighbourhood", 'celltypes', 'celltypes'],
            'radii': radii,
            'neighbors': None,
            'edges': f"MetaPathNeighbourhood_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_MPCNsxImmunefuncxImmunefunc_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "MetaPathNeighbourhood", "Immune_func", "Immune_func"],
            'radii': radii,
            'neighbors': None,
            'edges': f"MetaPathNeighbourhood_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_MPCNsxImmunefuncxTumourfunc_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "MetaPathNeighbourhood", "Immune_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': f"MetaPathNeighbourhood_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_MPCNsxImmunemetaxTumourmeta_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "MetaPathNeighbourhood", "Immune_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': f"MetaPathNeighbourhood_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_MPCNsxImmunemetaxImmunemeta_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "MetaPathNeighbourhood", "Immune_meta", "Immune_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': f"MetaPathNeighbourhood_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_MPCNsxTumourfuncxTumourfunc_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "MetaPathNeighbourhood", "tumour_func", "tumour_func"],
            'radii': radii,
            'neighbors': None,
            'edges': f"MetaPathNeighbourhood_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_MPCNsxTumourmetaxTumourmeta_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "MetaPathNeighbourhood", "tumour_meta", "tumour_meta"],
            'radii': radii,
            'neighbors': None,
            'edges': f"MetaPathNeighbourhood_edges",
            'storedType': 'GxGxCxCxR'},
        'GCross_MPCNsxFuncmetapathxFuncmetapath_NoNeighs_EdgeCells': {
            'metricType': 'GCross',
            'groupBys': [f"MetaPathNeighbourhood", "MetaPathNeighbourhood", "cell_types_immune_metapath", "cell_types_immune_metapath"],
            'radii': radii,
            'neighbors': None,
            'edges': f"MetaPathNeighbourhood_edges",
            'storedType': 'GxGxCxCxR'},


        'JSDScores_k3CNxCellsxCells_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_20_3",'celltypes', 'celltypes'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k3CNxImmunefuncxImmunefunc_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_20_3", 'Immune_func', 'Immune_func'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k3CNxImmunefuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': [f"nb_tumournontumour_20_3", "Immune_func", "tumour_func"],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k3CNxImmunemetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': [f"nb_tumournontumour_20_3", "Immune_meta", "tumour_meta"],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k3CNxImmunemetaxImmunemeta_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_20_3", 'Immune_meta', 'Immune_meta'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k3CNxTumourfuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_20_3", 'tumour_func', 'tumour_func'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k3CNxTumourmetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_20_3", 'tumour_meta', 'tumour_meta'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k3CNxFuncmetapathxFuncmetapath_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_20_3", 'cell_types_immune_metapath', 'cell_types_immune_metapath'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},


        'JSDScores_k2CNxCellsxCells_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_50_2", 'celltypes', 'celltypes'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k2CNxImmunefuncxImmunefunc_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_50_2", 'Immune_func', 'Immune_func'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k2CNxImmunefuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': [f"nb_tumournontumour_50_2", "Immune_func", "tumour_func"],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k2CNxImmunemetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': [f"nb_tumournontumour_50_2", "Immune_meta", "tumour_meta"],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k2CNxImmunemetaxImmunemeta_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_50_2", 'Immune_meta', 'Immune_meta'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k2CNxTumourfuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_50_2", 'tumour_func', 'tumour_func'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k2CNxTumourmetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_50_2", 'tumour_meta', 'tumour_meta'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_k2CNxFuncmetapathxFuncmetapath_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["nb_tumournontumour_50_2", 'cell_types_immune_metapath', 'cell_types_immune_metapath'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},


        'JSDScores_MPCNxCellsxCells_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["MetaPathNeighbourhood", 'celltypes', 'celltypes'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_MPCNxImmunefuncxImmunefunc_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["MetaPathNeighbourhood", 'Immune_func', 'Immune_func'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_MPCNxImmunefuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': [f"MetaPathNeighbourhood", "Immune_func", "tumour_func"],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_MPCNxImmunemetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': [f"MetaPathNeighbourhood", "Immune_meta", "tumour_meta"],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_MPCNxImmunemetaxImmunemeta_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["MetaPathNeighbourhood", 'Immune_meta', 'Immune_meta'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_MPCNxTumourfuncxTumourfunc_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["MetaPathNeighbourhood", 'tumour_func', 'tumour_func'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_MPCNxTumourmetaxTumourmeta_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["MetaPathNeighbourhood", 'tumour_meta', 'tumour_meta'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
        'JSDScores_MPCNxFuncmetapathxFuncmetapath_NoNeighs_AllCells': {
            'metricType': 'JSDScores',
            'groupBys': ["MetaPathNeighbourhood", 'cell_types_immune_metapath', 'cell_types_immune_metapath'],
            'radii': None,
            'neighbors': None,
            'edges': None,
            'storedType': 'GxCxC'},
    }


    #'KRipleyCross_CNsxCNsxCellsxCells_NoNeighs_EdgeCells': {
    #    'metricType': 'KRipleyCross',
    #    'groupBys': [f"nolan_cellular_neighborhood_labels_Km{k}_Knn{n}",
    #                 f"nolan_cellular_neighborhood_labels_Km{k}_Knn{n}", 'cell_types', 'cell_types'],
    #    'radii': radii,
    #    'neighbors': None,
    #    'edges': f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",
    #    'storedType': 'GxGxCxCxR'},
    ##'MutualInformation_CellsxCells_None_None_AllCells': ['JSDScores', 'cell_types', 'cell_types', None, None, None, None, 'GxCxC']
    ##'GraphConductance_CellsxCells_CNs_ReqNeigh_EdgeCells': ["Conductance", 'cell_types', 'cell_types', None,
    ##       f"nolan_cellular_neighborhood_labels_Km{k}_Knn{n}", 'CNDelauneyNeighbors',
    ##       f"nolan_cellular_neighborhood_edges_Km{k}_Knn{n}",  'GxGxCxC']

    #
    # '''
    # 'GCross_CNsxCNsxCellsxCells_NoNeighs_EdgeCells': {
    #    'metricType': 'GCross',
    #    'groupBys': ["nb_celltypes_20_3",
    #                 "nb_celltypes_20_3", 'celltypes', 'celltypes'],
    #    'radii': radii,
    #    'neighbors': None,
    #    'edges': f"nb_celltypes_20_3_edges",
    #    'storedType': 'GxGxCxCxR'},
    # 'GCross_CNsxCNsxImmunefuncxImmunefunc_NoNeighs_EdgeCells': {
    #    'metricType': 'GCross',
    #    'groupBys': ["nb_celltypes_20_3",
    #                 "nb_celltypes_20_3", "Immune_func", "Immune_func"],
    #    'radii': radii,
    #    'neighbors': None,
    #    'edges': f"nb_celltypes_20_3_edges",
    #   'storedType': 'GxGxCxCxR'},
    # 'GCross_CNsxCNsxImmunemetaxImmunemeta_NoNeighs_EdgeCells': {
    #    'metricType': 'GCross',
    #    'groupBys': ["nb_celltypes_20_3",
    #                 "nb_celltypes_20_3", "Immune_meta", "Immune_meta"],
    #    'radii': radii,
    #    'neighbors': None,
    #    'edges': "nb_celltypes_20_3_edges",
    #    'storedType': 'GxGxCxCxR'},
    # 'GCross_CNsxCNsxfuncmetapathxfuncmetapath_NoNeighs_EdgeCells': {
    #    'metricType': 'GCross',
    #    'groupBys': ["nb_celltypes_20_3",
    #                 "nb_celltypes_20_3", "cell_types_immune_metapath", "cell_types_immune_metapath"],
    #    'radii': radii,
    #    'neighbors': None,
    #    'edges': "nb_celltypes_20_3_edges",
    #    'storedType': 'GxGxCxCxR'},
    # '''

    # '''
    # 'InfiltrationScore_CNsxImmunefuncxImmunefunc_NoNeighs_AllCells': {
    #        'metricType': 'InfiltrationScores',
    #        'groupBys': ["nb_celltypes_20_3", 'Immune_func', 'Immune_func'],
    #        'radii': None,
    #        'neighbors': None,#'CNDelauneyNeighbors',
    #        'edges': None,
    #        'storedType': 'GxCxC'},
    # 'InfiltrationScore_CNsxCellsxCells_NoNeighs_AllCells': {
    #        'metricType': 'InfiltrationScores',
    #        'groupBys': ["nb_celltypes_20_3", 'celltypes', 'celltypes'],
    #        'radii': None,
    #        'neighbors': None,#'CNDelauneyNeighbors',
    #        'edges': None,
    #        'storedType': 'GxCxC'},
    # 'InfiltrationScore_CNsxImmunemetaxImmunemeta_NoNeighs_AllCells': {
    #    'metricType': 'InfiltrationScores',
    #    'groupBys': [f"nb_celltypes_20_3", 'Immune_meta', 'Immune_meta'],
    #    'radii': None,
    #    'neighbors': None,  # 'CNDelauneyNeighbors',
    #    'edges': None,
    #    'storedType': 'GxCxC'},
    # 'InfiltrationScore_CNsxfuncmetapathxfuncmetapath_NoNeighs_AllCells': {
    #    'metricType': 'InfiltrationScores',
    #    'groupBys': [f"nb_celltypes_20_3", 'cell_types_immune_metapath', 'cell_types_immune_metapath'],
    #    'radii': None,
    #    'neighbors': None,  # 'CNDelauneyNeighbors',
    #    'edges': None,
    #    'storedType': 'GxCxC'},
    # '''

    CNs = [
        'nb_tumournontumour_20_3', 'nb_tumournontumour_50_2', 'MetaPathNeighbourhood'
        ]

    ## TODO: Seems to have no effect here
    CNDefintionLabels = [
        'TNTCN3', 'TNTCN2', 'MPN'
        ]

    # Currently the CNs are sorted by tumour content whereas the MNs are unsorted
    k3CNIndexMap = {'Stroma': '2', 'Tumour_Stroma_interface': '1', 'Tumour': '0'}
    k2CNIndexMap = {'Stroma': '1', 'Tumour': '0'}

    clinicalData.adata.obs['nb_tumournontumour_20_3'] = clinicalData.adata.obs['nb_tumournontumour_20_3'].map(k3CNIndexMap).astype(str)
    clinicalData.adata.obs['nb_tumournontumour_50_2'] = clinicalData.adata.obs['nb_tumournontumour_50_2'].map(k2CNIndexMap).astype(str)
    clinicalData.adata.obs['MetaPathNeighbourhood']   = clinicalData.adata.obs['MetaPathNeighbourhood'].astype(str)


    CNGroupLabels = {}
    for CNDefintionLabel, CN in zip(CNDefintionLabels, CNs):
            CNGroupLabels[f'{CN}'] = {
                 str(a): b for a, b in zip(np.array(list(sorted(set(clinicalData.adata.obs[f'{CN}'])))).astype(str),
                                           np.array(list(range(len(set(clinicalData.adata.obs[f'{CN}']))))))}

    patientAnnData = PatientAnnDataEP(clinicalData.adata, metricsDict, CNGroupTypes=CNGroupLabels,
                        clinicalOutcomes=clinicalOutcomes,
                        fileName=r'C:\Users\AaronKilgallon\YTMA_ANALYSIS_CHAIN\AnnDataFiles\patientAnnData_withMPN_'+f'{'_'.join(patient_list)}.h5ad',
                        radii_of_computation=radii)

    clusterClinicalData = NolanCNsAndSubClustering(adata=clinicalData.adata,
                                                   n=n, k=k,
                                                   grouping="Tissue_Subsample_ID",
                                                   phenotype='celltypes',
                                                   baseFolder='YTMS_HNSCC_Images')
    ##YTMS_NSCLC_Images
    #clusterClinicalData.cluster(updateCNs=True)
    ##clusterClinicalData.assignCellTypesToCNs(phenotype='cell_types', selection='mode')
    ## clusterClinicalData.assignCellTypesToCNs(phenotype='cell_types', selection='top3')
    #clusterClinicalData.baseCNVariableName = f"nolan_cellular_neighborhood_labels_Km{clusterClinicalData.k \
    #    }_Knn{clusterClinicalData.n}"

    # Analysis loop
    nolan_barycentric_triangles=None
    for tissue in clinicalData.returnAllTissueIDs():

        print("Tissue: ", tissue)
        if tissue not in patient_list:
            print("Tissue not in patient_list -- skipping")
            continue


        clinicalData.createImageAnnData(tissue, image_column=patientcolumnname)

        ################################################################################################################
        # Subcluster and form edges
        ################################################################################################################

        print("clinicalData.imageAData.celltypes: ", set(clinicalData.imageAData.obs.celltypes))

        # Compute the subclusters of the CNs
        '''clinicalData.imageAData, clusterClinicalData.baseSubclusterVariableName = \
                clusterClinicalData.subclusterNolanCNs(clinicalData.imageAData)

        # Compute the edges of the Nolan CNs
        clinicalData.imageAData.obs[clusterClinicalData.baseCNVariableName.replace('labels', 'edges')] = \
            computeConcaveHull(clinicalData.imageAData.obs.x, clinicalData.imageAData.obs.y,
                clinicalData.imageAData.obs[clusterClinicalData.baseCNVariableName], concavity=2.0, length_threshold=0.5)

        # Compute the edges of the Subclusters
        clinicalData.imageAData.obs[clusterClinicalData.baseSubclusterVariableName.replace('index', 'edges')] = \
            computeConcaveHull(clinicalData.imageAData.obs.x, clinicalData.imageAData.obs.y,
                clinicalData.imageAData.obs[clusterClinicalData.baseCNVariableName], concavity=2.0, length_threshold=0.5)
                '''

        clinicalData.imageAData.obs['nb_tumournontumour_20_3_edges'] = \
            computeConcaveHull(clinicalData.imageAData.obs.x, clinicalData.imageAData.obs.y,
                               clinicalData.imageAData.obs['nb_tumournontumour_20_3'], concavity=2.0,
                               length_threshold=0.5)
        clinicalData.imageAData.obs['nb_tumournontumour_50_2_edges'] = \
            computeConcaveHull(clinicalData.imageAData.obs.x, clinicalData.imageAData.obs.y,
                               clinicalData.imageAData.obs['nb_tumournontumour_50_2'], concavity=2.0,
                               length_threshold=0.5)
        clinicalData.imageAData.obs['MetaPathNeighbourhood_edges'] = \
            computeConcaveHull(clinicalData.imageAData.obs.x, clinicalData.imageAData.obs.y,
                               clinicalData.imageAData.obs['MetaPathNeighbourhood'], concavity=2.0,
                               length_threshold=0.5)


        #print("Fraction of Edges: ", clinicalData.imageAData.obs[clusterClinicalData.baseCNVariableName.
        #      replace('labels', 'edges')].mean())

        ################################################################################################################
        # Get all neighbours of CNs and subclusters
        ################################################################################################################

        #CNDelauneyNeighbors, uniqueCDDelauneyNeighbors, actualLabelsOfCNs = clusterClinicalData.getClusterNeighbours(
        #    clinicalData.imageAData, clusterLabel=clusterClinicalData.baseCNVariableName,
        #)

        #subclusterDelauneyNeighbors, uniqueSubclusterDelauneyNeighbors, actualLabelsOfSubclusters = (
        #    clusterClinicalData.getClusterNeighbours(clinicalData.imageAData,
        #                                             clusterLabel=clusterClinicalData.baseSubclusterVariableName)
        #)

        ##clusterClinicalData.plot_clustering(tissue, clinicalData.imageAData)
        #clusterClinicalData.plot_enrichment_scores(tissue, clinicalData.imageAData)

        for metricLabel, metric in metricsDict.items():
            print("metricLabel: ", metricLabel, "\tmetric: ", metric)

            #if metric['neighbors'] == 'CNDelauneyNeighbors':
            #    cdN = CNDelauneyNeighbors
            #elif metric['neighbors'] == 'subclusterDelauneyNeighbors':
            #    cdN = subclusterDelauneyNeighbors
            #else:
            cdN = None

            # Compute JSD scores!
            if 'JSD' in metricLabel:
                s2DCorr = Spatial2DDensityCorrelationAnnData(clinicalData.imageAData, patientAnnData, metric)
                jsdScores = s2DCorr.compute()

                patientAnnData.updateValues(jsdScores, metric, tissue, metricLabel)
                continue

            if 'Graph' in metricLabel:
                assert metric['storedType'] in ['GxCxC', 'GxGxCxC'], "Graph-based methods are only implemented for GxGxCxC"

                graphMethods = ComputeGraphMetrics(clinicalData.imageAData, metric)

                continue

            if 'Infiltration' in metricLabel:
                assert metric['storedType'] in ["GxCxC"], "Infiltration needs a GxCxC metric type"

            # Compute Generic Metrics if not JSD score
            clinicalCompRiplMet = ComputeMetricsAnnData(
                clinicalData.imageAData,
                tissue,
                [metric['metricType']],
                [metric['storedType']],
                groupBys=metric['groupBys'],
                radii_of_computation=radii,
                image_column='Tissue_Subsample_ID',
                requireNeighbours=cdN,
                edges=metric['edges'],
            )
            clinicalCompRiplMet.compute()

            #print("Values: ", clinicalCompRiplMet.dictGxCxCxRValues, metric, tissue, metricLabel)

            if metric['storedType'] == 'GxGxCxCxR':
                patientAnnData.updateValues(clinicalCompRiplMet.dictGxGxCxCxRValues, metric, tissue, metricLabel)
            elif metric['storedType'] == 'GxGxCxC':
                patientAnnData.updateValues(clinicalCompRiplMet.dictGxGxCxCValues, metric, tissue, metricLabel)
            elif metric['storedType'] == 'GxCxCxR':
                patientAnnData.updateValues(clinicalCompRiplMet.dictGxCxCxRValues, metric, tissue, metricLabel)
            elif metric['storedType'] == 'GxCxC':
                patientAnnData.updateValues(clinicalCompRiplMet.dictGxCxCValues, metric, tissue, metricLabel)
            elif metric['storedType'] == 'GxCxR':
                patientAnnData.updateValues(clinicalCompRiplMet.dictGxCxRValues, metric, tissue, metricLabel)
            else:
                raise(Exception("metric type not in list"))


        ################################################################################################################
        # Finish and write to Patient_AnnData file
        ################################################################################################################
    
        patientAnnData.write()

    print("COMPLETE: ", patient_list)
    return

from multiprocessing import Pool

if __name__ == '__main__':
    with Pool(processes=10) as pool:
        futures = []
        # loop through each batch

        #for batch_num, batch in enumerate([31586, 32365, 34679, 34684, 34709, 34716, 34736, 34744, 34757, 34761, 34762, 34769]):
        for batch_num, batch in enumerate([34678, 31590, 22193, 34668, 34686, 34688, 31565, 32372, 34738, 34692, 34759, 28757, 34718, 34726, 34758, 34711, 34687, 34767, 34749, 32054, 34680, 34694, 34713, 34673, 34695, 34693, 32340, 34700, 32749, 34689, 34697, 34704, 34722, 31267, 34715, 34706, 34670, 22333, 34672, 34765, 24490, 34698, 3892, 32350, 32343, 34743, 34737, 29937, 34703, 34746, 32761, 34699, 34721, 31584, 34733]):
            futures.append(pool.apply_async(analyze, args=([str(batch)],)))

        for future in futures:
            batch_logs = future.get()  # wait for work to be done




#analyze(['34688'])