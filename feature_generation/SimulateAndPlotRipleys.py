from PoissonProcessSim import *
from ComputeRipleyMetrics import *
from PlotRipleyFunctions import *
from ReadClinicalData import *
from ComputeNeighbourhoodCounts import *
from ComputeRipleyMetrics import calculate_cross_k_auc
from PerlinProcessSim import *
from ClusterPoints import *
import numpy as np

def computeTest():
    radii = np.arange(5, 200, 20)
    clinicalData = ReadClinicalData()

    for tissue in clinicalData.returnAllTissueIDs():

        clinicalData.createImageAnnData(tissue, image_column='Tissue_Subsample_ID')

        clusterClinicalData = ClusterPointsNolan(adata=clinicalData.imageAData,
                                                 n=8, k=8,
                                                 grouping="Tissue_Subsample_ID",
                                                 phenotype='tumournontumour',
                                                 imageName=tissue)
        clusterClinicalData.cluster()
        clinicalData.imageAData.uns[f"nolan_cellular_neighborhood_inertia_Km{\
            clusterClinicalData.k}_Knn{clusterClinicalData.n}"] = clusterClinicalData.inertia
        clinicalData.imageAData.obs[f"nolan_cellular_neighborhood_labels_Km{\
            clusterClinicalData.k}_Knn{clusterClinicalData.n}"] = clusterClinicalData.labels
        clinicalData.imageAData.uns[f"nolan_CN_enrichment_matrix_Km{\
            clusterClinicalData.k}_Knn{clusterClinicalData.n}"] = clusterClinicalData.enrichment_matrix

        clinicalDataComputeRipleyMetrics = ComputeMetricsAnnData(clinicalData.imageAData,
                                                                       clinicalData.imageName,
                                                                       phenotype=\
                                                                           f"nolan_cellular_neighborhood_labels_Km{\
                                                                               clusterClinicalData.k \
                                                                           }_Knn{clusterClinicalData.n}",
                                                                       radii_of_computation=radii,
                                                                       image_column='Tissue_Subsample_ID',
                                                                       reference_type=None,
                                                                       target_type=None,
                                                                       )
        phenotypes = clinicalDataComputeRipleyMetrics.returnPhenoTypeSet(
            f"nolan_cellular_neighborhood_labels_Km{clusterClinicalData.k}_Knn{clusterClinicalData.n}")
        print("phenotypes: ", phenotypes)
        clinicalDataComputeRipleyMetrics.reference_type = phenotypes
        clinicalDataComputeRipleyMetrics.target_type    = phenotypes
        clinicalDataComputeRipleyMetrics.calculateCross()

        # Plot the Two-Class K-Ripley functions
        plotKRipleyFunctions = PlotRipleyFunctions(clinicalDataComputeRipleyMetrics.radii_to_compute,
                                k_ripley_values_1=clinicalDataComputeRipleyMetrics.kripleyCrossValues,
                                label_1=[f'CN Cross-K {i}x{j} Image ' + tissue
                                     for j in clinicalDataComputeRipleyMetrics.reference_type
                                     for i in clinicalDataComputeRipleyMetrics.target_type],
                                file_name='k_ripley_crossk_phenoCN_tarrefCNs'
                            )
        plotKRipleyFunctions.plotKRipleys()




def compute():
    """
        Compute spatial metrics for each cluster
    """
    radii = np.arange(5, 200, 20)
    clinicalData = ReadClinicalData()
    clinicalData.plotAllTissues()
    #clinicalData.plotAllImages()

    return

    clinicalAndSpatialValues = []

    for tissue in clinicalData.returnAllTissueIDs():

        clinicalData.createImageAnnData(tissue, image_column='Tissue_Subsample_ID')

        clusterClinicalData = ClusterPointsNolan(adata=clinicalData.imageAData,
                                                 n=8, k=8,
                                                 grouping="Tissue_Subsample_ID",
                                                 phenotype='tumournontumour',
                                                 imageName=tissue)
        clusterClinicalData.cluster()
        clinicalData.imageAData.uns[f"nolan_cellular_neighborhood_inertia_Km{\
            clusterClinicalData.k}_Knn{clusterClinicalData.n}"] = \
            clusterClinicalData.inertia
        clinicalData.imageAData.obs[f"nolan_cellular_neighborhood_labels_Km{\
            clusterClinicalData.k}_Knn{clusterClinicalData.n}"] = \
            clusterClinicalData.labels
        clinicalData.imageAData.uns[f"nolan_CN_enrichment_matrix_Km{\
            clusterClinicalData.k}_Knn{clusterClinicalData.n}"] = \
            clusterClinicalData.enrichment_matrix

        clinicalDataComputeRipleyMetrics = ComputeMetricsAnnData(clinicalData.imageAData,
                                                                       clinicalData.imageName,
                                                                       phenotype=f"nolan_cellular_neighborhood_labels_Km{clusterClinicalData.k \
                                                                           }_Knn{clusterClinicalData.n}",
                                                                       radii_of_computation=radii,
                                                                       image_column='Tissue_Subsample_ID',
                                                                       reference_type=None,
                                                                       target_type=None,
                                                                       )
        phenotypes = clinicalDataComputeRipleyMetrics.returnPhenoTypeSet(
            f"nolan_cellular_neighborhood_labels_Km{clusterClinicalData.k}_Knn{clusterClinicalData.n}")
        print("phenotypes: ", phenotypes)
        clinicalDataComputeRipleyMetrics.reference_type = phenotypes
        clinicalDataComputeRipleyMetrics.target_type    = phenotypes
        clinicalDataComputeRipleyMetrics.calculateCross()

        clusterlabels = list(set(clinicalData.imageAData.obs[f'nolan_cellular_neighborhood_labels_Km{\
            clusterClinicalData.k}_Knn{clusterClinicalData.n}']))
        for label in clusterlabels:
            for label2 in clusterlabels:
                if label2 > label:

                    print(f"Most common cell type in cluster {label} is {str(clusterClinicalData.returnModeOfClusterCellTypes(label))}")

                    sums = clinicalData.adata.obs['immune_func'].value_counts().sum()
                    counts = [label, label2]
                    for vals in ['Fibroblasts', 'CD4_cell', 'Macrophage_LAMP1', 'CD8_cell']:
                        try:
                            counts.append(clinicalData.imageAData[clinicalData.imageAData.obs[f'nolan_cellular_neighborhood_labels_Km{\
                                clusterClinicalData.k}_Knn{clusterClinicalData.n}'].isin([label])].obs['immune_func'].value_counts()[vals]/sums)
                        except:
                            counts.append(0.0)

                    counts.append(
                        np.array(clinicalDataComputeRipleyMetrics.kripleyCrossValues)[
                                (np.array(clinicalDataComputeRipleyMetrics.kripleyCrossLabelsReference) == label) &
                                (np.array(clinicalDataComputeRipleyMetrics.kripleyCrossLabelsTarget) == label2)])
                    #counts.append(clinicalDataComputeRipleyMetrics.radii_to_compute[3])
                    counts.append(clinicalData.imageAData[clinicalData.imageAData.obs[f'nolan_cellular_neighborhood_labels_Km{ \
                            clusterClinicalData.k}_Knn{clusterClinicalData.n}'].isin([label])].obs['OS_time'].mean())
                    clinicalAndSpatialValues.append(counts)

                    #print(f"{label} and {label2}", clinicalAndSpatialValues)

    print("clinicalAndSpatialValues: ", clinicalAndSpatialValues)

    return


def runComputePerlinAndCluster():
    PIXEL_COUNT = [100, 100]
    DENSITY = 0.00338
    radii = np.arange(5, 300, 20)

    poissonProcessSim = PerlinProcessSim(density=DENSITY, pixel_count=PIXEL_COUNT, number_octaves=4)


    IMAGE_NAME = 'A514CR417X_0'#'A514CR418p_0'
    clinicalData = ReadClinicalData()
    print(set(clinicalData.adata.obs.Tissue_Subsample_ID))
    clinicalData.createImageAnnData(IMAGE_NAME, image_column='Tissue_Subsample_ID')
    clinicalData.plotImage()
    #clinicalData.plotAllTissues()




    clusterClinicalData = ClusterPointsNolan(adata=clinicalData.imageAData,
                                             n=8, k=8,
                                             grouping="Tissue_Subsample_ID",
                                             phenotype='tumournontumour',
                                             imageName=IMAGE_NAME)
    clusterClinicalData.cluster()
    clinicalData.imageAData.uns[f"nolan_cellular_neighborhood_inertia_Km{\
        clusterClinicalData.k}_Knn{clusterClinicalData.n}"] = \
        clusterClinicalData.inertia
    clinicalData.imageAData.obs[f"nolan_cellular_neighborhood_labels_Km{\
        clusterClinicalData.k}_Knn{clusterClinicalData.n}"] = \
        clusterClinicalData.labels
    clinicalData.imageAData.uns[f"nolan_CN_enrichment_matrix_Km{\
        clusterClinicalData.k}_Knn{clusterClinicalData.n}"] = \
        clusterClinicalData.enrichment_matrix
    clusterClinicalData.plot_clustering()
    clusterClinicalData.plot_enrichment_scores()
    clusterClinicalData.printClusterInfo()







    clinicalDataComputeRipleyMetrics = ComputeMetricsAnnData(clinicalData.imageAData,
                                                                   clinicalData.imageName,
                                                                   phenotype=f"nolan_cellular_neighborhood_labels_Km{clusterClinicalData.k \
                                                                       }_Knn{clusterClinicalData.n}",
                                                                   radii_of_computation=radii,
                                                                   image_column='Tissue_Subsample_ID',
                                                                   reference_type=[0],
                                                                   target_type=[1],
                                                                   )
    phenotypes = clinicalDataComputeRipleyMetrics.returnPhenoTypeSet(
        f"nolan_cellular_neighborhood_labels_Km{clusterClinicalData.k}_Knn{clusterClinicalData.n}")
    print("phenotypes: ", phenotypes)
    clinicalDataComputeRipleyMetrics.reference_type = phenotypes
    clinicalDataComputeRipleyMetrics.target_type = phenotypes
    clinicalDataComputeRipleyMetrics.calculateCross()
    #clinicalDataComputeRipleyMetrics.calculateSingle()


    # Plot the Two-Class K-Ripley functions
    PlotRipleyFunctions(clinicalDataComputeRipleyMetrics.radii_to_compute,
                            k_ripley_values_1=clinicalDataComputeRipleyMetrics.kripleyCrossValues,
                            label_1=[f'CN Cross-K {i}x{j} Image ' + IMAGE_NAME
                                 for j in clinicalDataComputeRipleyMetrics.reference_type
                                 for i in clinicalDataComputeRipleyMetrics.target_type],
                            file_name='k_ripley_crossk_phenoCN_tarrefCNs'
                        )

    computeNeighbourhoodCountsAnnData = ComputeNeighbourhoodCountsAnnData(clinicalData.imageAData,
                                                                          radii,
                                                                          phenotype='tumournontumour',
                                                                          target='tumour',
                                                                          reference='nontumour'
                                                                          )
    computeNeighbourhoodCountsAnnData.computeCINAndNMS()

    # Plot the Cells-In-Neighbourhood of Clinical Data
    PlotRipleyFunctions(computeNeighbourhoodCountsAnnData.radii_of_computation,
                        k_ripley_values_1 = computeNeighbourhoodCountsAnnData.cin,
                        label_1 = 'CIN: Clinical',
                        file_name = 'CIN_tumourNontumour'
                        )
    PlotRipleyFunctions(computeNeighbourhoodCountsAnnData.radii_of_computation,
                        k_ripley_values_1 = computeNeighbourhoodCountsAnnData.nms,
                        label_1 = 'NMS: Clinical',
                        file_name = 'NMS_tumourNontumour'
                        )


    computeNeighbourhoodCountsAnnData = ComputeNeighbourhoodCountsAnnData(clinicalData.imageAData,
                                                                          radii,
                                                                          phenotype=f"nolan_cellular_neighborhood_labels_Km{clusterClinicalData.k}_Knn{clusterClinicalData.n}",
                                                                          target=0,
                                                                          reference=6
                                                                          )
    computeNeighbourhoodCountsAnnData.computeCINAndNMS()

    # Plot the Normalized Mixing Score of Clinical Data
    PlotRipleyFunctions(computeNeighbourhoodCountsAnnData.radii_of_computation,
                        k_ripley_values_1 = computeNeighbourhoodCountsAnnData.cin,
                        label_1 = 'CIN: Clinical',
                        file_name = 'CIN_CNs0x6'
                        )
    PlotRipleyFunctions(computeNeighbourhoodCountsAnnData.radii_of_computation,
                        k_ripley_values_1 = computeNeighbourhoodCountsAnnData.nms,
                        label_1 = 'NMS: Clinical',
                        file_name = 'NMS_CNs0x6'
                        )





    clinicalDataComputeRipleyMetrics = ComputeMetricsAnnData(clinicalData.imageAData,
                                                                   clinicalData.imageName,
                                                                   phenotype = "tumournontumour",
                                                                   radii_of_computation=radii,
                                                                   image_column='Tissue_Subsample_ID',
                                                                   reference_type="tumour",
                                                                   target_type="nontumour",
                                                                   )
    clinicalDataComputeRipleyMetrics.calculateSingle()

    # Plot the Two-Class K-Ripley functions
    PlotRipleyFunctions(clinicalDataComputeRipleyMetrics.radii_to_compute,
                                              k_ripley_values_1=clinicalDataComputeRipleyMetrics.kripleyOneClassValues,
                                              label_1 = 'Tumour / Non-Tumour Cross-K Image ' + IMAGE_NAME,
                                              file_name = 'k_ripley_crossk_phenoCN_tarrefTumourNontumour'
                                              )





def runtest():

    radii = np.arange(10, 400, 5)

    #############################################################################
    # Get Clinical Data
    #############################################################################

    IMAGE_NAME = 'A514CR417X_0'
    clinicalData = ReadClinicalData()
    clinicalData.createImageAnnData(IMAGE_NAME, image_column='Tissue_Subsample_ID')
    clinicalData.plotImage()
    print(clinicalData.imageAData)

    #############################################################################
    # Do simulations of Poisson Image
    #############################################################################

    PIXEL_COUNT = [int(np.max(clinicalData.imageAData.obs['x']) - np.min(clinicalData.imageAData.obs['x'])),
                   int(np.max(clinicalData.imageAData.obs['y']) - np.min(clinicalData.imageAData.obs['y']))]
    DENSITY = len(clinicalData.imageAData.obs['x'])/(PIXEL_COUNT[0]*PIXEL_COUNT[1])
    print(f"PIXEL_COUNT {PIXEL_COUNT}")
    print(f"DENSITY {DENSITY}")

    poissonProcessSimulation = PoissonProcessSim(density=DENSITY, pixel_count=PIXEL_COUNT)
    poissonProcessSimulation_Validation = PoissonProcessSim(density=DENSITY, pixel_count=PIXEL_COUNT)

    #############################################################################
    # Compute Ripley One-Class Function
    #############################################################################

    clinicalDataComputeRipleyMetrics = ComputeMetricsAnnData(clinicalData.imageAData, clinicalData.imageName,
                                                                   radii_of_computation=radii, image_column='Tissue_Subsample_ID')
    print("One-Class Ripley K-Function Clinical Data: ", clinicalDataComputeRipleyMetrics.kripleyOneClassValues)

    # Compute K-Ripley metrics
    print("poissonProcessSimulation.x.shape: ", poissonProcessSimulation.x.shape)
    print("poissonProcessSimulation.y.shape: ", poissonProcessSimulation.y.shape)
    poissonProcessComputeRipleyMetrics = ComputeRipleyMetrics()
    poissonProcessComputeRipleyMetrics.computeRipleyKFunction(poissonProcessSimulation.x,
                                                              poissonProcessSimulation.y,
                                                              radii_of_computation=radii)
    print("One-Class Ripley K-Function Poisson Process: ", poissonProcessComputeRipleyMetrics.kripleyOneClassValues)

    #############################################################################
    # Compute Ripley Two-Class Function
    #############################################################################

    clincialDataCrossPoissonRipleyMetrics = ComputeMetrics()
    clincialDataCrossPoissonRipleyMetrics.computeKRipleyCrossFunction(radii, clinicalData.imageAData.obs['x'], clinicalData.imageAData.obs['y'],
                                                                      poissonProcessSimulation.x, poissonProcessSimulation.y
                                                                      )

    #############################################################################
    # Compute CIN and NMS metrics for Clinical Data
    #############################################################################

    computeNeighbourhoodCountsAnnData = ComputeNeighbourhoodCountsAnnData(clinicalData.imageAData, radii)
    computeNeighbourhoodCountsAnnData.computeCINAndNMS()
    print("Average Minimal Distance ", computeNeighbourhoodCountsAnnData.computeMinimalDistance())
    print("Average Minimal Distance to Target Cells ", computeNeighbourhoodCountsAnnData.computeMinimalDistanceToTarget())

    #############################################################################
    # Compute CIN and NMS metrics for Poisson Data
    #############################################################################

    computeNeighbourhoodCounts = ComputeNeighbourhoodCounts(radii)
    computeNeighbourhoodCounts.computeCIN(poissonProcessSimulation.x, poissonProcessSimulation.y,
                                          poissonProcessSimulation.x, poissonProcessSimulation.y)
    computeNeighbourhoodCounts.computeNMS(poissonProcessSimulation.x, poissonProcessSimulation.y,
                                          poissonProcessSimulation.x, poissonProcessSimulation.y)

    computeNeighbourhoodCounts_Cross = ComputeNeighbourhoodCounts(radii)
    computeNeighbourhoodCounts_Cross.computeCIN(poissonProcessSimulation.x, poissonProcessSimulation.y,
                                          poissonProcessSimulation_Validation.x, poissonProcessSimulation_Validation.y)
    computeNeighbourhoodCounts_Cross.computeNMS(poissonProcessSimulation.x, poissonProcessSimulation.y,
                                          poissonProcessSimulation_Validation.x, poissonProcessSimulation_Validation.y)



    #############################################################################
    # Make plots
    #############################################################################

    # Plot the One-Class K-Ripley functions
    PlotRipleyFunctions(poissonProcessComputeRipleyMetrics.radii_of_computation,
                                              k_ripley_values_1=poissonProcessComputeRipleyMetrics.kripleyOneClassValues,
                                              k_ripley_values_2=clinicalDataComputeRipleyMetrics.kripleyOneClassValues,
                                              label_1 = 'K-Ripley Poisson Image',
                                              label_2 = 'K-Ripley Image ' + IMAGE_NAME,
                                              file_name = 'k_ripley_values_2'
                                              )

    # Plot the Two-Class K-Ripley functions
    PlotRipleyFunctions(poissonProcessComputeRipleyMetrics.radii_of_computation,
                                              k_ripley_values_1=clincialDataCrossPoissonRipleyMetrics.kripleyCrossValues,
                                              k_ripley_values_2=clinicalDataComputeRipleyMetrics.kripleyCrossValues,
                                              label_1 = 'K-Ripley Poisson Image',
                                              label_2 = 'Tumour / Non-Tumour Cross-K Image ' + IMAGE_NAME,
                                              file_name = 'k_ripley_poisson_crossk'
                                              )

    # Plot the Two-Class K-Ripley AUC functions
    PlotRipleyFunctions(poissonProcessComputeRipleyMetrics.radii_of_computation,
                                              k_ripley_values_1=calculate_cross_k_auc(radii, clincialDataCrossPoissonRipleyMetrics.kripleyCrossValues,
                                                                                        clinicalDataComputeRipleyMetrics.kripleyCrossValues,
                                                                                      ),
                                              label_1 = 'Cross-K AUC',
                                              file_name = 'cross_k_auc'
                                              )

    # Plot the Cells-In-Neighbourhood of Clinical Data
    PlotRipleyFunctions(computeNeighbourhoodCountsAnnData.radii_of_computation,
                        k_ripley_values_1 = computeNeighbourhoodCountsAnnData.cin,
                        k_ripley_values_2 = computeNeighbourhoodCounts_Cross.cins,
                        label_1 = 'CIN: Clinical',
                        label_2 = 'CIN: Poisson',
                        file_name = 'CIN'
                        )

    # Plot the Normalized Mixing Score of Clinical Data
    PlotRipleyFunctions(computeNeighbourhoodCountsAnnData.radii_of_computation,
                        k_ripley_values_1 = computeNeighbourhoodCountsAnnData.nms,
                        k_ripley_values_2 = computeNeighbourhoodCounts_Cross.nms,
                        label_1 = 'NMS: Clinical',
                        label_2 = 'NMS: Poisson',
                        file_name = 'NMS'
                        )

#runtest()
#runComputePerlinAndCluster()
computeTest()