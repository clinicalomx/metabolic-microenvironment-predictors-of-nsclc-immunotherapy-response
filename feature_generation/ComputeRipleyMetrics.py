import ripleyk
import numpy as np
from scipy import spatial
from statsmodels.distributions.empirical_distribution import ECDF

class ComputeMetric:
    """
        ComputeMetrics: Computes Spatial metrics

    """
    def __init__(self):
        self.metricValues     = None
        self.mode = None
        self.radii_of_computation = None
        self.algorithm = None

    def compute(self, mode, radii_of_computation, XY, XY2=None, algorithm=None, totalNumberOfCellsInGroup=None, kwargs=None):
        if len(XY) < 2 or (XY2 is not None and len(XY2) < 2):
            if radii_of_computation is not None:
                self.metricValues = np.nan*np.zeros(len(radii_of_computation))
            else:
                self.metricValues = np.nan
            return
        self.mode = mode
        self.algorithm = algorithm
        if self.mode == 'KRipleyCross':
            assert XY2 is not None, ' Needs two input datasets'
            self.computeKRipleyCrossFunction(radii_of_computation, XY, XY2)
        elif self.mode == 'KRipleyOneClass':
            self.computeRipleyKFunction(radii_of_computation, XY)
        elif self.mode == 'GCross':
            assert XY2 is not None, ' Needs two input datasets'
            self.computeGCrossFunction(radii_of_computation, XY, XY2)
        elif self.mode == 'GOneClass':
            self.computeGFunction(radii_of_computation, XY)
        elif self.mode == 'InfiltrationScores':
            self.computeInfiltrationScores(XY, XY2, totalNumberOfCellsInGroup, kwargs)
        elif self.mode == 'MCross':
            self.computeMCross(XY, XY2, kwargs)
        else:
            if self.algorithm is None:
                raise Exception(f"{self.mode} is not in the allowed types and custom algorithm not provided.")
            else:
                print(f"Running with user-provided algorithm {self.algorithm}")
                self.algorithm(radii_of_computation, XY, XY2, *kwargs)

        self.radii_of_computation = radii_of_computation

    def computeRipleyKFunction(self, radii_of_computation, XY):
        X, Y = XY[:,0], XY[:,1]

        kvals = []
        for r in radii_of_computation:
            kvals.append(ripleyk.calculate_ripley(r, max(X), d1=X, d2=Y))
        self.metricValues = np.array(kvals, dtype=np.float32)

    def computeGFunction(self, radii_of_computation, XY):
        X, Y = XY[:,0], XY[:,1]
        gvals = []
        for r in radii_of_computation:
            gvals.append(ripleyk.calculate_ripley(r, max(X), d1=X, d2=Y))
        self.metricValues = np.array(gvals, dtype=np.float32)

    def computeKRipleyCrossFunction(self, radii_of_computation, X1Y1, X2Y2):
        if X1Y1.shape[0] < 2 or X2Y2.shape[0] < 2:
            self.metricValues = np.nan*np.empty(len(radii_of_computation), dtype=np.float32)
            return
        X1, Y1 = X1Y1[:,0], X1Y1[:,1]
        X2, Y2 = X2Y2[:,0], X2Y2[:,1]

        area = ((max(max(X1), max(X2)) - min(min(X1), min(X2)))*
                (max(max(Y1), max(Y2)) - min(min(Y1), min(Y2))))

        self.metricValues  = calculate_crossk_ripley(radii_of_computation, area, X1, Y1, X2, Y2)

    def computeGCrossFunction(self, radii_of_computation, X1Y1, X2Y2):
        if X1Y1.shape[0] < 2 or X2Y2.shape[0] < 2:
            self.metricValues = np.nan*np.empty(len(radii_of_computation), dtype=np.float32)
            return
        X1, Y1 = X1Y1[:, 0], X1Y1[:, 1]
        X2, Y2 = X2Y2[:, 0], X2Y2[:, 1]
        area = (max(max(X1), max(X2)) - min(min(X1), min(X2)))*(max(max(Y1), max(Y2)) - min(min(Y1), min(Y2)))

        self.metricValues = calculate_gcross(radii_of_computation, area, X1, Y1, X2, Y2)

    def computeMCross(self, radii_of_computation, XY, XY2, kwargs):
        mvals = []
        for r in radii_of_computation:
            pass
        raise(Exception("M Cross not implemented yet"))

    def computeInfiltrationScores(self, XY1, XY2, extras, diagonal=False):
        assert extras, ' Need total cell quantities to compute diagonal components'
        X1 = XY1[:, 0]
        X2 = XY2[:, 0]
        if diagonal:
            if extras > 5 and len(X1) > 5:
                self.metricValues = len(X1) / extras#, len(X2) / extras[1])
            else:
                self.metricValues = np.nan
        else:
            if len(X1) > 0:
                self.metricValues = len(X2) / len(X1)
            else:
                self.metricValues = np.nan

class ComputeMetricsAnnData:
    def __init__(self, aData,
                       image_name,
                       modes,
                       modetypes,
                       groupBys,
                       radii_of_computation=None,
                       image_column='Image',
                       requireNeighbours = None,
                       edges=None,
                 ):
        self.aData = aData
        self.image_name = image_name
        self.modes = modes
        self.modetypes = modetypes
        self.image_column = image_column
        self.groupBys = groupBys
        self.radii_to_compute = radii_of_computation
        self.edges = edges
        self.requireNeighbours = requireNeighbours

        assert [mode in ['InfiltrationScore', 'KRipleyCross', 'KRipley', 'MetaCross', 'GCross', 'G'] for mode in self.modes], \
            f' Modes not in allowable modes'

        # Computables
        self.dictGxCxRValues = {}
        self.dictGxCxCValues = {}
        self.dictGxCxCxRValues = {}
        self.dictGxGxCxCValues = {}
        self.dictGxGxCxCxRValues = {}

        # Other called classes
        self.metricFunction = ComputeMetric()

    def computeGxGxCxCMetric(self, mode, XYs1, XYs2, indexReference, indexTarget, group1, group2):
        raise(Exception("Not implemented"))

    def computeGxGxCxCxRMetric(self, mode, XYs1, XYs2, indexReference, indexTarget, group1, group2):
        self.metricFunction.compute(mode, self.radii_to_compute, XYs1, XYs2)

        if np.isnan(self.metricFunction.metricValues).any(): return

        if (mode, group1, group2, indexReference, indexTarget) not in self.dictGxCxCxRValues.keys():
            self.dictGxGxCxCxRValues[
                (mode, group1, group2, indexReference, indexTarget)] = self.metricFunction.metricValues

        else:
            self.dictGxGxCxCxRValues[
                (mode, group1, group2, indexReference, indexTarget)] += self.metricFunction.metricValues

    def computeGxCxCxRMetric(self, mode, XYs1, XYs2, indexReference, indexTarget, group1):
        self.metricFunction.compute(mode, self.radii_to_compute, XYs1, XYs2)

        if np.isnan(self.metricFunction.metricValues).any(): return

        if (mode, group1, indexReference, indexTarget) not in self.dictGxCxCxRValues.keys():
            self.dictGxCxCxRValues[
                (mode, group1, indexReference, indexTarget)] = self.metricFunction.metricValues

        else:
            self.dictGxCxCxRValues[
                (mode, group1, indexReference, indexTarget)] += self.metricFunction.metricValues

    def computeGxCxRMetric(self, mode, XYs1, indexReference, group1):
        self.metricFunction.compute(mode, self.radii_to_compute, XYs1)
        if np.isnan(self.metricFunction.metricValues).any(): return

        if (mode, group1, indexReference) not in self.dictGxCxRValues.keys():
            self.dictGxCxRValues[
                (mode, group1, indexReference)] = self.metricFunction.metricValues

        else:
            self.dictGxCxRValues[
                (mode, group1, indexReference)] += self.metricFunction.metricValues

    def computeGxCxCMetric(self, mode, XYs1, XYs2, indexReference, indexTarget, group1, totalNumberOfCellsInGroup=None):
        self.metricFunction.compute(mode, None, XYs1, XYs2, totalNumberOfCellsInGroup=totalNumberOfCellsInGroup)

        if np.isnan(self.metricFunction.metricValues).any(): return

        if (mode, group1, indexReference, indexTarget) not in self.dictGxCxCValues.keys():
            self.dictGxCxCValues[
                (mode, group1, indexReference, indexTarget)] = self.metricFunction.metricValues

        else:
            self.dictGxCxCValues[
                (mode, group1, indexReference, indexTarget)] += self.metricFunction.metricValues

    def compute(self):
        for mode, modetype in zip(self.modes, self.modetypes):

            if modetype == 'GxGxCxCxR' or modetype == "GxGxCxC":
                groupbyLabels = list(set(self.aData.obs[self.groupBys[0]]))
                for indexGroupbyReference in groupbyLabels:
                    for indexGroupbyTarget in groupbyLabels:
                        if indexGroupbyReference == indexGroupbyTarget: continue

                        ref_phenotypeList = list(set(self.aData.obs[self.groupBys[2]]))
                        tar_phenotypeList = list(set(self.aData.obs[self.groupBys[3]]))
                        for indexReference in ref_phenotypeList:
                            for indexTarget in tar_phenotypeList:

                                XYs1 = self.getXYPositions(phenotype=self.groupBys[2], selection=indexReference,
                                            edges=self.edges, groupby=self.groupBys[0], groupbyLabel=indexGroupbyReference)
                                XYs2 = self.getXYPositions(phenotype=self.groupBys[3], selection=indexTarget,
                                            edges=self.edges, groupby=self.groupBys[0], groupbyLabel=indexGroupbyTarget)

                                if modetype == "GxGxCxCxR":
                                    self.computeGxGxCxCxRMetric(mode, XYs1, XYs2, indexReference, indexTarget,
                                                                indexGroupbyReference, indexGroupbyTarget)
                                else:
                                    self.computeGxGxCxCMetric(mode, XYs1, XYs2, indexReference, indexTarget,
                                                                indexGroupbyReference, indexGroupbyTarget)

            elif modetype == "GxCxCxR" or modetype == "GxCxC":

                groupbyLabels = list(set(self.aData.obs[self.groupBys[0]]))
                for indexGroupbyReference in groupbyLabels:
                    totalNumberOfCellsInGroup = len(
                        self.getXYPositions(phenotype=None, selection=None, edges=self.edges,
                                            groupby=self.groupBys[0], groupbyLabel=indexGroupbyReference))

                    ref_phenotypeList = list(set(self.aData.obs[self.groupBys[1]]))
                    tar_phenotypeList = list(set(self.aData.obs[self.groupBys[2]]))
                    for indexReference in ref_phenotypeList:
                        for indexTarget in tar_phenotypeList:

                            XYs1 = self.getXYPositions(phenotype=self.groupBys[1], selection=indexReference,
                                                       edges=self.edges, groupby=self.groupBys[0],
                                                       groupbyLabel=indexGroupbyReference)
                            XYs2 = self.getXYPositions(phenotype=self.groupBys[2], selection=indexTarget,
                                                       edges=self.edges, groupby=self.groupBys[0],
                                                       groupbyLabel=indexGroupbyReference)

                            if modetype == "GxCxCxR":
                                self.computeGxCxCxRMetric(mode, XYs1, XYs2, indexReference, indexTarget,
                                                            indexGroupbyReference)
                            else:
                                self.computeGxCxCMetric(mode, XYs1, XYs2, indexReference, indexTarget,
                                                          indexGroupbyReference, totalNumberOfCellsInGroup=totalNumberOfCellsInGroup)

            elif modetype == "GxCxR":
                pass
            else:
                raise(Exception("Unsupported metric"))

    '''def loopOverTarPhenos(self, mode, modetype, XYs1, indexReference, indexGroupbyTarget=None, totalGroupCells=None):
        for indexTarget in self.tar_phenotypeList:
            XYs2 = self.getXYPositions(phenotype=self.target_phenotype, selection=indexTarget,
                                       edges=self.edges, groupby=self.groupby, groupbyLabel=indexGroupbyTarget)
            if modetype == 'GxCxCxR':
                self.computeGxCxCxRMetric(mode, XYs1, XYs2, indexReference, indexTarget, indexGroupbyTarget)
            elif modetype == 'GxCxC':
                 self.computeGxCxCMetric(mode, XYs1, XYs2, indexReference, indexTarget, totalGroupCells=totalGroupCells)

    def loopOverRefPhenos(self, indexGroupbyReference, indexGroupbyTarget):
        for indexReference in self.ref_phenotypeList:
            XYs1 = self.getXYPositions(phenotype=self.reference_phenotype, selection=indexReference,
                                       edges=self.edges, groupby=self.groupby, groupbyLabel=indexGroupbyReference)

            if XYs1.shape[0] < 2: break

            for mode, modetype in zip(self.modes, self.modetypes):
                if modetype == 'GxCxCxR' or modetype == 'GxCxC':
                    totalNumberOfCellsInGroup = len(self.getXYPositions(phenotype=None, selection=None, edges=self.edges,
                        groupby=self.groupby, groupbyLabel=indexGroupbyReference))
                    
                elif modetype == "GxGxCxCxR":
                    totalNumberOfCellsInGroup = len(self.getXYPositions(phenotype=None, selection=None, edges=self.edges,
                        groupby=self.groupby, groupbyLabel=indexGroupbyReference))

                    self.loopOverTarPhenos(mode, modetype, XYs1, indexReference, indexGroupbyTarget=indexGroupbyTarget,
                        totalGroupCells=totalNumberOfCellsInGroup)#Total number of cells (for infiltration)

                elif modetype == 'GxCxR':
                    self.computeGxCxRMetric(mode, XYs1, indexReference, indexGroupbyReference)

                else:
                    raise(Exception(""))'''

    '''def compute(self):

        if self.groupby:
            groupbyLabels = list(set(self.aData.obs[self.groupby]))
                        
            for indexGroupbyReference in groupbyLabels:
                for indexGroupbyTarget in groupbyLabels:
                    if indexGroupbyReference == indexGroupbyTarget:
                        continue

                    if self.requireNeighbours is not None and ([indexGroupbyReference, indexGroupbyTarget] not in
                            self.requireNeighbours) and ([indexGroupbyTarget, indexGroupbyReference] not in self.requireNeighbours):
                        continue

                    self.loopOverRefPhenos(indexGroupbyReference, indexGroupbyReference)


        else:
            self.loopOverRefPhenos(indexGroupbyReference=None, indexGroupbyTarget=None)'''



    def getXYPositions(self, phenotype, selection, edges, groupby=None, groupbyLabel=None):
        if groupbyLabel:
            if edges:
                if selection is not None:
                    return np.array(self.aData.obs[(self.aData.obs[phenotype] == selection) \
                                      & (self.aData.obs[edges] == True)
                                      & (self.aData.obs[groupby] == groupbyLabel)
                                     ][['x', 'y']])
                else:
                    return np.array(self.aData.obs[(self.aData.obs[edges] == True)
                                      & (self.aData.obs[groupby] == groupbyLabel)
                                     ][['x', 'y']])

            else:
                if selection is not None:
                    return np.array(self.aData.obs[(self.aData.obs[phenotype] == selection) \
                                      & (self.aData.obs[groupby] == groupbyLabel) \
                                     ][['x', 'y']])
                else:
                    return np.array(self.aData.obs[(self.aData.obs[groupby] == groupbyLabel)
                                     ][['x', 'y']])
        else:
            if edges:
                if selection is not None:
                    return np.array(self.aData.obs[(self.aData.obs[phenotype] == selection) \
                                      & (self.aData.obs[edges] == True)][['x', 'y']])
                else:
                    return np.array(self.aData.obs[self.aData.obs[edges] == True][['x', 'y']])

            else:
                if selection is not None:
                    return np.array(self.aData.obs[self.aData.obs[phenotype] == selection][['x', 'y']])
                else:
                    return np.array(self.aData.obs[['x', 'y']])

    def returnPhenoTypeSet(self, column):
        return list(set(self.aData[self.aData.obs[self.image_column].isin([self.image_name])].obs[column]))


def make_tree(d1=None, d2=None):
    #active_dimensions = [dimension for dimension in [d1,d2]]
    #points = np.c_[active_dimensions[0], active_dimensions[1]]
    return spatial.cKDTree(np.c_[d1, d2])

def calculate_crossk_ripley(r, d, x1, y1, x2, y2, besag_correction=None):
    if x1.shape[0] == 1 or x2.shape[0] == 1:
        return np.nan*np.empty(len(r), dtype=np.float32)
    rtree = make_tree(d1=x1, d2=y1)
    ttree = make_tree(d1=x2, d2=y2)

    counts = ttree.count_neighbors(rtree, r)
    besag = 1.0
    #TODO: find a way to compute the overlap of the inside part for the correction
    if besag_correction:
        besag = np.pi*r**2/1.0

    return 2.0*counts*d/(len(x1)*len(x2))*besag

def calculate_cross_k_auc(radii_to_compute, ripley_1, ripley_2):
    auc_diff = []
    for i in range(len(radii_to_compute)):
        ripley_int_1 = np.trapz(ripley_1[:i], radii_to_compute[:i])
        ripley_int_2 = np.trapz(ripley_2[:i], radii_to_compute[:i])

        auc_diff.append(ripley_int_1 - ripley_int_2)
    return auc_diff

def calculate_gcross(r, d, x1, y1, x2, y2):
    """
    Compute the G-Cross function with Poisson and area correction between two sets of points A and B.

    Parameters:
    - r: numpy array of shape (n) where n is the number of radius bins
    - points_A: numpy array of shape (n, 2), where n is the number of points in set A.
    - points_B: numpy array of shape (m, 2), where m is the number of points in set B.

    Returns:
    - distances: the range of distances (array of size num_bins).
    - g_cross_corrected: the corrected G-Cross values for each distance.
    """

    # Intensity (lambda) of points
    if d < 1e-2 or (len(x1) + len(x2)) / d < 1e-9:
        return np.nan*np.empty(len(r))
    lambda_B = (len(x1) + len(x2)) / d

    ## Compute the distance matrix between points
    rtree = make_tree(d1=x1, d2=y1)
    nearest_distances_A_to_B, _ = np.array(rtree.query(np.vstack([x2, y2]).T, 2))
    nearest_distances_A_to_B = np.array([min(a[a > 0]) for a in nearest_distances_A_to_B])
    #nearest_distances_A_to_B = nearest_distances_A_to_B[nearest_distances_A_to_B < r[-1]]
    if len(nearest_distances_A_to_B) == 0:
        return np.nan*np.empty(len(r))

    # Compute the ECDF of these distances (empirical G-Cross)
    ecdf = ECDF(nearest_distances_A_to_B)

    # Compute the raw G-Cross values at each distance
    g_cross_values = ecdf(r)

    # Poisson correction: G_poisson(r) = 1 - exp(-lambda_B * pi * r^2)
    #g_poisson = 1 - np.exp(-lambda_B * np.pi * (r+1e-3) ** 2)

    # Compute the corrected G-Cross function
    #if g_poisson < 1e-9:
    #    g_poisson = 1.0
    #g_cross_corrected = g_cross_values / g_poisson

    return g_cross_values


'''def computeKRipleyCrossAndGCross(modes, adata, tissue, groupby, referencePhenotype, targetPhenotype, phenotypes, reftype,
                            tartype, radii, image_column, reqNeighs=None, edges=None, gcross_radii=None):
    t0 = time.time()
    print(f"Compute K-Ripley Metrics of groupby: {groupby} for phenotype {phenotypes} and edges = {edges}")
    if edges is not None:
        assert edges in adata.obs.keys(), " Edge key not in given AnnData structure"
    clinicalCompRiplMet = ComputeMetricsAnnData(
        adata,
        tissue,
        modes,
        reference_phenotype=referencePhenotype,
        target_phenotype=targetPhenotype,
        radii_of_computation=radii,
        image_column=image_column,
        reference_type=reftype,
        target_type=tartype,
        groupby=groupby,
        requireNeighbours=reqNeighs,
        edges = edges,
        gcross_radii=gcross_radii
    )

    if not reftype:
        phenotypeList = clinicalCompRiplMet.returnPhenoTypeSet(phenotypes)  # Get all cell types
        clinicalCompRiplMet.reference_type = clinicalCompRiplMet.target_type = phenotypeList
    clinicalCompRiplMet.compute()
    label = 'None_' if groupby is None else ('CNs_' if 'labels' in groupby else 'Subclusters_')
    label += phenotypes.replace('_', '') + '_'
    label += 'EdgeCells' if edges is not None else 'AllCells'

    print(f"\tDone in {time.time() - t0} seconds.")

    return clinicalCompRiplMet.dictCrossValues, clinicalCompRiplMet.dictCrossCounts, label'''
