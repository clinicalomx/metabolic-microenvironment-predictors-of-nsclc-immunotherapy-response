import sys
import numpy as np
from scipy.spatial.distance import cdist
from ComputeRipleyMetrics import make_tree


class ComputeScalarMetrics:
    def __init__(self, metric, metricFunction=None):
        self.values = np.empty([])
        self.metric = metric

        # Currently only implemented for InfiltrationScore
        if self.metric not in ['InfiltrationScore']:
            if metricFunction is None:
                raise Exception("Custom metric requested but no function passed")
            else:
                self.metricFunction = metricFunction

    def compute(self, XY1, XY2, extras=None, diagonal=False):

        # Diagonal is cell enrichment relative to all cells.  Off-diagonals are relative to other cells types
        if self.metric == 'InfiltrationScore':
            assert extras, ' Need total cell quantities to compute diagonal components'
            X1 = XY1[:, 0]
            X2 = XY2[:, 0]
            if diagonal:
                self.values = len(X1) / extras#, len(X2) / extras[1])
            else:
                if len(X1) > 0:
                    self.values = len(X2) / len(X1)
                else:
                    self.values = np.nan
        else:
            self.values = self.metricFunction(XY1, XY2, extras, diagonal)

class ComputeScalarMetricsAnnData:
    def __init__(self, adata, metrics, phenotypes, phenotypeList, groupby, requireNeighbours=None, edges=None):
        self.adata = adata
        self.phenotypes = phenotypes
        self.phenotypeList = phenotypeList
        self.groupby = groupby
        self.metrics = metrics
        self.requireNeighbours = requireNeighbours
        self.edges = edges
        self.computeScalarMetrics = None

        self.dictValues = {}
        self.dictCounts = {}

    def compute(self):
        for metric in self.metrics:
            self.computeScalarMetrics = ComputeScalarMetrics(metric)
            if metric == 'InfiltrationScore':

                groupbyLabels = list(set(self.adata.obs[self.groupby]))
                for indexGroupbyReference in groupbyLabels:

                    numCellsInReference = len(self.adata.obs[self.adata.obs[self.groupby].isin([indexGroupbyReference])])

                    #phenotypeList = list(set(self.adata.obs[self.phenotypes]))
                    for indexReference in self.phenotypeList:
                        XYs1 = self.getXYPositions(phenotype=self.phenotypes, selection=indexReference,
                                                   edges=self.edges, groupby=self.groupby,
                                                   groupbyLabel=indexGroupbyReference)

                        for indexTarget in self.phenotypeList:
                            XYs2 = self.getXYPositions(phenotype=self.phenotypes, selection=indexTarget,
                                            edges=self.edges, groupby=self.groupby, groupbyLabel=indexGroupbyReference)
                            self.computeScalarMetrics.compute(XYs1, XYs2, extras=numCellsInReference,
                                                              diagonal=(indexTarget==indexReference))

                            if (metric, indexReference, indexTarget) not in self.dictValues.keys():
                                self.dictValues[
                                    (metric, indexReference, indexTarget)] = self.computeScalarMetrics.values
                                self.dictCounts[(metric, indexReference, indexTarget)] = 1
                            else:
                                self.dictValues[
                                    (metric, indexReference, indexTarget)] += self.computeScalarMetrics.values \
                                        if ~np.isnan(self.computeScalarMetrics.values) else 0.0
                                self.dictCounts[(metric, indexReference, indexTarget)] += 1

    def getXYPositions(self, phenotype, selection, edges, groupby=None, groupbyLabel=None):
        if edges:
            if selection is not None:
                return np.array(self.adata.obs[(self.adata.obs[phenotype] == selection) \
                                  & (self.adata.obs[edges] == True)
                                  & (self.adata.obs[groupby] == groupbyLabel)
                                 ][['x', 'y']])
            else:
                return np.array(self.adata.obs[(self.adata.obs[edges] == True)
                                  & (self.adata.obs[groupby] == groupbyLabel)
                                 ][['x', 'y']])
        else:
            if selection is not None:
                return np.array(self.adata.obs[(self.adata.obs[phenotype] == selection) \
                                  & (self.adata.obs[groupby] == groupbyLabel) \
                                 ][['x', 'y']])
            else:
                return np.array(self.adata.obs[(self.adata.obs[groupby] == groupbyLabel)
                                 ][['x', 'y']])



class ComputeNeighbourhoodCounts:
    def __init__(self, radii_of_computation):
        self.radii_of_computation = radii_of_computation
        self.minimalDistanceToTarget = np.empty([])
        self.minimalDistance = np.empty([])

        # Computables
        self.cins = None
        self.nms  = None

    def computeCIN(self, rx, ry, tx, ty):
        rtree = make_tree(d1=rx, d2=ry)
        ttree = make_tree(d1=tx, d2=ty)

        counts = np.zeros(len(self.radii_of_computation))
        for rI, r in enumerate(self.radii_of_computation):
            for x, y in zip(rx, ry):
                counts[rI] += len(ttree.query_ball_point([x, y], r)) - 1

        self.cins = np.array(counts, dtype=np.float32)
        return self.cins

    def computeMinimalDistance(self, rx, ry):
        tree = make_tree(d1=rx, d2=ry)
        dists = cdist(np.vstack([rx, ry]), np.vstack([rx, ry]))
        self.minimalDistance = np.min(dists[dists > sys.float_info.epsilon])
        return self.minimalDistance

    def computeMinimalDistanceToTarget(self, rx, ry, tx, ty):
        rtree = make_tree(d1=rx, d2=ry)
        ttree = make_tree(d1=tx, d2=ty)
        amd = 0.0

        for x, y in zip(rx, ry):
            results = ttree.query_ball_point([x, y], max(rx) - min(rx))
            amd += np.sqrt((x - tx[results])**2 + (y - ty[results])**2).min()
        amd /= len(rx)
        self.minimalDistanceToTarget = amd
        return self.minimalDistanceToTarget

    #                    rx  ry  tx  ty
    def computeNMS(self, rx, ry, tx, ty):
        rtree = make_tree(d1=rx, d2=ry)
        ttree = make_tree(d1=tx, d2=ty)

        countsNumerator   = np.zeros(len(self.radii_of_computation))
        countsDenominator = np.zeros(len(self.radii_of_computation))
        for rI, r in enumerate(self.radii_of_computation):
            for x, y in zip(rx, ry):
                countsNumerator[rI] += len(ttree.query_ball_point([x, y], r)) - 1
            for x, y in zip(rx, ry):
                countsDenominator[rI] += len(rtree.query_ball_point([x, y], r)) - 1
        countsNumerator *= len(rx) - 1
        countsDenominator *= 2.0*len(tx)

        self.nms = countsNumerator/countsDenominator

        return self.nms

class ComputeNeighbourhoodCountsAnnData:
    def __init__(self, adata,
                 radii_of_computation,
                 phenotype='tumournontumour',
                 target='tumour',
                 reference='nontumour'):
        self.radii_of_computation = radii_of_computation
        self.adata = adata
        self.phenotype = phenotype
        self.reference = reference
        self.target = target

        # Computables
        self.cin = None
        self.nms = None
        self.minimalDistance = None
        self.minimalDistanceToTarget = None

        # Other classes
        self.computeNeighbourhoodCounts = ComputeNeighbourhoodCounts(self.radii_of_computation)

    def computeCINAndNMS(self):
        self.computeCIN()
        self.computeNMS()

    def computeCIN(self):
        self.cin = self.computeNeighbourhoodCounts.computeCIN(self.adata[self.adata.obs[self.phenotype].isin(
                                                              [self.reference])].obs['x'],
                                                          self.adata[self.adata.obs[self.phenotype].isin(
                                                              [self.reference])].obs['y'],
                                                          self.adata[self.adata.obs[self.phenotype].isin(
                                                              [self.target])].obs['x'],
                                                          self.adata[self.adata.obs[self.phenotype].isin(
                                                              [self.target])].obs['y'],
                                               )

    def computeNMS(self):
        self.nms = self.computeNeighbourhoodCounts.computeNMS(self.adata[self.adata.obs[self.phenotype].isin(
                                                              [self.reference])].obs['x'],
                                                          self.adata[self.adata.obs[self.phenotype].isin(
                                                              [self.reference])].obs['y'],
                                                          self.adata[self.adata.obs[self.phenotype].isin(
                                                              [self.target])].obs['x'],
                                                          self.adata[self.adata.obs[self.phenotype].isin(
                                                              [self.target])].obs['y'],
                                               )

    def computeMinimalDistance(self):
        self.minimalDistance = self.computeNeighbourhoodCounts.computeMinimalDistance(self.adata[self.adata.obs[\
                                        self.phenotype].isin([self.reference])].obs['x'],
                                        self.adata[self.adata.obs[self.phenotype]
                                            .isin([self.reference])].obs['y'],
                                    )
        return self.minimalDistance

    def computeMinimalDistanceToTarget(self):
        self.minimalDistanceToTarget = self.computeNeighbourhoodCounts.computeMinimalDistanceToTarget(self.adata[self.adata.obs[self.phenotype].isin(
                                                             [self.reference])].obs['x'],
                                                          self.adata[self.adata.obs[self.phenotype].isin(
                                                             [self.reference])].obs['y'],
                                                          self.adata[self.adata.obs[self.phenotype].isin(
                                                             [self.target])].obs['x'],
                                                          self.adata[self.adata.obs[self.phenotype].isin(
                                                             [self.target])].obs['y'])
        return self.minimalDistanceToTarget
