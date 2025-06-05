import numpy as np
from scipy.integrate import simpson
from sklearn.neighbors import KernelDensity

class Spatial2DDensityCorrelation:
    def __init__(self, adata, referencePhenotype, targetPhenotype,
                 bandwidth=80.0):
        self.adata = adata
        self.referencePhenotype = referencePhenotype
        self.targetPhenotype = targetPhenotype

        self.bandwidth = bandwidth

        self.referenceKDE = None
        self.targetKDE = None
        self.referenceCellXYs = None
        self.targetCellXYs = None
        self.X = None
        self.Y = None
        self.Xmin = self.Xmax = self.Ymin = self.Ymax = None

        if self.referencePhenotype not in self.adata.obs.keys() or self.targetPhenotype not in self.adata.obs.keys():
            raise(Exception(f"\"{self.referencePhenotype}\" or \"{self.targetPhenotype}\" not in given anndata structure"))


    def computeKDE(self, group1, phenotype1, group2, phenotype2):
        cellXs = self.adata.obs[(self.adata.obs[group1] == phenotype1) & (self.adata.obs[group2] == phenotype2)]['x']
        cellYs = self.adata.obs[(self.adata.obs[group1] == phenotype1) & (self.adata.obs[group2] == phenotype2)]['y']

        cellXYs = np.array([[x, y] for x, y in zip(cellXs, cellYs)])
        if len(cellXYs) > 4:
            return KernelDensity(kernel='gaussian', bandwidth=self.bandwidth, algorithm='kd_tree', leaf_size=100, atol=1e-9).fit(cellXYs)
        else:
            return []

    def returnXYMinMax(self):
        return np.array([self.Xmin, self.Xmax, self.Ymin, self.Ymax])

    def storeMinMax(self, referenceLabel, targetLabel, group, groupby):
        referenceCellXs = self.adata.obs[(self.adata.obs[self.referencePhenotype] == referenceLabel) &
                                         (self.adata.obs[groupby] == group)]['x']
        referenceCellYs = self.adata.obs[(self.adata.obs[self.referencePhenotype] == referenceLabel) &
                                         (self.adata.obs[groupby] == group)]['y']
        targetCellXs = self.adata.obs[(self.adata.obs[self.targetPhenotype] == targetLabel) &
                                      (self.adata.obs[groupby] == group)]['x']
        targetCellYs = self.adata.obs[(self.adata.obs[self.targetPhenotype] == targetLabel) &
                                      (self.adata.obs[groupby] == group)]['y']

        self.Xmin, self.Xmax, self.Ymin, self.Ymax = (min(referenceCellXs.min(), targetCellXs.min()),
                                                      max(referenceCellXs.max(), targetCellXs.max()),
                                                      min(referenceCellYs.min(), targetCellYs.min()),
                                                      max(referenceCellYs.max(), targetCellYs.max()))

    def JSD(self, referenceKDE, targetKDE):

        X1D, Y1D = (np.arange(self.Xmin, self.Xmax, 15),
                    np.arange(self.Ymin, self.Ymax, 15))
        #(self.Xmax - self.Xmin) / 200.0

        X, Y = np.meshgrid(X1D, Y1D)
        XY = np.vstack([X.flatten(), Y.flatten()]).T

        #t0 = time.time()
        ZReference = np.exp(referenceKDE.score_samples(XY).reshape((len(X1D), len(Y1D))))

        ZTarget = np.exp(targetKDE.score_samples(XY).reshape((len(X1D), len(Y1D))))
        #print(f"KDEs sampled in {time.time() - t0} s")

        eps = 1e-20
        term1 = 2.0 * ZReference / (ZReference + ZTarget + eps)
        term1[np.isnan(term1) | (term1 < eps)] = eps
        term1 = ZReference * np.log2(term1)
        term1[np.isnan(term1) | (term1 < eps)] = eps

        term2 = 2.0 * ZTarget / (ZReference + ZTarget + eps)
        term2[np.isnan(term2) | (term2 < eps)] = eps
        term2 = ZTarget * np.log2(term2)
        term2[np.isnan(term2) | (term2 < eps)] = eps

        term1 = simpson(simpson(term1, x=Y1D), x=X1D)
        term2 = simpson(simpson(term2, x=Y1D), x=X1D)

        #print(f"\tJSD is {np.sqrt(term1 / 2.0 + term2 / 2.0)}")
        return np.sqrt(term1 / 2.0 + term2 / 2.0)

class Spatial2DDensityCorrelationAnnData:
    def __init__(self, adata, patientAnnData, metric, bandwidth = 80.0):
        self.adata = adata
        self.patientAnnDataClass = patientAnnData
        self.metric = metric
        self.bandwidth = bandwidth

        self.s2DCorr = Spatial2DDensityCorrelation(self.adata, metric['groupBys'][-2], metric['groupBys'][-1],  bandwidth=bandwidth)
        self.corrValues = {}
        self.KDEs = {}

    def compute(self):
        print("here!")
        print(self.metric)
        assert len(self.metric['groupBys']) == 3, 'Can only do GxCxC for JSD scores'

        # TODO: probably computes some unnecessary pairwise combinations
        # TODO: Also this only works because phenotypes and groups are not similar (i.e. ints together)
        pairs = []
        for group1 in self.metric['groupBys']:
             for phenoOrGroup1 in self.patientAnnDataClass.padata.uns[group1]:
                 for group2 in self.metric['groupBys']:
                     if group1 != group2:
                        for phenoOrGroup2 in self.patientAnnDataClass.padata.uns[group2]:
                            if phenoOrGroup1 != phenoOrGroup2:# else would select objects simultaneously in two cell types
                                if [phenoOrGroup1, phenoOrGroup2] not in pairs:
                                    pairs.append([phenoOrGroup1, phenoOrGroup2])
                                    self.KDEs[(phenoOrGroup1,phenoOrGroup2)] = self.s2DCorr.computeKDE(group1, phenoOrGroup1, group2, phenoOrGroup2)

        # self.KDEs looks like (['CN1', 'cellA'], ['CN2', 'CellC'], ...), etc.

        for group in self.patientAnnDataClass.padata.uns[self.metric['groupBys'][0]]:
            types = []
            for referenceType in self.patientAnnDataClass.padata.uns[self.metric['groupBys'][1]]:
                for targetType in self.patientAnnDataClass.padata.uns[self.metric['groupBys'][2]]:
                    if (targetType, referenceType) in types:
                        continue
                    types.extend([(referenceType, targetType), (targetType, referenceType)])
                    #print("Reference: ", referenceType, "\tTarget: ", targetType)

                    self.s2DCorr.storeMinMax(referenceType, targetType, group=group, groupby=self.metric['groupBys'][0])
                    if ((self.s2DCorr.Xmax - self.s2DCorr.Xmin) > 0) and ((self.s2DCorr.Ymax - self.s2DCorr.Ymin) > 0):
                        if type(self.KDEs[(referenceType,group)]) is not list and type(self.KDEs[(targetType,group)]) is not list:
                            self.corrValues[('JSDScores', group, referenceType, targetType)] = self.s2DCorr.JSD(self.KDEs[(referenceType,group)],
                                                                                              self.KDEs[(targetType,group)])



        return self.corrValues
