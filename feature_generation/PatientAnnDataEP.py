import numpy as np
import ehrapy as ep
import pandas as pd
import anndata as ad

class PatientAnnDataEP:
    def __init__(self,  Adata, metricsToCompute, CNGroupTypes, clinicalOutcomes=None, fileName=None, radii_of_computation=None):
        self.metricsToCompute = metricsToCompute
        self.clinicalOutcomes = clinicalOutcomes
        self.CNGroupTypes = CNGroupTypes
        self.fileName = fileName
        self.radii_of_computation = radii_of_computation

        self.patients = []
        obsdf = self.getObsDF()#columnNames=['OR'])
        self.padata = ad.AnnData(obs=obsdf)
        self.Adata = Adata

        self.padata.uns['patientIndices']    = {str(pat): i for i, pat in enumerate(self.patients)}

        #self.adata.uns['cell_types'] = np.array(list(set(cell_types)))
        #self.adata.uns['Immune_func'] = np.array(list(set(Immune_func)))
        #self.adata.uns['Immune_meta'] = np.array(list(set(Immune_meta)))

        #self.adata.uns['cellTypeIndices']   = {typ: i for i, typ in enumerate(self.adata.uns['cell_types'])}
        #self.adata.uns['immuneTypeIndices'] = {typ: i for i, typ in enumerate(self.adata.uns['Immune_func'])}
        #self.adata.uns['immuneMetaTypeIndices'] = {typ: i for i, typ in enumerate(self.adata.uns['Immune_meta'])}

        self.padata.uns['CNGroupTypes'] = CNGroupTypes


        self.initializeMetricsInEPStructure()

        print(self.padata)


    def read(self):
        if self.fileName:
            self.padata = ad.read_h5ad(self.fileName)

    def write(self):
        if self.fileName:
            self.padata.write(self.fileName)
        else:
            raise(Exception("Please provide a filename"))

    def getObsDF(self, columnNames=['Response', 'GroupedResponse', 'OSTime', 'OSEvent']):
        obs = []
        for i, ii in self.clinicalOutcomes.items():
            obs.append(ii)
            self.patients.append(i)
        return pd.DataFrame(obs, index=self.patients, columns=columnNames)

    def initializeMetricsInEPStructure(self):
        for metricLabel, metric in self.metricsToCompute.items():

            shape = [len(self.patients)]
            for group in metric['groupBys']:
                if not group in self.padata.uns.keys():
                    if group in [a for a, b in self.padata.uns['CNGroupTypes'].items()]:
                        self.padata.uns[group] = self.padata.uns['CNGroupTypes'][group]
                    else:
                        self.padata.uns[group] = np.array(list(set(self.Adata.obs[group])))

                shape.append(len(self.padata.uns[group]))
            if metric['radii'] is not None:
                self.padata.uns['radii'] = metric['radii']
                shape.append(len(metric['radii']))

            vals = np.nan*np.empty(shape, dtype=np.float32)

            metrics_withradii = metric['groupBys']
            if metric['radii'] is not None:
                metrics_withradii.append('radii')
            self.padata.obsm[metricLabel] = vals.copy()
            self.padata.uns[metricLabel + '_dims'] = {"dims": {str(i): lab for i, lab in enumerate(metrics_withradii)},
                                                      "dimLabels": {str(i): self.padata.uns[group] for i, group in enumerate(metrics_withradii)},
                                                         }

        '''
        vals3D, vals2D, vals1D, labels3D, labels2D, labels1D = [], [], [], [], [], []
        dimGroupbyLabels3D, dimGroupbyLabels2D, dimGroupbyLabels1D = [], [], []
        for i, ii in self.metricsToCompute.items():
            print("i: ", i, "\tii: ", ii)
            #TODO: Rename from 1D, 2D, etc., to dimension types of tensors
            if ii[0] in ['KRipleyCross', 'GCross']:
                if 'GxG' in ii[-1]:
                    vals = np.nan*np.empty((len(self.patients), len(self.CNGroupTypes), len(self.CNGroupTypes),
                                            len(self.adata.uns[ii[1]]), len(self.adata.uns[ii[2]]), len(ii[3])),
                                           dtype=np.float32)
                elif 'Gx' in ii[-1]:
                    vals = np.nan*np.empty((len(self.patients), len(self.CNGroupTypes),
                                           len(self.adata.uns[ii[1]]), len(self.adata.uns[ii[2]]), len(ii[3])),
                                           dtype=np.float32)
                else:
                    raise(Exception("Unknown metric type"))
                vals3D.append(vals)
                dimGroupbyLabels3D.append('')
                labels3D.append(i)
            elif ii[0] in ['KRipleyOneClass', 'GOneClass']:
                if 'Gx' in ii[-1] and 'GxGx' not in ii[-1]:
                    vals = np.nan*np.empty((len(self.patients), len(self.CNGroupTypes), len(self.adata.uns[ii[2]]),
                                            len(ii[3])), dtype=np.float32)
                else:
                    raise(Exception("Unknown metric type"))
                vals2D.append(vals)
                dimGroupbyLabels2D.append('')
                labels2D.append(i)
            elif ii[0] in ['InfiltrationScores']:
                if 'Gx' in ii[-1] and 'GxGx' not in ii[-1]:
                    vals = np.nan*np.empty((len(self.patients), len(self.CNGroupTypes), len(self.adata.uns[ii[1]]),
                                            len(self.adata.uns[ii[2]])), dtype=np.float32)
                else:
                    raise(Exception("Unknown metric type"))
                vals1D.append(vals)
                dimGroupbyLabels1D.append('')
                labels1D.append(i)
            elif ii[0] in ['JSDScores']:
                if 'Gx' in ii[-1] and 'GxGx' not in ii[-1]:
                    vals = np.nan*np.empty((len(self.patients), len(self.CNGroupTypes), len(self.adata.uns[ii[1]]),
                                            len(self.adata.uns[ii[2]])), dtype=np.float32)
                else:
                    raise(Exception("Unknown metric type"))
                vals1D.append(vals)
                dimGroupbyLabels1D.append('')
                labels1D.append(i)
            else:
                print(ii[0], " not in list of computable metrics")
                raise(Exception("Requested metric not supported"))

        for i in range(len(labels3D)):
            vals = np.array(np.nan*np.zeros(vals3D[i].shape))
            self.adata.obsm[labels3D[i]] = vals.copy()
            self.adata.uns[labels3D[i]+'_dims'] = {"dims": dimGroupbyLabels3D,
                                                   "dimLabels": [[] for _ in range(len(dimGroupbyLabels3D))],
                                                   }
        for i in range(len(labels2D)):
            vals = np.array(vals2D[i])
            self.adata.obsm[labels2D[i]] = vals
            self.adata.uns[labels2D[i] + '_dims'] = {"dims": dimGroupbyLabels2D,
                                                     "dimLabels": [[] for _ in range(len(dimGroupbyLabels2D))],
                                                     }
        for i in range(len(labels1D)):
            vals = np.array(vals1D[i])
            self.adata.obsm[labels1D[i]] = vals
            self.adata.uns[labels1D[i] + '_dims'] = {"dims": dimGroupbyLabels1D,
                                                     "dimLabels": [[] for _ in range(len(dimGroupbyLabels1D))],
                                                     }'''

    def updateValues(self, computedmetrics, metricDict, patient, label):#, ref_phenotypes, tar_phenotypes, selection=None, metrictype=None):

        metrictype = metricDict['storedType']
        groupby = metricDict['groupBys']

        for i, ii in computedmetrics.items():
            if metrictype == 'GxGxCxCxR':
                self.padata.obsm[label][self.padata.obs.index == patient, np.array(self.padata.uns['CNGroupTypes'][groupby[0]][str(i[1])]), np.array(self.padata.uns['CNGroupTypes'][groupby[1]][str(i[2])]), self.padata.uns[groupby[2]] == i[3],
                        self.padata.uns[groupby[3]] == i[4], :] = ii
                self.padata.uns[label+'_dims']['dims'] = {'0': groupby[0],
                                                          '1': groupby[1],
                                                          '2': groupby[2],
                                                          '3': groupby[3],
                                                          '4': 'radius'}
                self.padata.uns[label+'_dims']['dimLabels'] = {'0': self.padata.uns[groupby[0]],
                                                               '1': self.padata.uns[groupby[1]],
                                                               '2': self.padata.uns[groupby[2]],
                                                               '3': self.padata.uns[groupby[3]],
                                                               '4': list(self.radii_of_computation)}
            elif metrictype == 'GxGxCxC':
                self.padata.obsm[label][self.padata.obs.index == patient, np.array(self.padata.uns['CNGroupTypes'][groupby[0]][str(i[1])]), np.array(self.padata.uns['CNGroupTypes'][groupby[1]][str(i[2])]), self.padata.uns[groupby[2]] == i[3],
                        self.padata.uns[groupby[3]] == i[4]] = ii
                self.padata.uns[label+'_dims']['dims'] = {'0': groupby[0],
                                                          '1': groupby[1],
                                                          '2': groupby[2],
                                                          '3': groupby[3]}
                self.padata.uns[label+'_dims']['dimLabels'] = {'0': self.padata.uns[groupby[0]],
                                                               '1': self.padata.uns[groupby[1]],
                                                               '2': self.padata.uns[groupby[2]],
                                                               '3': self.padata.uns[groupby[3]]}
            elif metrictype == 'GxCxCxR':
                self.padata.obsm[label][self.padata.obs.index == patient, np.array(self.padata.uns['CNGroupTypes'][groupby[0]][str(i[1])]),
                        self.padata.uns[groupby[1]] == i[2], self.padata.uns[groupby[2]] == i[3], :] = ii
                self.padata.uns[label+'_dims']['dims'] = {'0': groupby[0],
                                                          '1': groupby[1],
                                                          '2': groupby[2],
                                                          '3': 'radius'}
                self.padata.uns[label+'_dims']['dimLabels'] = {'0': self.padata.uns[groupby[0]],
                                                               '1': self.padata.uns[groupby[1]],
                                                               '2': self.padata.uns[groupby[2]],
                                                               '3': list(self.radii_of_computation)}
            elif metrictype == 'GxCxC':
                self.padata.obsm[label][self.padata.obs.index == patient, np.array(self.padata.uns['CNGroupTypes'][groupby[0]][str(i[1])]),
                        self.padata.uns[groupby[1]] == i[2], self.padata.uns[groupby[2]] == i[3]] = ii
                self.padata.uns[label+'_dims']['dims'] = {'0': groupby[0],
                                                          '1': groupby[1],
                                                          '2': groupby[2]}
                self.padata.uns[label+'_dims']['dimLabels'] = {'0': self.padata.uns[groupby[0]],
                                                               '1': self.padata.uns[groupby[1]],
                                                               '2': self.padata.uns[groupby[2]]}
            elif metrictype == 'GxCxR':
                self.padata.obsm[label][self.padata.obs.index == patient, np.array(self.padata.uns['CNGroupTypes'][groupby[0]][str(i[1])]),
                                        self.padata.uns[groupby[1]] == i[2], :] = ii
                self.padata.uns[label+'_dims']['dims'] = {'0': groupby[0],
                                                          '1': groupby[1],
                                                          '2': 'radius'}
                self.padata.uns[label+'_dims']['dimLabels'] = {'0': self.padata.uns[groupby[0]],
                                                               '1': self.padata.uns[groupby[1]],
                                                               '2': list(self.radii_of_computation)}
            else:
                raise(Exception("metric type not in list"))

'''class PatientAnnData:
    def __init__(self, patients, metricsToCompute, cell_types, scalarMetrics=None, immune_types = None,
                 clinicalOutcomes = None,
                 radii_of_computation=np.arange(1, 100, 1), fileName='patientAnnData.h5ad',
                 gcross_radii=[]):
        self.radii_of_computation = radii_of_computation

        if len(gcross_radii) > 0:
            self.gcross_radii = gcross_radii
        else:
            print("Defaulting to standard radii for computations")
            self.gcross_radii = radii_of_computation
        self.cell_types = np.array(list(set(cell_types)))
        self.immune_types = np.array(list(set(immune_types)))
        self.patients = np.array(list(set(patients)))
        self.clinicalOutcomes = clinicalOutcomes
        self.fileName = fileName
        self.metricsToCompute = metricsToCompute
        self.scalarMetrics = scalarMetrics

        assert len(self.patients) > 0 and len(self.cell_types) > 0

        self.patientInd    = {pat: i for i, pat in enumerate(self.patients)}
        self.cellTypeInd   = {typ: i for i, typ in enumerate(self.cell_types)}
        self.immuneTypeInd = {typ: i for i, typ in enumerate(self.immune_types)}

        self.patientAnnData = None

        arr = [[[] for i in range(len(self.cell_types))] for j in range(len(self.cell_types))]

        self.metric3DArrCxC = {metric: pd.DataFrame(arr, columns=self.cell_types) for metric in self.metricsToCompute
                               +self.scalarMetrics}
        for metric in self.metricsToCompute+self.scalarMetrics:
            self.metric3DArrCxC[metric].index = self.cell_types

        arr = [[[] for i in range(len(self.cell_types))] for j in range(len(self.immune_types))]

        self.metric3DArrCxK = {metric: pd.DataFrame(arr, columns=self.cell_types) for metric in self.metricsToCompute
                               +self.scalarMetrics}
        for metric in self.metricsToCompute+self.scalarMetrics:
            self.metric3DArrCxK[metric].index = self.immune_types

    def writeValuesToArrays(self, metric, values, label, count):
        #TODO: Fix this ridiculous method of sorting the types
        if 'cell' in label:
            self.metric3DArrCxC[metric[0]].loc[metric[1], metric[2]] = values/count
        else:
            self.metric3DArrCxK[metric[0]].loc[metric[1], metric[2]] = values/count

        return

    def refactorToNormalDF(self):
        alist = [[b.replace('\'', '') for b in a.split('(')[1].split(')')[0].split(',')] for a in
                 self.patientAnnData.obs.index.tolist()]
        aadata = self.patientAnnData.obs.reset_index(drop=True)
        aadata[['patient', 'metrictype', 'targetcells', 'referencecells', 'radius', 'selection']] = pd.DataFrame(alist,
                                                                                                        dtype=str)
        self.patientAnnData.obs = aadata

    def refactorToTupleIndex(self):
        self.patientAnnData.obs = self.patientAnnData.obs \
            .set_index(['patient', 'metrictype', 'targetcells', 'referencecells', 'radius', 'selection'])

    def updateClinicalValues(self, patient, key, value):
        assert self.patientAnnData, ' Create patient AnnData first'
        if not key in self.patientAnnData.obs.keys():
            self.patientAnnData.obs[key] = 'NaN'

        #self.refactorToNormalDF()
        self.patientAnnData.obs.loc[self.patientAnnData.obs['patient'] == patient, key] = value
        #self.refactorToTupleIndex()

    def updateValues(self, metrics, metriccounts, tissue, selection='Default', newPatient=False):

        for metricCxC, values in metrics.items():
            if len(metricCxC) < 3:
                continue

            self.writeValuesToArrays(metricCxC, values, selection, metriccounts[metricCxC])

        vals = []
        for metric in self.metricsToCompute:
            keys = self.metric3DArrCxC[metric].keys()
            indices = self.metric3DArrCxC[metric].index
            for i in range(len(self.metric3DArrCxC[metric].index)):
                for j in range(len(keys)):
                    for k in range(len(self.metric3DArrCxC[metric].iloc[i, j])):
                        vals.append([selection, tissue, self.clinicalOutcomes[tissue][0], self.clinicalOutcomes[tissue][1],
                                     self.clinicalOutcomes[tissue][2], self.clinicalOutcomes[tissue][3],
                                     metric, indices[i], keys[j], str(self.radii_of_computation[k]),
                                     self.metric3DArrCxC[metric].iloc[i, j][k]])
        df = pd.DataFrame(vals, columns=['selection', 'patient', 'clinicalOutcomes', 'groupedOutcomes', 'OSTime',
                                         'OSEvent', 'metrictype', 'targetcells', 'referencecells', 'metric_x', 'metric'])
        df = df.set_index(['patient', 'metrictype', 'targetcells', 'referencecells', 'metric_x', 'selection'])
        df.index = df.index.tolist()

        if self.patientAnnData is None:
            self.patientAnnData = ad.AnnData(obs=df)
        elif newPatient:
            self.patientAnnData = ad.AnnData(obs=pd.concat([self.patientAnnData.obs, df], axis=0))
            self.patientAnnData.uns['cell_types'] = self.cell_types
            self.patientAnnData.uns['immune_types'] = self.immune_types
        else:
            self.patientAnnData = ad.AnnData(obs=pd.concat([self.patientAnnData.obs, df], axis=0))

            self.patientAnnData.uns['cell_types'] = self.cell_types
            self.patientAnnData.uns['immune_types'] = self.immune_types

        return

    def updateValuesScalar(self, metrics, metriccounts, tissue, selection='', newPatient=False):

        for metricCxC, values in metrics.items():
            if len(metricCxC) < 3:
                continue

            self.writeValuesToArrays(metricCxC, values, selection, metriccounts[metricCxC])

        vals = []
        for metric in self.scalarMetrics:
            keys = self.metric3DArrCxC[metric].keys()
            indices = self.metric3DArrCxC[metric].index
            for i in range(len(indices)):#self.metric3DArrCxC[metric]
                for j in range(len(keys)):#keys[0]
                    if (type(self.metric3DArrCxC[metric].iloc[i, j]) == list and
                            len(self.metric3DArrCxC[metric].iloc[i, j]) == 0):
                        continue
                    else:
                        vals.append([selection, tissue, self.clinicalOutcomes[tissue][0],
                                     self.clinicalOutcomes[tissue][1], self.clinicalOutcomes[tissue][2],
                                     self.clinicalOutcomes[tissue][3], metric, indices[i], keys[j], '0',
                                 self.metric3DArrCxC[metric].iloc[i, j]])
        df = pd.DataFrame(vals, columns=['selection', 'patient', 'clinicalOutcomes', 'groupedOutcomes', 'OSTime',
                                         'OSEvent', 'metrictype', 'targetcells', 'referencecells', 'metric_x', 'metric'])
        df = df.set_index(['patient', 'metrictype', 'targetcells', 'referencecells', 'metric_x', 'selection'])
        df.index = df.index.tolist()

        if self.patientAnnData is None:
            self.patientAnnData = ad.AnnData(obs=df)
        elif newPatient:
            self.patientAnnData = ad.AnnData(obs=pd.concat([self.patientAnnData.obs, df], axis=0))
            self.patientAnnData.uns['cell_types'] = self.cell_types
            self.patientAnnData.uns['immune_types'] = self.immune_types
        else:
            self.patientAnnData = ad.AnnData(obs=pd.concat([self.patientAnnData.obs, df], axis=0))
            self.patientAnnData.uns['cell_types'] = self.cell_types
            self.patientAnnData.uns['immune_types'] = self.immune_types

        return

    def getObservationPandasDataFrame(self):
        df = self.patientAnnData.obs

    def write(self):
        print(self.patientAnnData.obs)
        #embed()
        self.patientAnnData.write_h5ad(self.fileName)'''
