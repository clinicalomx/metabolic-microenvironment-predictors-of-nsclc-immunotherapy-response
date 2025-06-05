from ClusterPoints import *
import numpy as np
import scanpy as sc
import anndata as ad
import re

class ReadClinicalData:
    """
        ReadClinicalData: reads


            inputs: str: file_location  (default=D:\QSBC_BMS\qsbc_bms_celltypedfinal2.h5ad)
            stored: file_location: str
                    adata: AnnData
                    imageAData: AnnData
                    imageName: str

            functions:
                    # Slices images into distinct tissues (for whole-slide samples)
                    sliceImages(slice_dict: dict)

    """

    def __init__(self, file_location=r'D:\QSBC_BMS\qsbc_bms_celltypedfinal2.h5ad', patients="Image",
                 colsToFillNaNToZero=None, restrictToPatientList=None):
        self.file_location = file_location
        self.adata = ad.read_h5ad(self.file_location)
        self.patientImageId = patients
        self.imageAData = None
        self.imageName = None
        self.clusterlabels = {}
        self.restrictToPatientList = restrictToPatientList

        if self.restrictToPatientList:
            self.adata = self.adata[self.adata.obs[self.patientImageId].isin(self.restrictToPatientList)]

        #colstofill = ['pfs.12months', 'pfs.6months', 'OS.18months', 'OS.24months']
        if colsToFillNaNToZero:
            for col in colsToFillNaNToZero:
                self.adata.obs[col] = self.adata.obs[col].fillna(0)
        self.adata.obs = self.adata.obs.rename(columns=lambda x: x.strip())

        # Bounds of images, [minX, maxX, minY, maxY, angle]
        # Test [3000, 8000, 3000, 8000, 0.0],
        '''self.sliceImages({'A514CR417X': [[None, None, 0, 9000, 0.0],
                                         [None, None, 9000, 17000, 0.0],
                                         [None, None, 17000, None, 0.0]
                                        ],
                          'A514CR418p': [[None, None, None, 10000, 60.0],
                                         [None, None, 10000, None, 60.0]
                                        ],
                          'A514CR163':  [[None, None, None, 10000, 45.0],
                                         [None, None, 10000, None, 45.0],
                                        ],
                          'A513CR369p': [[None, None, None, 12000, 20.0],
                                         [None, None, 12000, None, 20.0],
                                        ],
                          #'A512CR955':  [[None, None, ],
                          #              ]
                         })'''
    @staticmethod
    def fixLabels(lab):

        replace_dict = {
            'asct2high': '_ASCT2high_',
            'atpa5high': '_ATPA5high_',
            'citratesynthasehigh': '_CITRATESYNTHASEhigh_',
            'cpt1ahigh': '_CPT1Ahigh_',
            'g6pdhigh': '_G6PDhigh_',
            'glut1high': '_GLUT1high_',
            'hexokinase1high': '_HEXOKINASE1high_',
            'idh2high': '_IDH2high_',
            'nakatpasehigh': '_NAKATPASEhigh_',
            'pnrf2high': '_PNRF2high_',
            'sdhahigh': '_SDHAhigh_',
            'pdhigh': '_PDhigh_',

            'asct2': '_ASCT2_',
            'atpa5': '_ATPA5_',
            'citratesynthase': '_CITRATESYNTHASE_',
            'cpt1a': '_CPT1A_',
            'g6pd': '_G6PD_',
            'glut1': '_GLUT1_',
            'hexokinase1': '_HEXOKINASE1_',
            'idh2': '_IDH2_',
            'nakatpase': '_NAKATPASE_',
            'pnrf2': '_PNRF2_',
            'sdha': '_SDHA_',
            #'pd': '_PD_',

            'ido1high': '_IDO1_',
            'ido1': '_IDO1_',
            'granzymebhigh': '_Granzymeb_',
            'granzymeb': '_Granzymeb_',
            'icoshigh': '_ICOS_',
            'icos': '_ICOS_',
            'pd1high': '_PD1_',
            'pd1': '_PD1_',
            'pdl1high': '_PDL1_',
            'pdl1': '_PDL1_',

            'hlaahigh': '_HLAA_',
            'hlaa': '_HLAA_',
            'ki67': '_KI67_',
            'vimentin': "_VIMENTIN_",
        }

        lab = lab.replace('_cells', 'cells')
        for key, item in replace_dict.items():
            lab = lab.replace(key, item)
        while '__' in lab:
            lab = lab.replace('__', '_')
        if lab[-1] == '_':
            lab = lab[:-1]

        #for key, item in replace_dict.items():
        #    if 'high' in item and item[1:-1] in lab:
        #        lab = lab.replace(item[:-5] + '_', '_')

        return lab

    @staticmethod
    def replacenth(string_, sub, wanted):
        while len(list(re.finditer(sub, string_))) > 1:
            where = [m.start() for m in re.finditer(sub, string_)][1]
            before = string_[:where]
            after = string_[where:]
            after = after.replace(sub, wanted, 1)
            string_ = before + after
        return string_

    def combineMetaAndImmuneAndRefactorPathways(self):
        self.adata.obs['cell_types_immune_metapath'] = self.adata.obs['Immune_meta'].astype(str) + '_' +  self.adata.obs['Immune_func'].astype(str) + \
            '_' + self.adata.obs['tumour_meta'].astype(str) + '_' + self.adata.obs['tumour_func'].astype(str)

        pathways = {
            'AminoAcidUptake' : ['ASCT2high', 'ASCT2'],
            'ATPSynthesis'    : ['ATPA5high', 'ATPA5', 'NAKATPASE', 'SDHAhigh', 'SDHA'],
            'TCACyle'         : ['CITRATESYNTHASEhigh', 'CITRATESYNTHASE', 'IDH2high', 'IDH2'],
            'FatOxidation'    : ['CPT1Ahigh', 'CPT1A'],
            'Glycolysis'      : ['HEXOKINASE1high', 'HEXOKINASE1', 'G6PDhigh', 'G6PD', 'GLUT1high', 'GLUT1'],
            'Regulatory'      : ['PNRF2high', 'PNRF2'],
        }

        for path, markers in pathways.items():
            for marker in markers:
                self.adata.obs['cell_types_immune_metapath'] = self.adata.obs['cell_types_immune_metapath'].apply(lambda x: x.replace(marker, path))


        cts = ['granulocytecells', 'tumorcells', 'bcells', 'plasmacells', 'cd4tregcells', 'endothilialcells', 'artifactcells', 'myeloidnoscells', 'myofibroblastcells', 'fibroblastcells', \
               'immunenoscells', 'cd4tcells', 'cd8tcells', 'macrophagecells', 'othercells', 'AminoAcidUptake', 'ATPSynthesis', 'TCACyle', 'FatOxidation', 'Glycolysis', 'Regulatory']

        for ct in cts:
            self.adata.obs['cell_types_immune_metapath'] = self.adata.obs['cell_types_immune_metapath'].apply(
                lambda x: self.replacenth(x, ct, '')
            )

        self.adata.obs['cell_types_immune_metapath'] = self.adata.obs['cell_types_immune_metapath'].apply(self.fixLabels)

    def refactorMetaAndImmune(self):
        self.adata.obs['Immune_meta'] =  self.adata.obs['Immune_meta'].apply(self.fixLabels)
        self.adata.obs['Immune_func'] =  self.adata.obs['Immune_func'].apply(self.fixLabels)
        self.adata.obs['tumour_meta'] =  self.adata.obs['tumour_meta'].apply(self.fixLabels)
        self.adata.obs['tumour_func'] =  self.adata.obs['tumour_func'].apply(self.fixLabels)

        #self.combineMetaAndImmuneAndRefactorPathways()


    def createImageAnnData(self, image, image_column='Image'):
        print(f"\tSelected {image} from {image_column}")
        #print(f"\tTissues {set(self.adata.obs[image_column])}")

        self.imageAData = self.adata[self.adata.obs[image_column].isin([image])].copy()
        #self.imageAData.obs = self.imageAData.obs.reset_index(drop=True)
        self.imageName = image

    def sliceImages(self, slicing_dict):
        """
            :param slicing_dict: dict
            :return: None, will update self.adata.obs.Tissue_Subsample_ID with unique tissue IDs
        """
        self.adata.obs['Tissue_Subsample_ID'] = self.adata.obs[self.patientImageId].copy().astype(str)

        for image_sep in slicing_dict.items():
            img_sep_array = image_sep[1]

            k = 0
            img_sep_arr = np.array(self.adata.obs['Tissue_Subsample_ID'].copy())
            for sliceSep in img_sep_array:
                # Rotating (convert to its own function)
                x, y, im = self.adata.obs.x.copy(), self.adata.obs.y.copy(), self.adata.obs[self.patientImageId].copy()

                meanx, meany = x.mean(), y.mean()
                #x -= meanx
                #y -= meany
                theta = np.radians(sliceSep[4])
                c, s = np.cos(theta), np.sin(theta)
                rot = np.array(((c, -s), (s, c)))
                x_, y_ = x[im == image_sep[0]], y[im == image_sep[0]]
                xy = np.dot(rot, np.vstack([x_, y_]))
                x_, y_ = xy[0,:], xy[1,:]
                x[im == image_sep[0]] = x_
                y[im == image_sep[0]] = y_
                self.adata.obs.x = x.copy()
                self.adata.obs.y = y.copy()

                # Change Nones to actual bounds of images
                sliceSep = self.fixSlices(sliceSep)

                print(f"ReadClinicalData.sliceImages\n\tSlicing {sliceSep}\n\tTo Index {image_sep[0] + '_' + str(k)}")

                num = len(img_sep_arr[np.array(self.adata.obs[self.patientImageId].isin([image_sep[0]])) &
                           (np.array(self.adata.obs['x']) >= sliceSep[0]) &
                           (np.array(self.adata.obs['x'])  < sliceSep[1]) &
                           (np.array(self.adata.obs['y']) >= sliceSep[2]) &
                           (np.array(self.adata.obs['y'])  < sliceSep[3])
                    ])

                img_sep_arr[np.array(self.adata.obs[self.patientImageId].isin([image_sep[0]])) &
                           (np.array(self.adata.obs['x']) >= sliceSep[0]) &
                           (np.array(self.adata.obs['x'])  < sliceSep[1]) &
                           (np.array(self.adata.obs['y']) >= sliceSep[2]) &
                           (np.array(self.adata.obs['y'])  < sliceSep[3])
                    ] = np.array([str(image_sep[0]) + '_' + str(k)]*num, dtype=str)

                k += 1

            self.adata.obs['Tissue_Subsample_ID'] = img_sep_arr.copy()
            print(f"\tFound {len( self.adata[self.adata.obs['Tissue_Subsample_ID'].isin([image_sep[0]+'_'+str(k)])] )}")
        return

    def printKeys(self):
        print(r'ReadClinicalData.printKeys\n\tPrinting keys')
        for key in self.adata.obs_keys():
            if len(list(set(self.adata.obs[key]))) < 50:
                print(f"\tKey {key}\t{set(self.adata.obs[key])}")

    def fixSlices(self, sliceS):
        if not sliceS[0]:
            sliceS[0] = -100000
        if not sliceS[1]:
            sliceS[1] = 1000000
        if not sliceS[2]:
            sliceS[2] = -100000
        if not sliceS[3]:
            sliceS[3] = 1000000
        return sliceS

    #############################################################################
    # Plot the
    #############################################################################
    def plotImage(self, plotChannel="tumournontumour", save=True):
        sc.set_figure_params(format='png', dpi=80, dpi_save=300, figsize=(30, 30))

        fig, ax = plt.subplots(figsize=(10, 6))
        sc.pl.spatial(self.imageAData, color=[plotChannel],
                      spot_size=12, color_map="coolwarm",
                      layer='log1p_normalised', vmin='p0', vmax='p100',
                      size=1.5, alpha_img=0, return_fig=True, show=False, ax=ax)
        ax.tick_params(axis='x', color='m', length=4, width=4,
                       labelcolor='g', grid_color='b')
        #gc.collect()
        if save:
            plt.savefig(self.imageName + '.png')
            plt.close()
        else:
            plt.show()

    def plotAllTissues(self):
        sc.set_figure_params(format='png', dpi=200, dpi_save=300, figsize=(30, 30))

        #plotaarray = ['A514CR418p_0', 'A514CR418p_1']

        for image in list(set(self.adata.obs.Tissue_Subsample_ID)):

            print("plotAllTissues: plotting ", image)
            fig, ax = plt.subplots(figsize=(10, 6))
            sc.pl.spatial(self.adata[self.adata.obs.Tissue_Subsample_ID.isin([image])], color=["tumournontumour"],
                          spot_size=12, color_map="coolwarm",
                          layer='log1p_normalised', vmin='p0', vmax='p100',
                          size=1.5, alpha_img=0, return_fig=True, show=False, ax=ax)
            ax.tick_params(axis='x', color='m', length=4, width=4,
                           labelcolor='g', grid_color='b')
            # gc.collect()
            plt.savefig(r'allimagefigures\\' + image + '.png')
            plt.close()

    def plotAllImages(self):
        sc.set_figure_params(format='png', dpi=200, dpi_save=300, figsize=(30, 30))

        #plotaarray = ['A514CR418p_0', 'A514CR418p_1']

        for image in list(set(self.adata.obs.Image)):

            print("plotAllImages: plotting ", image)
            fig, ax = plt.subplots(figsize=(10, 6))
            sc.pl.spatial(self.adata[self.adata.obs.Image.isin([image])], color=["tumournontumour"],
                          spot_size=12, color_map="coolwarm",
                          layer='log1p_normalised', vmin='p0', vmax='p100',
                          size=1.5, alpha_img=0, return_fig=True, show=False, ax=ax)
            ax.tick_params(axis='x', color='m', length=4, width=4,
                           labelcolor='g', grid_color='b')
            # gc.collect()
            plt.savefig(r'allimagefigures\\WholeImage' + image + '.png')
            plt.close()

    def returnAllTissueIDs(self):
        return list(sorted(set(self.adata.obs.Tissue_Subsample_ID.astype(str))))

    #TODO: Fix logical flow
    def returnPatientsAndOutcomes(self, outcomeLabel, timeLabel=None, eventLabel=None, groupby=None, labels=None):
        if not outcomeLabel:
            return {i: [np.nan, np.nan, np.nan, np.nan] for i, a in self.adata.obs.groupby(by=self.patientImageId)}
        outcomes = {i: a[outcomeLabel].iloc[0] for i, a in self.adata.obs.groupby(by=self.patientImageId)}
        times, events = None, None
        if timeLabel:
            assert eventLabel, 'Please supply a time label and an event label'
            times   = {i: a[timeLabel].iloc[0] for i, a in self.adata.obs.groupby(by=self.patientImageId)}
            events  = {i: a[eventLabel].iloc[0] for i, a in self.adata.obs.groupby(by=self.patientImageId)}
        if groupby:
            assert labels
            outcomes_ = {}
            for i, ii in outcomes.items():
                if ii in groupby[0]:
                    if timeLabel:
                        outcomes_[i] = [ii, labels[0], times[i], events[i]]
                    else:
                        outcomes_[i] = [ii, labels[0], -10000, -10000]
                elif ii in groupby[1]:
                    if timeLabel:
                        outcomes_[i] = [ii, labels[1], times[i], events[i]]
                    else:
                        outcomes_[i] = [ii, labels[1], -10000, -10000]
                else:
                    if timeLabel:
                        outcomes_[i] = [ii, np.nan, times[i], events[i]]
                    else:
                        outcomes_[i] = [ii, np.nan, -10000, -10000]

            outcomes = outcomes_

        return outcomes