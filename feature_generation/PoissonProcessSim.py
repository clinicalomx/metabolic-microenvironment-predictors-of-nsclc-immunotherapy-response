import scipy
import numpy as np
import anndata as ad

class PoissonProcessSim:
    def __init__(self, density, pixel_count, print=False):
        self.density     = density     # Point density / pixel^2
        if type(pixel_count) is list:
            self.pixel_count = pixel_count # Pixel count
        else:
            self.pixel_count = [pixel_count, pixel_count]

        self.numberOfPoints = None
        self.pointsAnnData = None
        self.x = np.empty([])
        self.y = np.empty([])
        self.print = print

        self.simulatePointImage()

    def simulatePointImage(self):
        if self.print:
            print(f"PoissonProcessSim.simulatePointImage:\n\tSimulating {self.density*self.pixel_count[0]*self.pixel_count[1]} points")
        self.numberOfPoints = scipy.stats.poisson(self.density*self.pixel_count[0]*self.pixel_count[1]).rvs()
        if self.print:
            print(f"PoissonProcessSim.simulatePointImage:\n\tSimulating {self.numberOfPoints} points")
        self.x    = np.squeeze(scipy.stats.uniform.rvs(0, self.pixel_count[0],((self.numberOfPoints,1))))
        self.y    = np.squeeze(scipy.stats.uniform.rvs(0, self.pixel_count[1],((self.numberOfPoints,1))))
        #self.pointsStack    = np.hstack((Xpositions, Ypositions))
        #print(f"PoissonProcessSim.simulatePointImage:\n\tCreated points\n\t{self.pointsStack}")

    def createAnnDataOfImage(self):
        self.pointsAnnData = ad.AnnData(obs=np.hstack((self.x, self.y)))
        if self.print:
            print(f"PoissonProcessSim.createAnnDataOfImage:\n\tCreated AnnData object\n\t{self.pointsAnnData}")