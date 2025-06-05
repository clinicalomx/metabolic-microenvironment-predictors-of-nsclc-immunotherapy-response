import matplotlib.pyplot as plt
import scipy
import numpy as np
from perlin_noise import PerlinNoise

#  Needs work
class PerlinProcessSim:
    def __init__(self, density, pixel_count, number_octaves=20):
        self.density = density
        self.pixel_count = pixel_count
        self.number_octaves = number_octaves
        self.numberOfPoints = None
        self.X = None
        self.Y = None
        self.picture = None
        self.picture_name = 'perlin_image'

        self.simulatePointImage()
        self.plotPointImage()

    def simulatePointImage(self):
        self.numberOfPoints = scipy.stats.poisson(self.density*self.pixel_count[0]*self.pixel_count[1]).rvs()
        noise = PerlinNoise(octaves=self.number_octaves, seed=1)
        pic = np.array([[noise([i/self.pixel_count[0], j/self.pixel_count[1]]) \
                         for j in range(self.pixel_count[0]) ] \
                         for i in range(self.pixel_count[1]) ])
        pic -= pic.min()
        pic /= pic.max()
        pic *= 2
        pic = pic.astype(int)

        xs, ys = [], []
        for i in range(pic.shape[0]):
            for j in range(pic.shape[1]):
                for k in range(pic[i, j]):
                    xs.append(i + np.random.uniform(-1.0, 1.0))
                    ys.append(j + np.random.uniform(-1.0, 1.0))
        self.X = np.array(xs)
        self.Y = np.array(ys)
        idx = np.random.choice(np.arange(len(self.X)), int(self.density*self.pixel_count[0]*self.pixel_count[1]), replace=False)
        self.X = self.X[idx]
        self.Y = self.Y[idx]
        self.picture = pic
        print(f"Simulated {len(self.X)} points")

    def plotPointImage(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.imshow(self.picture)
        plt.scatter(self.Y, self.X, s=1)
        plt.savefig(self.picture_name + '.png')
        plt.close()