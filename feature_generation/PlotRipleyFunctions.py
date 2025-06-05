import matplotlib.pyplot as plt
from PoissonProcessSim import *
from ComputeRipleyMetrics import *

# TODO: figure out pixel scale so that poisson sim is at same scale as images.
class PlotRipleyFunctions:
    def __init__(self, radii_of_computation,
                 k_ripley_values_1,
                 k_ripley_values_2=None,
                 label_1=None,
                 label_2=None,
                 file_name=None,
                 simulateAndPlotPoisson=False,
                 density=0.0,
                 pixel_count=0,
                 npseudo=100,
                 ):
        self.radii_of_computation = radii_of_computation
        self.k_ripley_values_1 = k_ripley_values_1
        self.k_ripley_values_2 = k_ripley_values_2
        self.label_1 = label_1
        self.label_2 = label_2
        self.file_name = file_name
        self.simulateAndPlotPoisson = simulateAndPlotPoisson
        self.density = density
        self.pixel_count = pixel_count
        self.npseudo = npseudo,
        self.poissonVals = []
        if self.simulateAndPlotPoisson:
            assert self.density > 0 and self.pixel_count > 0
            self.simulatePoisson(npseudo=100)
        # self.plotKRipleys()

    def simulatePoisson(self):
        krip = ComputeRipleyMetrics()
        self.poissonVals = []
        for ps in range(self.npseudo):
            pps = PoissonProcessSim(density=self.density / 2.0, pixel_count=self.pixel_count)
            pps.simulatePointImage()
            pps2 = PoissonProcessSim(density=self.density / 2.0, pixel_count=self.pixel_count)
            pps2.simulatePointImage()

            krip.computeKRipleyCrossFunction(self.radii_of_computation, pps.x, pps.y, pps2.x, pps2.y)
            self.poissonVals.append(krip.kripleyCrossValues)

    def plotKRipleys(self):
        if type(self.label_1) == list:
            self.plotAll()
        else:
            if self.label_2:
                print("Plotting both")
                self.plotTwoKRipleyFunctions()
            else:
                self.plotOneKRipleyFunction()

    def plotTwoKRipleyFunctions(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.radii_of_computation, self.k_ripley_values_1, label=self.label_1)
        ax.plot(self.radii_of_computation, self.k_ripley_values_2, label=self.label_2)
        ax.set_title('K-Ripley Values')
        ax.set_ylabel('K-Ripley')
        ax.set_xlabel('Radius [um]')
        if self.label_1 or self.label_2:
            plt.legend()
        print("saving to k_ripley_values_2.png")
        if self.file_name:
            plt.savefig(self.file_name + '.png')
        plt.close()

    def plotOneKRipleyFunction(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        if self.label_1:
            ax.plot(self.radii_of_computation, self.k_ripley_values_1, label=self.label_1)
        else:
            ax.plot(self.radii_of_computation, self.k_ripley_values_1)
        ax.set_title('K-Ripley Values')
        ax.set_ylabel('K-Ripley')
        ax.set_xlabel('Radius [um]')
        if self.label_1:
            plt.legend()
        if self.file_name:
            plt.savefig(self.file_name + '.png')
        plt.close()

    def plotAll(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in range(len(self.k_ripley_values_1)):
            plt.plot(self.radii_of_computation, self.k_ripley_values_1[k], label=self.label_1[k])
        if self.simulateAndPlotPoisson:
            print("np.array(self.poissonVals).shape: ", np.array(self.poissonVals).shape)
            plt.fill_between(self.radii_of_computation,
                             np.mean(np.array(self.poissonVals), axis=0) - np.std(np.array(self.poissonVals), axis=0),
                             np.mean(np.array(self.poissonVals), axis=0) + np.std(np.array(self.poissonVals), axis=0),
                             fc='k', alpha=0.2)
            plt.plot(self.radii_of_computation, np.mean(np.array(self.poissonVals), axis=0), 'k--',
                     label="Poisson CrossK")
        ax.set_title('K-Ripley Cross Values')
        ax.set_ylabel('K-Ripley')
        ax.set_xlabel('Radius [um]')
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        if self.file_name:
            plt.savefig(self.file_name + '.png')
            plt.close()
        else:
            plt.show()