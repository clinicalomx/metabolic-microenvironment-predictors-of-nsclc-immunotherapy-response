import numpy as np
from concave_hull import concave_hull_indexes

# TODO: incorporate hull depth
def computeConcaveHull(x, y, types, concavity=2.0, length_threshold=0.5):
    """
    Compute the concave hull of a set of points for each type.

    Parameters:
    - x: numpy array of shape (n) where n is the number of points, x positions
    - y: numpy array of shape (n) where n is the number of points, y positions
    - t: numpy array of shape (n) where n is the number of points, types of phenotype indices

    Returns:
    - hullPoints: numpy array (n) of booleans indicating the edges
    """

    hullPoints = np.zeros(shape=len(types), dtype=bool)
    for subcluster in list(set(types)):
        scx = x[types == subcluster]
        scy = y[types == subcluster]
        ind = np.where(types == subcluster)[0]
        points = np.vstack([scx, scy]).T

        indexes = concave_hull_indexes(
            points[:, :2], length_threshold=length_threshold, concavity=concavity)

        hullLayer = np.zeros(shape=len(scx), dtype=bool)

        for i in indexes:
            ind_ij = np.where((x == scx[i]) & (y == scy[i]) & (types == subcluster))[0]
            if len(ind_ij) > 1:
                ind_ij = ind_ij[0]
            hullLayer[ind == ind_ij] = True
        hullPoints[types == subcluster] = hullLayer

    return hullPoints

def makefakedata():
    import anndata as ad
    posx, posy = [0, 1, 0, 1], [0, 0, 1, 1]
    xs = np.array([np.random.normal(scale=0.3, loc=posx[i], size=50) for i in range(len(posx))])
    ys = np.array([np.random.normal(scale=0.3, loc=posy[i], size=50) for i in range(len(posy))])
    subclusterindex = np.array([[i]*50 for i in range(len(posx))])
    #df = pd.DataFrame({'x': xs.flatten(), 'y': ys.flatten(), 'types': subclusterindex.flatten()})
    adata = ad.AnnData()
    adata.obs['x'] = xs.flatten()
    adata.obs['y'] = ys.flatten()
    adata.obs['types'] = subclusterindex.flatten()
    adata.obs['image'] = '1'

    print(adata)
    print(adata.obs)

    '''clusterClinicalData = ClusterPointsNolan(adata=adata,
                                             n=20, k=4,
                                             grouping="image",
                                             phenotype='types',
                                             imageName='1')
    clusterDelauneyNeighbors, uniqueDelauneyNeighbors = clinicalData.getSubClusterNeighbours(clusterClinicalData)'''
    return