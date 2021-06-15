from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture

import matplotlib.pylab as plt

import torch


def cluster_kmeans(weights, mapsize, **kwargs):
        kmeans = KMeans(**kwargs)
        kmeans.fit(weights)
        matrix_clusters = torch.tensor(kmeans.labels_).reshape(mapsize)  

        return matrix_clusters

def cluster_dbscan(weights, mapsize, **kwargs):
    
    clustering = DBSCAN(**kwargs).fit(weights)

    matrix_clusters = torch.tensor(clustering.labels_).reshape(mapsize)
  
    return matrix_clusters 


def cluster_AffinityPropagation(weights, mapsize, **kwargs):

    model = AffinityPropagation(**kwargs)
    model.fit(weights) 

    matrix_clusters = torch.tensor(model.labels_).reshape(mapsize)

    return matrix_clusters

def cluster_AgglomerativeClustering(weights, mapsize, **kwargs):

    model = AgglomerativeClustering(**kwargs).fit(weights)
    matrix_clusters = torch.tensor(model.labels_).reshape(mapsize)

    return matrix_clusters

def cluster_Birch(weights, mapsize, **kwargs):

    model = Birch(**kwargs).fit(weights)
    matrix_clusters = torch.tensor(model.labels_).reshape(mapsize)

    return matrix_clusters

def cluster_MeanShift(weights, mapsize, **kwargs):

    model = MeanShift(**kwargs).fit(weights)
    matrix_clusters = torch.tensor(model.labels_).reshape(mapsize)

    return matrix_clusters

def cluster_OPTICS(weights, mapsize, **kwargs):

    model = OPTICS(**kwargs).fit(weights)
    matrix_clusters = torch.tensor(model.labels_).reshape(mapsize)

    return matrix_clusters 

def cluster_GaussianMixture(weights, mapsize, **kwargs):

    model = GaussianMixture(**kwargs).fit_predict(weights)
    matrix_clusters = torch.tensor(model).reshape(mapsize)

    return matrix_clusters

def plot_matrix(matrix_clusters, mapsize, figsize):

    fig, ax = plt.subplots(figsize=figsize)
    min_val, max_val = 0, mapsize[0]
    ax.matshow(matrix_clusters, cmap=plt.cm.Oranges)

    for i in range(max_val):
        for j in range(max_val):
            c = matrix_clusters[j,i]
            ax.text(i, j, str(c.item()), va='center', ha='center')

    plt.show()


