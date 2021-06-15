import os
import numpy as np
from sklearn.utils import shuffle
from .codebook import Codebook
from .neighborhood import NeighborhoodFactory
from .normalization import NormalizerFactory
from .learningrate import LearningrateFactory
from . import clusters as cluster

import torch
from torch.utils import data
from torch import nn

import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA

_list_cluster = {
        'kmeans': 'cluster_kmeans',
        'dbscan': 'cluster_dbscan',
        'affpro': 'cluster_AffinityPropagation',
        'agglo': 'cluster_AgglomerativeClustering',
        'birch': 'cluster_Birch',
        'meanshift': 'cluster_MeanShift',
        'optics': 'cluster_OPTICS',
        'GM': 'cluster_GaussianMixture',
    }

class SOMException(Exception):
                                                 
      def __init__(self,message):
         super(SOMException, self).__init__()
         self.message = message
                                                 
      def __str__(self):
         return repr(self.message)   

def _load_array(data_arrays, y, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    if y == None:
        dataset = data.TensorDataset(data_arrays)
    else:
        dataset = data.TensorDataset(data_arrays, y)    

    return data.DataLoader(dataset, batch_size, shuffle=is_train)    
   
   
class SOM():

    def __init__(self, 
                mapsize=None,
                mask=None,
                mapshape='planar',
                lattice='rect',
                neighborhood='gaussian',
                lr = 'linear',
                init_radius = 10,
                init_lr = 1,
                epochs=1,
                batch_size = 128,
                shuffle = True):

        self.neighborhood = NeighborhoodFactory.build(neighborhood)
        self.lr = LearningrateFactory.build(lr)
        self.batch_size = batch_size
        self.lattice = lattice
        self.mapsize = mapsize
        self.codebook = None
        self._distance_matrix = None
        self.train_iter = 1
        self.max_iteration = None
        self._lambda = None
        self.epochs = epochs
        self.init_radius = init_radius
        self._radius = init_radius
        self.init_lr = init_lr
        self.log_error = dict()
        self.componente_names = None
        self.clusters_final = dict()
        self.shuffle = shuffle
        
    def _iter_neigh(self):
       self._radius = self.init_radius * np.exp((-self.train_iter/self._lambda)) 

    def _weights(self, nodes, dim):
        weights = torch.zeros(nodes, dim)
        nn.init.normal_(weights, std=0.1) 
        return weights

    def _compute_distance(self,input_1, input_2, p=2.0):
        return torch.sqrt(torch.cdist(input_1.float(), input_2.float(), p=p))

    def _find_shape_grid(self, samples):
        n = int(np.sqrt(5 * np.sqrt(samples)))
        return (n,n)

    def fit(self, data, y=None, features_names=None):
        
        if not torch.is_tensor(data):
            raise SOMException('The "data" parameter must be a Tensor')

        if (y != None) & (torch.is_tensor(y) == False):
            raise SOMException('The "y" parameter must be a Tensor')

        self.max_iteration = self.epochs * (data.shape[0] // self.batch_size)
        self._lambda = self.max_iteration / np.log(self.init_radius)    

        if features_names != None:
            self.componente_names = features_names
        else:
            self.componente_names = ['feature_'+str(_n + 1) for _n in range(data.shape[1])]        

        if self.mapsize is None:
           self.mapsize =  self._find_shape_grid(data.shape[0])

        self.codebook = Codebook(self.mapsize, self.lattice)
        self._distance_matrix = self._calculate_map_dist()     

        nodes = self.codebook.nnodes
        cols = data.shape[1]

        self.weights = self._weights(nodes, cols)
        train_iter = _load_array(data, y, batch_size=self.batch_size, is_train=self.shuffle)
 
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch}')
            batch_log = 1000
            self.dict_names = dict()
            self.dict_positives = dict()
            for idx, _X_train in enumerate(train_iter):
                X_train = _X_train[0]
                y_train = _X_train[1] if len(_X_train) > 1 else None
                if idx * self.batch_size > batch_log:
                   self._train(X_train,y_train, idx=True)
                   batch_log += 1000
                else:
                   self._train(X_train,y_train)

        self.dict_names_final = dict()
        self._bmu_final = torch.argmin(self.predict(data), dim=1)
        self._compomente_planes_final(data, self._bmu_final)

        print('Finalizando')           
                
    def _train(self, X, y, idx=False):
        _bmu = self._find_bmu(X)
        _bmu = torch.argmin(_bmu, dim=1)

        if idx:
            qe_error = self._quantization_error(X,_bmu)
            te_error = self._calculate_topographic_error(X)

            print(f'Quantization error: {qe_error}')
            print(f'Topographic error: {te_error}')
            self.log_error[str(self.train_iter)] = [qe_error.numpy(), te_error]

        else:
            values = self._update_weights(X, _bmu)
            self.weights = self.weights + values
            self._iter_neigh()
            
        self.train_iter += 1    

    def predict(self, X):

        if not torch.is_tensor(X):
            raise SOMException('The "data" parameter must be a Tensor')   

        _bmu = self._find_bmu(X)
        return _bmu


    def _compomente_planes_final(self, data, bmu, name='train'):
        _dict_names_final = dict()
        for idx, ind in enumerate(bmu):
            if ind.item() not in _dict_names_final.keys():
                _dict_names_final[ind.item()] = [data[idx]]
            else:     
                _dict_names_final[ind.item()].append(data[idx])

        '''for idx, ind in enumerate(bmu):
            if ind.item() not in self.dict_positives.keys():
                if y != None:
                    if y[idx].item() == 1:
                        self.dict_positives[ind.item()] = [y[idx].item()]
            else: 
                if y != None:
                    if y[idx].item() == 1:
                        self.dict_positives[ind.item()].append(y[idx].item())'''        

        self.dict_names_final[name] = _dict_names_final        

    def _update_weights(self, X, bmu):

        neighborhood = self.neighborhood.calculate(
                self._distance_matrix, self._radius, self.codebook.nnodes)
        if self.train_iter > 1:
            _lr = self.lr(self.init_lr, self.train_iter, self.max_iteration)
        else:
            _lr = self.init_lr
            
        _value = 0
        for row in range(X.shape[0]):
            ind = bmu[row]
            _value_interm = neighborhood[ind].reshape(-1, 1) * (X[row] - self.weights).numpy()
            _value_interm = _lr * _value_interm
            _value += _value_interm

        return _value / X.shape[0]
           
    def _quantization_error(self, X, bmu):

        error = torch.mean(torch.sum(torch.abs(X - self.weights[bmu, :]), axis=1)) 
        return error    

    def _calculate_topographic_error(self,X):
        _list_nodes = torch.arange(1, self.codebook.nnodes + 1)
        _list_nodes = _list_nodes.reshape(self.codebook.mapsize)        
        bmus = self._compute_distance(X, self.weights)
        bmus = torch.topk(bmus, 2, dim=1)[1]

        TE = 0
        for ind in range(X.shape[0]):
            row_1, col_1 = torch.where(_list_nodes==bmus[ind][0])
            row_2, col_2 = torch.where(_list_nodes==bmus[ind][1])
            row_1, col_1 = row_1.numpy(), col_1.numpy()
            row_2, col_2 = row_2.numpy(), col_2.numpy()

            if (row_2, col_2) not in [(row_1 -1, col_1), (row_1 + 1, col_1), (row_1, col_1 - 1),
                                  (row_1, col_1 + 1)]:
                TE += 1

        TE = TE / X.shape[0]
        return TE


    def _find_bmu(self, X):
        return self._compute_distance(X, self.weights, p=1.0)
        
    def _calculate_map_dist(self):
        """
        Calculates the grid distance, which will be used during the training
        steps. It supports only planar grids for the moment
        """
        nnodes = self.codebook.nnodes

        distance_matrix = np.zeros((nnodes, nnodes))
        for i in range(nnodes):
            distance_matrix[i] = self.codebook.grid_dist(i).reshape(1, nnodes)
        return distance_matrix

    def u_matrix(self, figsize=(10,40)):

        _list_nodes = torch.arange(1, self.codebook.nnodes + 1)
        self.list_nodes = _list_nodes.reshape(self.codebook.mapsize)
        self.view_matrix = self._view_umatrix(self.list_nodes, self.codebook.nnodes)

        self.value_plot_u_matrix = np.expand_dims(self.view_matrix, axis=0).reshape(self.mapsize)

        plt.figure(figsize=figsize)
        plt.imshow(self.value_plot_u_matrix)
        plt.show()

    def heatmap_view(self):

        dict_names_final = self.dict_names_final['train']
        self.res_final = dict()
        for idx, _name in enumerate(self.componente_names):
            _res = []
            for _id in range(self.codebook.nnodes):
                if _id not in dict_names_final.keys():
                    _res.append(np.nan)
                    continue 
                dd = torch.stack(dict_names_final[_id])
                _res.append(torch.mean(dd[:, idx]))

            value_plot = np.expand_dims(_res, axis=0).reshape(self.mapsize)
            self.res_final[_name] = value_plot

        self._plot_heatmap()    

    def _plot_heatmap(self):

        b = torch.tensor([0.25, 0.50, 0.75, 1.0], dtype=torch.float64)
        for i, _name in enumerate(self.res_final.keys()):

            fig = plt.figure()
            ax = plt.axes()
            im = ax.imshow(self.res_final[_name], 
                    interpolation ='nearest')
            plt.title(_name)
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            plt.colorbar(im, cax=cax) 
            plt.show()

    def _get_neight(self, row, col):
        left = (row, max(0, col - 1))
        right = (row, min(self.mapsize[1] - 1, max(0, col + 1)))
        bottom = (min(self.mapsize[0] - 1, max(0, row + 1)), col)
        top = (max(0, row - 1), col)

        return left, right, bottom, top

    def _view_umatrix(self, lista_nodes, nnodes):

        u_matrix = []
        for _n in np.arange(1, nnodes + 1):
            row, col = torch.where(lista_nodes==_n)
            row, col = row.numpy(), col.numpy()
            left, right, bottom, top = self._get_neight(row, col)
            interm = [lista_nodes[left].numpy()[0] - 1,
                    lista_nodes[right].numpy()[0] - 1,
                    lista_nodes[bottom].numpy()[0] - 1,
                    lista_nodes[top].numpy()[0] - 1]
            idx = list(set(interm).difference(set([_n - 1])))
            dist = self._compute_distance(self.weights[_n - 1].reshape(1, -1), self.weights[idx])
            u_matrix.append(torch.mean(dist))

        return u_matrix        
    
    def cluster_model_plot(self, figsize=(10,10), name='kmeans', **kwargs):
        if name not in _list_cluster.keys():
            return 'erro'

        matrix_clusters = eval(f'cluster.{_list_cluster[name]}')(self.weights, self.mapsize, **kwargs)
        cluster.plot_matrix(matrix_clusters, self.mapsize, figsize)

        self.clusters_final[name] = matrix_clusters

    def counts_examples_final(self, figsize=(30,30), name='train'):
        
        dict_names_final = self.dict_names_final.get(name, 0)
        if dict_names_final == 0:
            return 'Erro'

        dici_new = []
        self.dici_final = []
        for idx in range(self.codebook.nnodes):
            tamanho = dict_names_final.get(idx, 0)
            dici_new.append(len(tamanho) if tamanho !=0 else 0)
            
        dici_new = torch.tensor(dici_new).reshape(self.mapsize)
        
        fig, ax = plt.subplots(figsize=figsize)
        min_val, max_val = 0, self.mapsize[0]

        ax.matshow(dici_new, cmap=plt.cm.binary, aspect='auto')
        for i in range(max_val):
            for j in range(max_val):
                c_1 = np.round(dici_new[j,i].item() / torch.sum(dici_new).item(),4)

                #c_2 = dici_count_positives[j,i].item()
                #c_final = str(c_2) + '/' + str(c_1)

                ax.text(i, j, c_1, va='center', ha='center', fontsize=12, color='red')
        plt.show()
    

    def trajectory(self, bmus, cluster = 'GM', figsize=(30,30)):
        
        _list_nodes = torch.arange(0, self.codebook.nnodes)
        self.list_nodes = _list_nodes.reshape(self.codebook.mapsize)

        fig, ax = plt.subplots(figsize=(30,30))
        min_val, max_val = 0, self.mapsize[0]
        ax.matshow(self.clusters_final[cluster])
        for i in range(max_val):
            for j in range(max_val):
                c_1 = self.list_nodes[j,i].item()
                if c_1 in bmus:
                    ax.text(i, j, 'X', va='center', ha='center', color='red', fontsize=10)

        plt.show()  

    def pca_weights(self):
        x = self.weights.numpy()
        x = StandardScaler().fit_transform(x) 
        feat_cols = ['feature'+str(i) for i in range(x.shape[1])]  
        normalised_breast = pd.DataFrame(x,columns=feat_cols)  

        pca_breast = PCA(n_components=2)
        principalComponents_breast = pca_breast.fit_transform(x)

        self.principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])

        print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))
             


               





          
            
            


          






                



