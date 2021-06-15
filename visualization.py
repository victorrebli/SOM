import matplotlib.pylab as plt
import torch
import numpy as np


def compute_distance(input_1, input_2):
        return torch.sqrt(torch.cdist(input_1.float(), input_2.float(), p= 2.0))[0]

def _get_neight(row, col, size_map):
    left = (row, max(0, col - 1))
    right = (row, min(size_map[1] - 1, max(0, col + 1)))
    bottom = (min(size_map[0] - 1, max(0, row + 1)), col)
    top = (max(0, row - 1), col)

    return left, right, bottom, top

def view_umatrix(weights, lista_nodes, nnodes, size_map):

    u_matrix = []
    for _n in np.arange(1, nnodes + 1):
        row, col = torch.where(lista_nodes==_n)
        row, col = row.numpy(), col.numpy()
        left, right, bottom, top = _get_neight(row, col, size_map)
        interm = [lista_nodes[left].numpy()[0] - 1,
                  lista_nodes[right].numpy()[0] - 1,
                  lista_nodes[bottom].numpy()[0] - 1,
                  lista_nodes[top].numpy()[0] - 1]
        idx = list(set(interm).difference(set([_n - 1])))
        dist = compute_distance(weights[_n - 1].reshape(1, -1), weights[idx])
        u_matrix.append(torch.mean(dist))

    return  u_matrix   

        
        






