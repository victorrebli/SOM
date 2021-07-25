import numpy as np
import scipy as sp
from sklearn.decomposition import PCA

class InvalidNodeIndexError(Exception):
    pass


class InvalidMapsizeError(Exception):
    pass

def generate_hex_lattice(n_rows, n_columns):
    x_coord = []
    y_coord = []
    for i in range(n_rows):
        for j in range(n_columns):
            x_coord.append(i*1.5)
            y_coord.append(np.sqrt(2/3)*(2*j+(1+i)%2))
    coordinates = np.column_stack([x_coord, y_coord])
    return coordinates


class Codebook(object):

    def __init__(self, mapsize, lattice='rect'):
        self.lattice = lattice

        if 2 == len(mapsize):
            _size = [1, np.max(mapsize)] if 1 == np.min(mapsize) else mapsize
            
        else:
            raise InvalidMapsizeError(
                "Mapsize is expected to be a 2 element list or a single int")

        self.mapsize = _size
        self.nnodes = self.mapsize[0]*self.mapsize[1]
        self.matrix = np.asarray(self.mapsize)
        self.initialized = False

        if lattice == "hexa":
            n_rows, n_columns = mapsize
            coordinates = generate_hex_lattice(n_rows, n_columns)
            self.lattice_distances = (sp.spatial.distance_matrix(coordinates, coordinates)
                                      .reshape(n_rows * n_columns, n_rows, n_columns))
                                      
    def grid_dist(self, node_ind):
        """
        Calculates grid distance based on the lattice type.

        :param node_ind: number between 0 and number of nodes-1. Depending on
                         the map size, starting from top left
        :returns: matrix representing the distance matrix
        """
        if self.lattice == 'rect':
            return self._rect_dist(node_ind)

        elif self.lattice == 'hexa':
            return self._hexa_dist(node_ind)

    def _hexa_dist(self, node_ind):
        return self.lattice_distances[node_ind]

    def _rect_dist(self, node_ind):
        """
        Calculates the distance of the specified node to the other nodes in the
        matrix, generating a distance matrix

        Ej. The distance matrix for the node_ind=5, that corresponds to
        the_coord (1,1)
           array([[2, 1, 2, 5],
                  [1, 0, 1, 4],
                  [2, 1, 2, 5],
                  [5, 4, 5, 8]])

        :param node_ind: number between 0 and number of nodes-1. Depending on
                         the map size, starting from top left
        :returns: matrix representing the distance matrix
        """
        rows = self.mapsize[0]
        cols = self.mapsize[1]
        dist = None

        # bmu should be an integer between 0 to no_nodes
        if 0 <= node_ind <= (rows*cols):
            node_col = int(node_ind % cols)
            node_row = int(node_ind / cols)
        else:
            raise InvalidNodeIndexError(
                "Node index '%s' is invalid" % node_ind)

        if rows > 0 and cols > 0:
            r = np.arange(0, rows, 1)[:, np.newaxis]
            c = np.arange(0, cols, 1)
            dist2 = (r-node_row)**2 + (c-node_col)**2

            dist = dist2.ravel()
        else:
            raise InvalidMapsizeError(
                "One or both of the map dimensions are invalid. "
                "Cols '%s', Rows '%s'".format(cols=cols, rows=rows))

        return dist
