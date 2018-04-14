# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:35:26 2017

@author: carmonda
"""
import sys
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

PLOT = True
LABELS = [-1., 1.]
alpha = 0.9
beta = 0.5


class Vertex(object):
    def __init__(self, idx, row, col, name='', y=None, neighs=None, in_msgs=None):
        self._name = name
        self._y = y  # original pixel
        self.row = row
        self.col = col
        self.idx = idx
        self.belief = -sys.maxint
        if neighs is None:
            neighs = set()  # set of neighbour nodes
        if in_msgs is None:
            in_msgs = {}  # dictionary mapping neighbours to their messages
        self._neighs = neighs
        self._in_msgs = in_msgs

    def add_neigh(self, vertex):
        self._neighs.add(vertex)

        # initialize messages to zero for each label
        self._in_msgs[vertex] = {1.: 1, -1.: 1}

    def rem_neigh(self, vertex):
        self._neighs.remove(vertex)

    def log_data_term(self, label):
        return np.exp(alpha * label * self._y)

    def log_smoothness_term(self, my_label, neigh_label):
        return np.exp(beta * my_label * neigh_label)

    def lbp(self, queried_neigh, epsilon):
        assert isinstance(queried_neigh, Vertex)
        soft_max = 0
        msg_delta = 0
        best_msg_lbl = {1.: -sys.maxint, -1.: -sys.maxint}
        for label in LABELS:
            best_msg = -sys.maxint
            for sender_lbl in LABELS:
                cur_match = self.log_data_term(sender_lbl) * self.log_smoothness_term(sender_lbl, label)
                neigh_lst = [neigh for neigh in self._neighs if neigh is not queried_neigh]
                for neigh in neigh_lst:
                    cur_match *= self._in_msgs[neigh][sender_lbl]
                best_msg = max(cur_match, best_msg)
            best_msg_lbl[label] = best_msg
            soft_max += best_msg
        for label in LABELS:
            msg_delta += queried_neigh._in_msgs[self][label] - best_msg_lbl[label] / soft_max
            queried_neigh._in_msgs[self][label] = best_msg_lbl[label] / soft_max
        if msg_delta > epsilon:
            return msg_delta
        else:
            return 0

    def get_belief(self):
        (self.belief, best_value) = (1., -sys.maxint)
        for label in LABELS:
            import operator
            mul_messages = reduce(operator.mul, [self._in_msgs[neigh][label] for neigh in self._neighs], 1)
            cur_value = self.log_data_term(label) * mul_messages
            if cur_value > best_value:
                (self.belief, best_value) = (label, cur_value)
        return self.belief

    def snd_msg(self, neigh):
        """ Combines messages from all other neighbours
            to propagate a message to the neighbouring Vertex 'neigh'.
        """
        return

    def __str__(self):
        ret = "Name: " + self._name
        ret += "\nNeighbours:"
        neigh_list = ""
        for n in self._neighs:
            neigh_list += " " + n._name
        ret += neigh_list
        return ret


class Graph(object):
    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary is given, an empty dict will be used
        """
        if graph_dict is None:
            graph_dict = {}
        self._graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph"""
        return sorted(list(self._graph_dict.keys()), key=lambda vertex: vertex.idx)

    def edges(self):
        """ returns the edges of a graph """
        return self.generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple, or list;
            between two vertices can be multiple edges.
        """
        edge = set(edge)
        (v1, v2) = tuple(edge)
        if v1 in self._graph_dict:
            self._graph_dict[v1].append(v2)
        else:
            self._graph_dict[v1] = [v2]
        # if using Vertex class, update data:
        if type(v1) == Vertex and type(v2) == Vertex:
            v1.add_neigh(v2)
            v2.add_neigh(v1)

    def generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one or two vertices
        """
        e = []
        for v in self._graph_dict:
            for neigh in self._graph_dict[v]:
                if {neigh, v} not in e:
                    e.append({v, neigh})
        return e

    def __str__(self):
        res = "V: "
        for k in self._graph_dict:
            res += str(k) + " "
        res += "\nE: "
        for edge in self.generate_edges():
            res += str(edge) + " "
        return res


def build_grid_graph(n, m, img_mat):
    """ Builds an nxm grid graph, with vertex values corresponding to pixel intensities.
    n: num of rows
    m: num of columns
    img_mat = np.ndarray of shape (n,m) of pixel intensities
    
    returns the Graph object corresponding to the grid
    """
    V = []
    g = Graph()
    # add vertices:
    for i in range(n * m):
        row, col = (i // m, i % m)
        v = Vertex(i, row, col, name="v" + str(i), y=img_mat[row][col])
        g.add_vertex(v)
        if (i % m) != 0:  # has left edge
            g.add_edge((v, V[i - 1]))
        if i >= m:  # has up edge
            g.add_edge((v, V[i - m]))
        V += [v]
    return g


def grid2mat(grid, n, m):
    """ convertes grid graph to a np.ndarray
    n: num of rows
    m: num of columns
    
    returns: np.ndarray of shape (n,m)
    """
    mat = np.zeros((n, m))
    l = grid.vertices()  # list of vertices
    for v in l:
        assert isinstance(v, Vertex)
        i = int(v._name[1:])
        row, col = (i // m, i % m)
        mat[row][col] = v.get_belief()
    return mat


def main():
    # begin:
    if len(sys.argv) < 3:
        print 'Please specify input and output file names.'
        exit(0)

    # load image:
    in_file_name = sys.argv[1]
    image = misc.imread(in_file_name + '.png')
    n, m = image.shape

    # binarize the image.
    image = image.astype(np.float32)
    image[image < 128] = -1.
    image[image > 127] = 1.

    if PLOT:
        plt.imshow(image)
        plt.show()

    # build grid:
    g = build_grid_graph(n, m, image)

    # run the lbp #iterations or until convergence
    iterations = 20
    epsilon = np.power(10., -5)
    for _ in range(iterations):
        delta_counter = 0
        for vertex in g.vertices():
            for neigh in vertex._neighs:
                delta_counter += 1 if (vertex.lbp(neigh, epsilon)) > 0 else 0
        if delta_counter == 0:
            break

    # convert grid to image:
    inferred_img = grid2mat(g, n, m)

    if PLOT:
        plt.imshow(inferred_img)
        plt.show()

    # save result to output file
    out_file_name = sys.argv[2]
    misc.toimage(inferred_img).save(out_file_name + '.png')


if __name__ == "__main__":
    main()
