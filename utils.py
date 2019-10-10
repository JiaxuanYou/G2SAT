import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time

import matplotlib.pyplot as plt
import io
import pickle
import pdb
import copy
import networkx as nx
import numpy as np
import os
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
import subprocess
import multiprocessing as mp
import random
import json


# save a list of graphs
def save_graph_list(G_list, fname, clean=False, has_par=False, nodes_par1_list=None, nodes_par2_list=None):
    with open(fname, "wb") as f:
        graphs_info = []
        for i,G in enumerate(G_list):
            if clean:
                G = max(nx.connected_component_subgraphs(G), key=len)
            if has_par:
                graphs_info.append([G.nodes(), G.edges(), nodes_par1_list[i], nodes_par2_list[i]])
            else:
                graphs_info.append([G.nodes(), G.edges()])
        pickle.dump(graphs_info, f)

def load_graph_list(fname, has_par=False):
    with open(fname, "rb") as f:
        graphs = []
        if has_par:
            nodes_par1_list = []
            nodes_par2_list = []
        graphs_info = pickle.load(f)
        for graph_info in graphs_info:
            G = nx.Graph()
            G.add_nodes_from(graph_info[0])
            G.add_edges_from(graph_info[1])
            graphs.append(G)
            if has_par:
                nodes_par1_list.append(graph_info[2])
                nodes_par2_list.append(graph_info[3])
    if has_par:
        return graphs, nodes_par1_list, nodes_par2_list
    else:
        return graphs


# draw a list of graphs [G]
def draw_graph_list(G_list, row, col, fname = 'figures/test', layout='spring', is_single=False,k=1,node_size=55,alpha=1,width=1.3):
    # # draw graph view
    # from pylab import rcParams
    # rcParams['figure.figsize'] = 12,3
    if len(G_list)>row*col:
        G_list = G_list[:row*col]
    plt.switch_backend('agg')
    for i,G in enumerate(G_list):
        plt.subplot(row,col,i+1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0)
        plt.axis("off")
        if layout=='spring':
            pos = nx.spring_layout(G,k=k/np.sqrt(G.number_of_nodes()),iterations=20) # default 100
            # pos = nx.spring_layout(G)

        elif layout=='spectral':
            pos = nx.spectral_layout(G)
        # # nx.draw_networkx(G, with_labels=True, node_size=2, width=0.15, font_size = 1.5, node_color=colors,pos=pos)
        # nx.draw_networkx(G, with_labels=False, node_size=1.5, width=0.2, font_size = 1.5, linewidths=0.2, node_color = 'k',pos=pos,alpha=0.2)

        if is_single:
            # node_size default 60, edge_width default 1.5
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0, font_size=0)
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699',alpha=1, linewidths=0.2, font_size = 1.5)
            nx.draw_networkx_edges(G, pos, alpha=0.3,width=0.2)

    plt.tight_layout()
    plt.savefig(fname+'.png', dpi=600)
    plt.close()

if __name__ == '__main__':
    configs_file = '/Users/jiaxuan/Downloads/best_configs.json'
    with open(configs_file, 'r') as f:
        configs = json.load(f)
    pdb.set_trace()
    fname = 'GCN_3_32_preTrue_dropFalse_yield1_08000.dat'
    graphs = load_graph_list('graphs/'+fname)
    graph = graphs[0]
    pdb.set_trace()
    draw_graph_list(graphs, row=4, col=4, fname='fig/'+fname)
