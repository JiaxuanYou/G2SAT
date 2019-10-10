import networkx as nx
import numpy as np
import os
import pdb
import random
import torch

import time
from torch_geometric.data import Data
from torch_scatter import scatter_add


from batch import DataLoader, Dataset_mine
import csv
from utils import *
import copy

def load_graphs_lcg(data_dir, stats_dir):
    # load stats
    with open(stats_dir+'lcg_stats.csv') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        stats = []
        for stat in data:
            stats.append(stat)

    # load graphs
    graphs = []
    nodes_par1s = []
    nodes_par2s = []
    for filename in os.listdir(data_dir):
        if 'lcg_edge' in filename:
            with open(data_dir + filename, 'rb') as fh:
                graph = nx.read_edgelist(fh)
            filename = filename[:-14] # remove postfix
            # find partite split
            for stat in stats:
                if filename in stat[0]:
                    n = graph.number_of_nodes()
                    n_var = int(stat[1])
                    n_clause = int(stat[2])
                    if graph.number_of_nodes() != n_var*2+n_clause:
                        print('Stats not match!')
                        print(stat[0], filename, graph.number_of_nodes(), graph.number_of_edges(), n_var, n_clause)
                    else:
                        # relabel nodes
                        keys = [str(i + 1) for i in range(n)]
                        vals = range(n)
                        mapping = dict(zip(keys, vals))
                        nx.relabel_nodes(graph, mapping, copy=False)
                        # add links between v and -v
                        graph.add_edges_from([(i, i + n_var) for i in range(n_var)])
                        # split nodes partite
                        nodes_par1 = list(range(n_var * 2))
                        nodes_par2 = list(range(n_var * 2, n_var * 2 + n_clause))
                        nodes_par1s.append(nodes_par1)
                        nodes_par2s.append(nodes_par2)
                        graphs.append(graph)
                    break

    return graphs, nodes_par1s, nodes_par2s



class Dataset_sat(torch.utils.data.Dataset):
    def __init__(self, graph_list, nodes_par1_list, nodes_par2_list, epoch_len,
                 yield_prob=1, speedup=False, hop=4, simple_sample=False):
        super(Dataset_sat, self).__init__()
        self.graph_list = graph_list
        self.nodes_par1_list = nodes_par1_list
        self.nodes_par2_list = nodes_par2_list
        self.epoch_len = epoch_len
        self.yield_prob = yield_prob
        self.speedup = speedup
        self.hop = hop
        self.simple_sample = simple_sample

        self.data_generator = self.get_data()
    def __getitem__(self, index):
        return next(self.data_generator)

    def __len__(self):
        # return len(self.data)
        return self.epoch_len

    @property
    def num_features(self):
        return 3

    @property
    def num_classes(self):
        return 2

    def get_template(self):
        graph_templates = []
        nodes_par1s = []
        nodes_par2s = []
        for i in range(len(self.graph_list)):
            graph = self.graph_list[i].copy()
            nodes_par1 = self.nodes_par1_list[i].copy()
            nodes_par2 = self.nodes_par2_list[i].copy()
            while True:
                degree_info = list(graph.degree(nodes_par2))
                node, node_degree = max(degree_info, key=lambda item: item[1])  # (node, degree)
                if node_degree == 1:
                    print('done',node_degree)
                    graph_templates.append(graph)
                    nodes_par1s.append(nodes_par1)
                    nodes_par2s.append(nodes_par2)
                    break
                node_nbrs = list(graph[node])
                node_nbr = random.choice(node_nbrs)
                graph.remove_edge(node, node_nbr)
                node_new = graph.number_of_nodes()  # new node in nodes_par2
                nodes_par2.append(node_new)
                graph.add_edge(node_nbr, node_new)
        return graph_templates, nodes_par1s, nodes_par2s

    def get_template_fast(self):
        graph_templates = []
        nodes_par1s = []
        nodes_par2s = []
        for i in range(len(self.graph_list)):
            graph = self.graph_list[i].copy()
            n_var = len(self.nodes_par1_list[i])//2
            graph.remove_edges_from([(i, i + n_var) for i in range(n_var)])
            nodes_par1 = self.nodes_par1_list[i].copy()
            nodes_par2 = list(range(len(nodes_par1),len(nodes_par1)+graph.number_of_edges()))
            graph_template = nx.Graph()
            graph_template.add_nodes_from(nodes_par1+nodes_par2)
            node_par2 = nodes_par2[0]
            for node_par1 in nodes_par1:
                # add additional message path
                if node_par1<len(nodes_par1)//2:
                    graph_template.add_edge(node_par1, node_par1+len(nodes_par1)//2)
                deg = graph.degree[node_par1]
                # print(deg)
                for i in range(deg):
                    graph_template.add_edge(node_par1, node_par2)
                    node_par2 += 1
            graph_templates.append(graph_template)
            nodes_par1s.append(nodes_par1)
            nodes_par2s.append(nodes_par2)
        return graph_templates, nodes_par1s, nodes_par2s

    def get_data(self):
        # assume we hold nodes_par1, while split node in nodes_par2
        # output node pair (node, node_new) and corresponding edge_list

        # 1 pick max degree node in nodes_par2
        while True:
            id = np.random.randint(len(self.graph_list))
            graph = self.graph_list[id].copy()
            nodes_par1 = self.nodes_par1_list[id].copy()
            nodes_par2 = self.nodes_par2_list[id].copy()

            while True:
                degree_info = list(graph.degree(nodes_par2))
                random.shuffle(degree_info)
                node, node_degree = max(degree_info, key=lambda item:item[1]) # (node, degree)
                if node_degree==1:
                    break
                node_nbrs = list(graph[node])
                node_nbr = random.choice(node_nbrs)
                graph.remove_edge(node, node_nbr)
                node_new = graph.number_of_nodes() # new node in nodes_par2
                nodes_par2.append(node_new)
                graph.add_edge(node_nbr, node_new)

                if np.random.rand()<self.yield_prob:
                    # generate output data
                    if self.speedup:
                        # sample negative examples
                        node_par1 = list(graph[node_new])[0]
                        if node_par1 < len(nodes_par1) // 2:
                            node_par1_not = node_par1 + len(nodes_par1) // 2
                        else:
                            node_par1_not = node_par1 - len(nodes_par1) // 2
                        node_par1_nbrs = set(graph[node_par1])
                        node_par1_not_nbrs = set(graph[node_par1_not])
                        nodes_candidates = set(nodes_par2) - node_par1_nbrs - node_par1_not_nbrs - {node_new}
                        node_sample = random.sample(nodes_candidates, k=1)[0]

                        nodes_sub1 = set(dict(nx.single_source_shortest_path_length(graph, node, cutoff=self.hop)).keys())
                        nodes_sub2 = set(dict(nx.single_source_shortest_path_length(graph, node_new, cutoff=self.hop)).keys())
                        nodes_sub3 = set(dict(nx.single_source_shortest_path_length(graph, node_sample, cutoff=self.hop)).keys())
                        graph_sub = graph.subgraph(nodes_sub1.union(nodes_sub2,nodes_sub3))
                        keys = list(graph_sub.nodes)
                        vals = range(len(keys))
                        mapping = dict(zip(keys, vals))
                        graph_sub = nx.relabel_nodes(graph_sub, mapping, copy=True)
                        x = torch.zeros((len(keys), 3))
                        nodes_par2_mapped = []
                        for i,key in enumerate(keys):
                            if key<len(nodes_par1) // 2:
                                x[i,0]=1
                            elif key<len(nodes_par1):
                                x[i,1]=1
                            else:
                                x[i,2]=1
                                nodes_par2_mapped.append(i)

                        edge_index = np.array(list(graph_sub.edges))
                        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
                        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

                        # # compute GCN norm
                        # row, col = edge_index
                        # deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
                        # deg_inv_sqrt = deg.pow(-0.5)
                        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

                        node_index_positive = torch.from_numpy(np.array([[mapping[node]], [mapping[node_new]]])).long()
                        node_index_negative = torch.from_numpy(np.array([[mapping[node]], [mapping[node_sample]]])).long()
                    else:
                        x = torch.zeros((graph.number_of_nodes(), 3))  # 3 types of nodes
                        x[:len(nodes_par1)//2,0] = 1
                        x[len(nodes_par1)//2:len(nodes_par1),1] = 1
                        x[len(nodes_par1):,2] = 1

                        edge_index = np.array(list(graph.edges))
                        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
                        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)
                        node_index_positive = torch.from_numpy(np.array([[node], [node_new]])).long()
                        # sample negative examples
                        if self.simple_sample:
                            node_neg_pair = random.sample(nodes_par2,2)
                            node_index_negative = torch.from_numpy(np.array([[node_neg_pair[0]], [node_neg_pair[1]]])).long()
                        else:
                            # sample additional node
                            node_par1 = list(graph[node_new])[0]
                            if node_par1 < len(nodes_par1) // 2:
                                node_par1_not = node_par1 + len(nodes_par1) // 2
                            else:
                                node_par1_not = node_par1 - len(nodes_par1) // 2
                            node_par1_nbrs = set(graph[node_par1])
                            node_par1_not_nbrs = set(graph[node_par1_not])
                            nodes_candidates = set(nodes_par2) - node_par1_nbrs - node_par1_not_nbrs
                            while True:
                                node_sample = random.sample(nodes_candidates, k=1)[0]
                                if node_sample != node_new:
                                    break
                            node_index_negative = torch.from_numpy(np.array([[node_new], [node_sample]])).long()
                    data = Data(x=x, edge_index=edge_index)
                    data.node_index_positive = node_index_positive
                    data.node_index_negative = node_index_negative
                    yield data
                else:
                    continue



class graph_generator():
    def __init__(self, graph, par_split, sample_size = 100, device='cpu', clause_num=None):
        self.graph_raw = graph
        self.graph = self.graph_raw.copy()
        self.par_split = par_split
        self.sample_size = sample_size
        self.clause_num = clause_num
        self.device = device
        # init once
        self.n = self.graph.number_of_nodes()
        self.data = Data()
        self.data.x = torch.zeros((self.n, 3))  # 3 types of nodes
        self.data.x[:self.par_split // 2, 0] = 1
        self.data.x[self.par_split // 2:self.par_split, 1] = 1
        self.data.x[self.par_split:, 2] = 1
        self.data.node_index = torch.zeros((2,self.sample_size),dtype=torch.long)

        self.reset()

        self.data.to(device)


    def reset(self):
        # reset graph to init state
        self.graph = self.graph_raw.copy()
        self.node_par2s = set(range(self.par_split, self.n))
        self.data.edge_index = np.array(list(self.graph.edges))
        self.data.edge_index = np.concatenate((self.data.edge_index, self.data.edge_index[:, ::-1]), axis=0)
        self.data.edge_index = torch.from_numpy(self.data.edge_index).long().permute(1, 0).to(self.device)
        self.resample()

    # picked version
    def resample(self):
        # select new node to merge
        degree_info = list(self.graph.degree(self.node_par2s))
        random.shuffle(degree_info)
        node_new, node_degree = min(degree_info, key=lambda item: item[1])
        # if node_degree>1:
        # print(len(self.node_par2s), self.clause_num)
        if len(self.node_par2s) == self.clause_num:
            return True  # exit_flag = True
        node_par1s = self.graph[node_new]
        # sample additional node
        node_par1_nbrs = set()
        node_par1_not_nbrs = set()
        for node_par1 in list(node_par1s):
            if node_par1<self.par_split // 2:
                node_par1_not = node_par1 + self.par_split // 2
            else:
                node_par1_not = node_par1 - self.par_split // 2
            node_par1_nbrs = node_par1_nbrs.union(set(self.graph[node_par1]))
            node_par1_not_nbrs = node_par1_not_nbrs.union(set(self.graph[node_par1_not]))
        nodes_candidates = self.node_par2s - node_par1_nbrs - node_par1_not_nbrs
        sample_size = min(self.sample_size,len(nodes_candidates))
        nodes_sample = torch.tensor(random.sample(nodes_candidates, k=sample_size),dtype=torch.long)
        # generate queries
        self.data.node_index = torch.zeros((2,sample_size),dtype=torch.long,device=self.device)
        self.data.node_index[0, :] = node_new
        self.data.node_index[1, :] = nodes_sample
        # pdb.set_trace()
        return False

    def merge(self, node_pair):
        # node_pair: node_new, node
        # merge node
        self.data.edge_index[self.data.edge_index==node_pair[0]] = node_pair[1]
        node_pair = node_pair.cpu().numpy()
        self.graph = nx.contracted_nodes(self.graph, node_pair[1], node_pair[0])
        self.node_par2s.remove(node_pair[0])

    def update(self, node_pair):
        # node_pair: node_new, node
        self.merge(node_pair)
        return self.resample()


    def get_graph(self):
        graph = nx.Graph()
        edge_list = self.data.edge_index.cpu().numpy().transpose(1, 0).tolist()
        graph.add_edges_from(edge_list)
        return graph
