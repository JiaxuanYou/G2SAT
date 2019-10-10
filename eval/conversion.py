import networkx as nx
import sys
import random
import io
import copy
import numpy as np
import time
import pickle
import os
import pdb

def main():
    # filename = sys.argv[1]
    # save_dir = sys.argv[2]
    # fname_prefix = 'GCN_3_32_preTrue_dropFalse_yield1_0110015000'
    fname_prefix = 'GCN_3_32_preTrue_dropFalse_yield1'
    # fname_prefix = 'train'
    load_dir = 'graphs/'
    save_dir = 'formulas/'
    try:
        os.mkdir(save_dir)
    except:
        pass
    for filename in os.listdir(load_dir):
        if fname_prefix in filename:
            graphs = load_graphs(load_dir+filename)
            # benchmark_name = filename.split("/")[-1].split('.')[0]
            benchmark_name = filename.split("/")[-1][:-4]
            print(benchmark_name)
            for i, graph in enumerate(graphs):
                bigen_lcg_to_sat(graph, save_dir + benchmark_name + "_{}.cnf".format(i))
    return

def bigen_lcg_to_sat(graph, save_name):
    nodes = list(graph.nodes())
    assert(0 in nodes)
    num_var = min(list(graph.neighbors(0)))
    clauses = []
    for node in nodes:
        if (node >= num_var * 2):
            neighbors = list(graph.neighbors(node))
            clause = ""
            assert(len(neighbors) > 0)
            for lit in neighbors:
                if lit < num_var:
                    clause += "{} ".format(lit + 1)
                else:
                    assert(lit < 2 * num_var)
                    clause += "{} ".format(-(lit - num_var + 1))
            clause += "0\n"
            clauses.append(clause)
    with open(save_name, 'w') as out_file:
        out_file.write("c generated for graphRnn lcg\n")
        out_file.write("p cnf {} {}\n".format(num_var, len(clauses)))
        for clause in clauses:
            out_file.write(clause)
    return

def load_graphs(filename):
    graphs = []
    Gs = nx.read_gpickle(filename)
    for G in Gs:
        graph = nx.Graph()
        graph.add_nodes_from(G[0])
        graph.add_edges_from(G[1])
        graphs.append(graph)
    return graphs     

def graphRnn_lcg_to_sat(graph, num_var, save_name):
    nodes = list(graph.nodes())
    assert(0 in nodes)
    clauses = []
    for node in nodes:
        if (node >= num_var * 2):
            neighbors = list(graph.neighbors(node))
            clause = ""
            assert(len(neighbors) > 0)
            for lit in neighbors:
                if lit < num_var:
                    clause += "{} ".format(lit + 1)
                else:
                    assert(lit < 2 * num_var)
                    clause += "{} ".format(-(lit - num_var + 1))
            clause += "0\n"
            clauses.append(clause)
    with open(save_name, 'w') as out_file:
        out_file.write("c generated for graphRnn lcg\n")
        out_file.write("p cnf {} {}\n".format(num_var, len(clauses)))
        for clause in clauses:
            out_file.write(clause)
    return

def load_graphs(filename):
    graphs = []
    Gs = nx.read_gpickle(filename)
    for G in Gs:
        graph = nx.Graph()
        graph.add_nodes_from(G[0])
        graph.add_edges_from(G[1])
        graphs.append(graph)
    return graphs

def cnf_to_kronfit_format(filename, outname):
    VCG = sat_to_VCG(filename)
    VCG = nx.adjacency_matrix(VCG).toarray()
    with open (outname, 'w') as outfile:
        for i in range(len(VCG)):
            for j in range(len(VCG[i])):
                if VCG[i][j] == 1:
                    outfile.write("%d\t%d\n" %(i + 1, j + 1))
    return


def cut_inner_edge(mat, partite):
    for i in range(partite):
        for j in range (partite):
            mat[i][j] = 0
    for i in range(len(mat) - partite):
        for j in range(len(mat) - partite):
            mat[i + partite][j + partite] = 0
    return mat

# convert a file of adjacency matrix to a matrix
def file_to_mat(filename):
    print ("reading file...")
    lst = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            if ',' in line:
                l = line.split(", ")
                while "\n" in l:
                    l.remove("\n")
            else:
                l = line.split()
            for i in range(len(l)):
                l[i] = int(float(l[i]))
            lst.append(l)
    print ("Successful!")
    return lst


# Takes in an adjacency matrix, convert it to a dimacs file
def lcg_graph_to_sat(mat, num_var, filename):
    #mat = cut_inner_edge(mat, num_var * 2)
    out = open(filename, 'w')
    out.write("c generated from SATGAN\n")
    num_clause = len(mat) - 2 * num_var
    out.write("p cnf %d %d\n" %(num_var, num_clause))
    for i in range(num_clause):
        clause = ""
        for j in range(2 * num_var):
            if mat[i + 2 * num_var][j] == 1:
                if j < num_var:
                    clause = clause + ("%d " %(j + 1))
                else:
                    clause = clause + ("-%d " %(j + 1 - num_var))
        clause = clause + "0\n"
        out.write(clause)
    return

# Takes in an adjacency matrix of a VCG, convert it to a dimacs file
def VCG_to_sat(mat, num_var, filename):
    mat = cut_inner_edge(mat, num_var)
    out = open(filename, 'w')
    out.write("c generated from SATGAN\n")
    num_clause = len(mat) - num_var
    out.write("p cnf %d %d\n" %(num_var, num_clause))
    pos_lits = set()
    neg_lits = set()
    for i in range(num_clause):
        clause = ""
        for j in range(num_var):
            if mat[i + num_var][j] == 1:
                lit = j + 1
                if lit not in pos_lits:
                    clause = clause + ("%d " %(lit))
                    pos_lits.add(lit)
                elif -lit not in neg_lits:
                    clause = clause + ("%d " %(-lit))
                    neg_lits.add(-lit)
                elif random.choice([0, 1]) == 0:
                    clause = clause + ("%d " %(lit))
                else:
                    clause = clause + ("%d " %(-lit))
        clause = clause + "0\n"
        out.write(clause)
    return


# Takes in an dimacs file, convert it to a networkx graph representation of the variable-clause graph
def sat_to_VCG(source):
    cnf = open(source)
    content = cnf.readlines()
    while content[0].split()[0] == 'c':
        content = content[1:]
    while len(content[-1].split()) <= 1:
        content = content[:-1]

    # Paramters
    parameters = content[0].split()
    formula = content[1:] # The clause part of the dimacs file
    formula = to_int_matrix(formula)
    num_vars = int(parameters[2])
    num_clause = int(parameters[3])

    VCG = nx.Graph()
    VCG.add_nodes_from(range(num_vars + num_clause + 1)[1:])
    preprocess_VCG(formula, VCG, num_vars) # Build a VCG
    return VCG




# Takes in an dimacs file, convert it to a nx graph representation of the literal-clause graph
def sat_to_LCG(source):
    cnf = open(source)
    content = cnf.readlines()
    while content[0].split()[0] == 'c':
        content = content[1:]
    while len(content[-1].split()) <= 1:
        content = content[:-1]

    # Paramters
    parameters = content[0].split()
    formula = content[1:] # The clause part of the dimacs file
    formula = to_int_matrix(formula)
    num_vars = int(parameters[2])
    num_clause = int(parameters[3])

    VCG = nx.Graph()
    VCG.add_nodes_from(range(num_vars * 2 + num_clause + 1)[1:])
    preprocess_LCG(formula, VCG, num_vars) # Build a VCG
    #    mat = nx.adjacency_matrix(VCG)
    return VCG

# Takes in an dimacs file, convert it to a nx graph representation of the literal incidence graph
def sat_to_LIG(source):
    cnf = open(source)
    content = cnf.readlines()
    while content[0].split()[0] == 'c':
        content = content[1:]
    while len(content[-1].split()) <= 1:
        content = content[:-1]

    # Paramters
    parameters = content[0].split()
    formula = content[1:]
    formula = to_int_matrix(formula)
    num_vars = int(parameters[2])
    num_clause = int(parameters[3])
    #print (num_vars)

    LIG = nx.Graph()
    LIG.add_nodes_from(range(num_vars * 2 + 1)[1:])
    preprocess_LIG(formula, LIG, num_vars) # Build a LIG
    return LIG


# Takes in an dimacs file, convert it to a nx graph representation of the variable incidence graph
def sat_to_VIG(source):
    cnf = open(source)
    content = cnf.readlines()
    while content[0].split()[0] == 'c':
        content = content[1:]
    while len(content[-1].split()) <= 1:
        content = content[:-1]

    # Paramters
    parameters = content[0].split()
    formula = content[1:]
    formula = to_int_matrix(formula)
    num_vars = int(parameters[2])
    num_clause = int(parameters[3])
    #print (num_vars)

    VIG = nx.Graph()
    VIG.add_nodes_from(range(num_vars + 1)[1:])
    preprocess_VIG(formula, VIG, num_vars) # Build a LIG
    return VIG


def get_cl_string(clause):
    s = ""
    clause = sorted(clause)
    for ele in clause:
        s += str(ele) + "-"
    return s[:-1]

def remove_duplicate(content):
    new_content = [content[0].split()]
    cs = set()
    num_clause = 0
    for line in content[1:]:
        line = map(int, line.split()[:-1])
        #        c = get_cl_string(line)
        #        if c not in cs:
        #            num_clause += 1
        new_content.append(line)
#            cs.add(c)
#    new_content[0][3] = num_clause
    return new_content


def preprocess_VCG(formula, VCG, num_vars):
    """
    Builds VCG
    """
    for cn in range(len(formula)):
        for var in formula[cn]:
            if var > 0:
                VCG.add_edge(var, cn +  num_vars + 1)
            elif var < 0:
                VCG.add_edge(abs(var), cn + num_vars + 1)



def preprocess_LCG(formula, LCG, num_vars):
    """
    Builds LCG
    """
    for cn in range(len(formula)):
        for var in formula[cn]:
            if var > 0:
                LCG.add_edge(var, cn + 2 * num_vars + 1)
            elif var < 0:
                LCG.add_edge(abs(var) + num_vars, cn + 2 * num_vars + 1)

def preprocess_LIG(formula, LIG, num_vars):
    """
    Builds LIG.
    """
    for cn in range(len(formula)):
        for i in range(len(formula[cn])-1):
            for j in range(len(formula[cn]))[i+1:]:
                lit1 = formula[cn][i]
                lit2 = formula[cn][j]
                if lit1 > 0:
                    node1 = lit1
                elif lit1 < 0:
                    node1 = abs(lit1) + num_vars
                if lit2 > 0:
                    node2 = lit2
                elif lit2 < 0:
                    node2 = abs(lit2) + num_vars
                LIG.add_edge(node1, node2)

def preprocess_VIG(formula, VIG, num_vars):
    """
    Builds VIG.
    """
    for cn in range(len(formula)):
        for i in range(len(formula[cn])-1):
            for j in range(len(formula[cn]))[i+1:]:
                lit1 = formula[cn][i]
                lit2 = formula[cn][j]
                if lit1 > 0:
                    node1 = lit1
                elif lit1 < 0:
                    node1 = abs(lit1)
                if lit2 > 0:
                    node2 = lit2
                elif lit2 < 0:
                    node2 = abs(lit2)
                VIG.add_edge(node1, node2)

def to_int_matrix(formula):
    new_formula = []
    for i in range(len(formula)):
        line = []
        for ele in formula[i].split()[: -1]:
            line.append(int(ele))
        new_formula.append(line)
    return new_formula

def get_binary_subgraph(formula):
    bin_formula = []
    for clause in formula:
        if len(clause) == 2:
            bin_formula.append(clause)
    return bin_formula


def sat_to_bin_LCG(source):
    # Takes in an dimacs file, convert it to a nx graph representation of the literal-clause graph of the binary subgraph
    cnf = open(source)
    content = cnf.readlines()
    while content[0].split()[0] == 'c':
        content = content[1:]
    while len(content[-1].split()) <= 1:
        content = content[:-1]

    # Paramters
    parameters = content[0].split()
    formula = content[1:] # The clause part of the dimacs file
    formula = get_binary_subgraph(to_int_matrix(formula))
    num_vars = int(parameters[2])
    num_clause = len(formula)

    LCG = nx.Graph()
    LCG.add_nodes_from(range(num_vars * 2 + num_clause + 1)[1:])
    preprocess_LCG(formula, LCG, num_vars) # Build a VCG
    return LCG, num_vars


def sat_to_bin_sat(source):
    VCG, num_vars = sat_to_bin_VCG(source)
    mat = nx.adjacency_matrix(VCG).toarray()
    if source[-3:] == "cnf":
        filename = source[:-3] + "bin.cnf"
    else:
        filename = source + ".bin"
    graph_to_sat(mat, num_vars, filename)



def subset(c_, c):
    for l in c_:
        if l not in c:
            return False
    return True

def remove_subset(cliques):
    new_cliques = []
    cur = len(cliques) - 1
    while cur >= 0:
        s = False
        for c in new_cliques:
            if subset(cliques[cur], c):
                s = True
        if not s:
            new_cliques.append(cliques[cur])
        cur -= 1
    return new_cliques

def clique_to_clause(c, num_vars):
    clause = []
    for l in c:
        l_ = int(l)
        if l_ >= num_vars:
            clause.append(- (l_ + 1 - num_vars))
        else:
            clause.append(l_ + 1)
    return clause

def create_formula(new_cliques, num_vars):
    formula = []
    for c in new_cliques:
        formula.append(clique_to_clause(c, num_vars))
    return formula


def break_down_clause(clause):
    assert(len(clause) > 2)
    excludes = np.random.choice(range(len(clause)), replace=False, size=3)
    c0 = []
    c1 = []
    c2 = []

    for i in range(len(clause)):
        if i != excludes[0]:
            c0.append(clause[i])
        if i != excludes[1]:
            c1.append(clause[i])
        if i != excludes[2]:
            c2.append(clause[i])
    return set([tuple(c0), tuple(c1), tuple(c2)])

def expand_to_n_clauses(formula, n):
    new_formula = set(map(tuple, copy.copy(formula)))
    while len(new_formula) < n:
        c = random.sample(new_formula, 1)[0]
        if len(c) == 2:
            continue
        new_formula.remove(c)
        new_formula = new_formula.union(break_down_clause(c))
    return list(map(list, new_formula))

def lig_to_sat(LIG, target_clauses, filename, timeout):
    """
    Convert a LIG to a SAT formula
    """
    num_vars = int(LIG.number_of_nodes() / 2)
    print ("Extracting cliques...")
    cliques = list(nx.enumerate_all_cliques(LIG))
    #cliques.reverse()
    #cliques = remove_subset(cliques)
    if timeout != 0:
        start_time = time.time()
    clique_cover, hasTimeout = greedy_hill_climbing(LIG, cliques, LIG.number_of_edges(), num_vars, start_time, timeout)
    expanded_clique_cover = expand_to_n_clauses(clique_cover, target_clauses)
    formula = create_formula(expanded_clique_cover, num_vars)
    with open(filename, 'w') as out_file:
        out_file.write("c generated from SAT-GAN\n")
        if hasTimeout:
            out_file.write("c timeout\n")
        out_file.write("p cnf {} {}\n".format(num_vars, len(formula)))
        for line in formula:
            out_file.write(" ".join(map(str, line)))
            out_file.write(" 0\n")

def cut_formula(filename, outname, n):
    cnf = open(filename)
    content = cnf.readlines()
    while content[0].split()[0] == 'c':
        content = content[1:]
    while len(content[-1].split()) <= 1:
        content = content[:-1]

    # Paramters
    parameters = content[0].split()
    print (parameters)
    formula = content[1:] # The clause part of the dimacs file
    num_vars = int(parameters[2])
    inds = np.random.choice(range(len(formula)), replace=False, size=n)
    with open(outname, 'w') as out:
        out.write("c generated from SATGAN\n")
        out.write("p cnf %d %d\n" %(num_vars, n))
        for i in inds:
            out.write(formula[i])
    return

def vacuous(clique_edges, num_vars):
    for ele in clique_edges:
        if ele[0] + num_vars == ele[1]:
            return True
    return False

def find_largest_gain(cliques, edges, prev_gain, num_vars, clique_edges_map, prev_gain_map):
    i = len(cliques) - 1
    tup = tuple(cliques[i])
    if tup in clique_edges_map:
        clique_edges = clique_edges_map[tup]
    else:
        clique_edges = all_edges(tup)
        clique_edges_map[tup] = clique_edges
    gain = 0
    num_edges = len(edges)
    to_remove = []
    gain_map = {} # upper bound of gain

    while gain < len(clique_edges) and i >= 0:
        #if vacuous(clique_edges, num_vars):
        #    to_remove.append(i)
        #    continue
        if tup in prev_gain_map and prev_gain_map[tup] < gain:
            gain_map[tup] = prev_gain_map[tup]
        else:
            new_gain = len(clique_edges.union(edges)) - num_edges
            gain_map[tup] = new_gain
            if new_gain >= 0.7 * prev_gain:
                gain = new_gain
                best = i
                best_edges = clique_edges
                break
            if new_gain > gain:
                gain = new_gain
                best = i
                best_edges = clique_edges
            elif new_gain / len(clique_edges) <= 0.3:
                to_remove.append(i)
        i -= 1
        tup = tuple(cliques[i])
        if tup in clique_edges_map:
            clique_edges = clique_edges_map[tup]
        else:
            clique_edges = all_edges(tup)
            clique_edges_map[tup] = clique_edges
    assert (gain > 0)
    return best, best_edges, gain, to_remove, clique_edges_map, gain_map

def greedy_hill_climbing(LIG, cliques, num_edges, num_vars, start_time=0, timeout=0):
    edges = all_edges(cliques[-1])
    clique_edges_map = {}
    cliques = cliques[:-1]
    selected_cliques = []
    prev_gain = len(edges)
    prev_gain_map = {}
    while len(edges) != num_edges:
        if timeout != 0 and time.time() - start_time > timeout:
            return selected_cliques, True
        #print (time.time() - start_time)
        #print ("\rPrevious Gain: {}; Target: {}; edges {}; Clique size: {}".format(prev_gain, num_edges, len(edges), len(cliques)), end="")
        ind, new_edges, prev_gain, to_remove, clique_edges_map, prev_gain_map = find_largest_gain(cliques, edges, prev_gain, num_vars, clique_edges_map, prev_gain_map)
        selected_cliques.append(cliques[ind])
        edges = edges.union(new_edges)
        del prev_gain_map[tuple(cliques[ind])]
        del clique_edges_map[tuple(cliques[ind])]
        del cliques[ind]
        for ele in to_remove:
            #assert(ele != ind)
            try:
                if ele > ind:
                    ele -= 1
                del prev_gain_map[tuple(cliques[ele])]
                del clique_edges_map[tuple(cliques[ele])]
                del cliques[ele]
            except:
                continue
    return selected_cliques, False


def all_edges(clique):
    clique_edges = set()
    for i in range(len(clique))[:-1]:
        for j in range(len(clique))[i + 1:]:
            min_ = min(clique[i], clique[j])
            max_ = max(clique[i], clique[j])
            clique_edges.add((min_, max_))
    return clique_edges

if __name__ == "__main__":
    main()
