#from sklearn import preprocessing
import sys
import subprocess
sys.path.append("/Library/Python/2.7/site-packages")
import networkx as nx
import numpy as np
import scipy as sp
from scipy import stats
import math
from pulp import *
from optparse import OptionParser
import matplotlib as plt
import community
import csv

"""
This file takes in a dimacs file, calculates the features of it and stores them in a
.txt file.
"""

def main():
    parser = getParser()

    (options, args) = parser.parse_args()

    path_to_formulas = options.path_to_formulas
    benchmark_set_name = path_to_formulas.split("/")[-2]
    out_name = options.out_file
    scale_free = options.scale_free

    title = [ "num. vars", "num. clauses", "VIG clust.",
              "mod. VIG", "mod. LCG", "mod. VCG", "mod. LCG",
              "var. alpha", "clause alpha"]

    lines = []
    for filename in os.listdir(path_to_formulas):
        source = path_to_formulas + filename
        cnf = open(source)
        content = cnf.readlines()
        cnf.close()

        print ("Successfully read generated file {}".format(source))

        while content[0].split()[0] == 'c':
            content = content[1:]
        while len(content[-1].split()) <= 1:
            content = content[:-1]


        parameters = content[0].split()
        formula = content[1:]
        formula = to_int_matrix(formula)
        (formula, num_clauses) = remove_duplicate(formula)
        
        num_vars = int(parameters[2])

        assert (get_vacuous(formula) == 0)
        assert(num_vars != 0)
        assert(num_clauses == len(formula))

        VIG = nx.Graph()
        VIG.add_nodes_from(range(num_vars+1)[1:])

        LIG = nx.Graph()
        LIG.add_nodes_from(range(num_vars * 2 + 1)[1:])

        VCG = nx.Graph()
        VCG.add_nodes_from(range(num_vars + num_clauses + 1)[1:])

        LCG = nx.Graph()
        VCG.add_nodes_from(range(2 * num_vars + num_clauses + 1)[1:])

        preprocess_VIG(formula, VIG) # Build a VIG
        preprocess_LIG(formula, LIG, num_vars) # Build a LIG
        preprocess_VCG(formula, VCG, num_vars) # Build a VCG
        preprocess_LCG(formula, LCG, num_vars) # Build a VCG

        features = []
        features.append(num_vars)
        features.append(num_clauses)
        features += [nx.average_clustering(VIG)]
        features += get_modularities(VIG, LIG, VCG, LCG) # Modularities of VIG & VCG
        features += get_scale_free(source, scale_free)
        lines.append(features)

    if out_name != None:
        with open(out_name, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(title, lines)
    else:
        lines = np.array(lines)
        means = np.nanmean(lines, axis=0)
        std = np.nanstd(lines, axis=0)
        for i, column_name in enumerate(title):
            print("mean/std {}: {}/{}".format(column_name, means[i], std[i]))

def getParser():
    parser = OptionParser(usage="usage: %prog [options] formula outfile",
                          version="%prog 1.0")
    parser.add_option("-o", "--out",
                      dest="out_file",
                      default=None,
                      help="save stats into a file")
    parser.add_option("-s", "--scale-free",
                      dest="scale_free",
                      default=None,
                      help="enter the path to the scale-free computing binary")
    parser.add_option("-d", "--path-to-formulas",
                      dest="path_to_formulas",
                      default=None,
                      help="Path to the directory of the formulas")
    return parser


#--------------------------------------------Unit Clause---------------------------------------------------#
def no_unit_clause(formula):
    for line in formula:
        if len(line) == 1:
            return False
    return True


#--------------------------------------------preprocess---------------------------------------------------#
def to_int_matrix(formula):
    for i in range(len(formula)):
        formula[i] = list(map(int, formula[i].split()))[: -1]
    return formula

def get_cl_string(clause):
    s = ""
    clause.sort()
    for ele in clause:
        s += str(ele) + ","
    return s[:-1]

def remove_duplicate(formula):
    cs = []
    new_formula = []
    num_clause = 0
    for line in formula:
        c = get_cl_string(line)
        if c not in cs:
            num_clause += 1
            new_formula.append(line)
            cs.append(c)
    return (new_formula, num_clause)


def preprocess_VIG(formula, VIG):
    """
    Builds VIG.
    """
    for cn in range(len(formula)):
        for i in range(len(formula[cn])-1):
            for j in range(len(formula[cn]))[i+1:]:
                VIG.add_edge(abs(formula[cn][i]), abs(formula[cn][j]))


def preprocess_LIG(formula, LIG, num_vars):
    for cn in range(len(formula)):
        for i in range(len(formula[cn])-1):
            for j in range(len(formula[cn]))[i+1:]:
                if formula[cn][i] > 0:
                    fst = formula[cn][i]
                else:
                    fst = abs(formula[cn][i]) + num_vars
                if formula[cn][j] > 0:
                    snd = formula[cn][j]
                else:
                    snd = abs(formula[cn][j]) + num_vars
                LIG.add_edge(fst, snd)

def preprocess_VCG(formula, VCG, num_vars):
    """
    Builds VCG
    """
    for cn in range(len(formula)):
        for var in formula[cn]:
            VCG.add_edge(abs(var), cn + num_vars + 1)


def preprocess_LCG(formula, LCG, num_vars):
    """
    Builds LCG
    """
    for cn in range(len(formula)):
        for var in formula[cn]:
            if var > 0:
                LCG.add_edge(abs(var), cn + num_vars + 1)
            else:
                LCG.add_edge(abs(var) + num_vars, cn + num_vars + 1)


def get_pos_neg(formula, LCG, num_vars):
    degrees = dict(LCG.degree())
    lst = []
    for var in range(num_vars + 1)[1:]:
        if var in degrees:
            if var + num_vars in degrees:
                lst.append(degrees[var] * 1.0 / (degrees[var] + degrees[var + num_vars]))
            else:
                lst.append(1.0)
        else:
            lst.append(0.0)
    return lst




#--------------------------------------------feature extraction methods-------------------------------------#

#-------------------Formula related----------------------#
def pure_variables(formula, num_vars):
    lst = [0] * num_vars
    num_pure = 0
    for line in formula:
        for ele in line:
            if ele > 0 and (lst[ele - 1] == 0 or lst[ele - 1] == 2):
                lst[ele - 1] += 3 # if pos, add three to lst[ele - 1]
            if ele < 0 and (lst[abs(ele) - 1] == 0 or lst[abs(ele) - 1] == 3):
                lst[abs(ele) - 1] += 2 #if neg, add two to lst[ele - 1]
    for i in range(len(lst)):
        if lst[i] == 2 or lst[i] == 3:
            num_pure += 1
    return [num_pure]


def get_binary(formula, num_clause):
    """
    get the ratio of binary clauses, ternary clauses, long clauses
    """
    num_bi = 0
    for line in formula:
        if len(line) == 2:
            num_bi += 1
    return [float(num_bi) / num_clause]


def get_vacuous(formula):
    vac = 0
    for line in formula:
        vac_loc = 0
        for ele in line:
            if -ele in line:
                vac_loc = 1
        if vac_loc == 1:
            vac += 1
    return vac

#-------------------Graph related----------------------#

def VCG_var_deg(VCG, num_vars):
    var_degs = []
    for var in range(num_vars + 1)[1:]:
        var_degs.append(VCG.degree[var])
    return add_stat(var_degs)

def VCG_clause_deg(VCG, num_vars, num_clauses, formula):
    clause_degs = []
    for clause in range(num_clauses):
        clause_degs.append(VCG.degree[clause + num_vars + 1])
    #assert (len(clause_degs) == num_clauses)
    #print ("number of different clause degrees:", len(set(clause_degs)))
    return add_stat(clause_degs)

def VCG_num_edges(VCG, formula):
    num_edges = 0
    for line in formula:
        num_edges += len(line)
    #assert(num_edges == VCG.number_of_edges())
    return [num_edges]


def get_binary_subgraph(formula):
    bin_formula = []
    for clause in formula:
        if len(clause) == 2:
            bin_formula.append(clause)
    return bin_formula


def get_long_subgraph(formula):
    long_formula = []
    for clause in formula:
        if len(clause) > 2:
            long_formula.append(clause)
    return long_formula

#***********************************************Modularity-related features ***************************************

def get_modularities(VIG, LIG, VCG, LCG):
    """
    Returns the modularities of VIG, LIG and VCG representations of the formula
    """
    part_VIG = community.best_partition(VIG)
    mod_VIG = community.modularity(part_VIG, VIG) # Modularity of VIG
    num_parts = len(part_VIG)

    part_LIG = community.best_partition(LIG)
    mod_LIG = community.modularity(part_LIG, LIG) # Modularity of VCG

    part_VCG = community.best_partition(VCG)
    mod_VCG = community.modularity(part_VCG, VCG) # Modularity of VCG


    part_LCG = community.best_partition(LCG)
    mod_LCG = community.modularity(part_LCG, LCG) # Modularity of LCG

    return [mod_VIG, mod_LIG, mod_VCG, mod_LCG]

#------------------------------------------- Subprocesses -------------------------------------------#


def get_scale_free(source, scale_free=False):
    feats = []
    f = open("blah.txt", "w")
    if scale_free:
        subprocess.call([scale_free, source], stdout=f)
    else:
        subprocess.call(["/Users/anwu/Monkeyswedding/Projects/SAT_GAN/sat_gen/cpp/scalefree", source], stdout=f)
    f.close()
    with open("blah.txt", 'r') as f:
        for line in f.readlines():
            if ("alpha" in line):# or ("min. value" in line) or ("beta" in line) or ("max. error" in line):
                feats.append(line.split()[-1])
    os.remove("blah.txt")
    return list(map(float, feats))

#-----------------------------------------------statistics-------------------------------------------------#

def add_stat(lst):
    """
    add max, min, mean, std of the give statistics to the features list.
    """
    return [max(lst),min(lst), np.mean(lst), np.std(lst)]

def kl_div(orig_data, data):
    size = int (max(max(orig_data), max(data)))
    orig_occurence = [0] * size
    for ele in orig_data:
        orig_occurence[ele - 1] += 1

    occurence = [0] * size
    for ele in data:
        occurence[ele - 1] += 1

    return stats.entropy(occurence, qk = orig_occurence)

if __name__ == "__main__":
    main()
