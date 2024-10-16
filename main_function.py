import os
import gzip
from scipy import stats

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import roc_curve, auc, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import requests
import pickle
import powerlaw
import gc

import math
import random
from bisect import bisect

import os
def SearchFiles(directory, fileType):      
    fileList=[]    
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(fileType):
                fileList.append(fileName[:-4])
    return fileList
    
## Generate network and observational data
# 定义求度分布的函数：获取各个不同度值对应的概率（适用于网络节点数量比较少的情况）
def get_pdf(G):
    all_k = [G.degree(i) for i in G.nodes()]
    k = list(set(all_k))  # 获取所有可能的度值
    N = len(G.nodes())

    Pk = []
    for ki in sorted(k):
        c = 0
        for i in G.nodes():
            if G.degree(i) == ki:  
                c += 1  
        Pk.append(c/N)     

    return sorted(k), Pk

# 自己写二分查找函数
def bisection_search(array, a):
    n = len(array)
    jl = 0
    ju = n-1
    flag = (array[n-1]>=array[0]) # 判断array数组是否为升序排序
    while ju-jl > 1:
        jm = math.ceil((ju+jl)/2)
        if (a > array[jm]) == flag:
            jl = jm
        else:
            ju = jm

    if a == array[0]:
        j = 1
    elif a == array[n-1]:
        j = n-2
    else:
        j = jl + 1

    return j

# 隐参数模型生成无标度网络：详细算法见《巴拉巴西网络科学》3.8节
def generate_SF_network(N, gamma, L):
    alpha = 1 / (gamma - 1)
    n = np.linspace(1, N, N)
    eta = n ** (-alpha)
    nom_eta = eta / np.sum(eta)
    random.shuffle(nom_eta)
    cum_eta = np.array([np.sum(nom_eta[:i]) for i in range(N)])
    edges = []

    c = 0
    while c < L:
        i = bisect(cum_eta, np.random.rand(2)[0])
        j = bisect(cum_eta, np.random.rand(2)[1])
        if i == j:
            continue
        e1 = (i, j)
        e2 = (j, i)
        if e1 not in edges and e2 not in edges:
            edges.append(e1)
            c += 1

    G = nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from(range(N))
    
    degree_zero_nodes = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(degree_zero_nodes)
    return G

def generate_ground_nograph(nodes, gamma, avk, threshold):
    from networkx.algorithms.graphical import is_graphical
    from networkx.utils.random_sequence import powerlaw_sequence
    #avk=6
    termination=0
    step=0
    while 1-termination:
        #avk = 6
        L = int(avk*nodes/2)
        G = generate_SF_network(nodes, gamma, L)
        degree_dict = dict(G.degree())
        degree_sequence = sorted([d for n, d in degree_dict.items()], reverse=True)
        fit = powerlaw.Fit(degree_sequence)
        step = step + 1
        if (fit.alpha>threshold+0.03)|(step>20):
            termination=1
        else:
            termination=0
            
    return G,fit.alpha

def generate_error(inputdata):
    np.random.seed()
    N = inputdata[0]
    A = inputdata[1]
    alpha = inputdata[2]
    beta = inputdata[3]
    
    
    # The probability of observing an edge between nodes i and j can then be succinctly written as alpha^(A_ij)*beta^(1-A_ij)
    prob_matrix = np.power(alpha, A) * np.power(beta,1 - A)
    
    adj_temporary = np.random.binomial(N, prob_matrix)
    
    upper_triangle = np.triu(adj_temporary)
    symmetric_matrix = upper_triangle + upper_triangle.T - 2* np.diag(upper_triangle.diagonal())
    
#     print('误差数据已产生')
    return symmetric_matrix



## Inference
def inference_BA(inputpa):
    N = inputpa[0]
    G_error = inputpa[1]
    nodes = G_error.shape[0]
    N_ij = N * np.ones((nodes, nodes))
    E_ij = G_error

    ## 初始值
    w = 0.006
    alpha_1 = 0.5
    alpha_0 = 0.1
    phi = np.random.random(nodes)
    phi = (nodes / phi.sum()) * phi # 用E-IJ
    steps = 0

    phi_th = min(nodes*np.sum(E_ij/N,1)/sum(sum(E_ij/N)))
    convergence = False
    while 1 - convergence:
        steps += 1
        
        phi_matrix = np.outer(phi, phi)

        expM = np.power(alpha_1, E_ij)
        expM_1 = np.power(1 - alpha_1, N_ij - E_ij)
        expM_01 = np.power(1 - alpha_0, N_ij - E_ij)
        expM_0 = np.power(alpha_0, E_ij)
        u = w * np.multiply(phi_matrix, np.multiply(expM, expM_1))
        s = np.multiply(expM_0, expM_01) + u
        Q_ij = np.divide(u, s)
        np.fill_diagonal(Q_ij, 1-np.exp(-0.5*w*phi**2))

        phi_new = nodes * (np.sum(Q_ij,axis=1) / Q_ij.sum())
        index_phi = phi_new <=(phi_th)
        phi_new[index_phi] = phi_th

        w_new = Q_ij.sum() / (nodes * (nodes-1))

        alpha_1_new = (Q_ij * E_ij).sum() / (N_ij * Q_ij).sum()

        alpha_0_new = ((np.ones((nodes, nodes)) - Q_ij) * E_ij).sum() / (N_ij * (np.ones((nodes, nodes)) - Q_ij)).sum()

        if (max(abs(np.array([alpha_1_new,alpha_0_new]) - np.array([alpha_1,alpha_0]))) < 1e-7) | (steps > 5000):
            convergence = True
            w = w_new
            alpha_1 = alpha_1_new
            alpha_0 = alpha_0_new
            phi = phi_new

#             print('循环停止，共迭代%s步' % steps)
            break
        else:
            alpha_1 = alpha_1_new
            alpha_0 = alpha_0_new
            w = w_new
            phi = phi_new

    return [Q_ij,steps]


def calculate_auc(inputdata):
    G = inputdata[0]
    Q_ij = inputdata[1]
    E_ij = inputdata[2]
    N = inputdata[3]
    G = nx.to_pandas_adjacency(G)
    # 将矩阵下三角部分提取出来并拉平
    row_idx, col_idx = np.tril_indices(n=np.array(G).shape[0], k=-1)
    # 按照下三角的顺序获取矩阵中的值，并将它们放入一维数组中
    d = np.array(G)[row_idx, col_idx]
    
    # 按照下三角的顺序获取矩阵中的值，并将它们放入一维数组中
    d1 = Q_ij[row_idx, col_idx]
    
    # baseline
    # 按照下三角的顺序获取矩阵中的值，并将它们放入一维数组中
    d2 = E_ij[row_idx, col_idx]/N
    
    from sklearn.metrics import precision_recall_curve, auc, roc_curve
    tpr, fpr, tt  = roc_curve(d,d1)
    auc_n = auc(tpr,fpr)
    tpr, fpr, tt  = roc_curve(d,d2)
    auc_baseline = auc(tpr,fpr)
    
    from scipy import integrate
    precision, recall, _  = precision_recall_curve(d,d1,pos_label=1)
    sorted_index = np.argsort(precision)
    fpr_list_sorted =  np.array(precision)[sorted_index]
    tpr_list_sorted = np.array(recall)[sorted_index]
    auprc = integrate.trapz(y=tpr_list_sorted, x=fpr_list_sorted)
#     print('AUPRC:{}'.format(auprc))

    # print(auc(precision,recall))

    precision, recall, _  = precision_recall_curve(d,d2,pos_label=1)
    sorted_index = np.argsort(precision)
    fpr_list_sorted =  np.array(precision)[sorted_index]
    tpr_list_sorted = np.array(recall)[sorted_index]
    auprc_baseline = integrate.trapz(y=tpr_list_sorted, x=fpr_list_sorted)
#     print('AUPRC_baseline:{}'.format(auprc_baseline))

    return [auc_n, auc_baseline, auprc, auprc_baseline]


def main_function(inputd):
#     print('开始计算')
    G_temp = inputd[0]
    N  = inputd[1]
    alpha = inputd[2]
    beta = inputd[3]
    gammax = inputd[4]
    
    A = np.array(nx.to_pandas_adjacency(G_temp))
    E = generate_error([N,A,alpha,beta])
    
    Qq = inference_BA([N,E])
    Q = Qq[0]
    step= Qq[1]
    
    all_index = calculate_auc([G_temp,Q,E,N])
    
    return [gammax,E,Q,all_index,step]

def parallel_calculation(G,N,alpha,beta):
    
    Q_all = dict()
    E_all = dict()
    index_all = dict()
    step_all = dict()
    
    input_data = list()
    for gammax in list(G.keys()):
        Q_all['%s' % gammax] = list()
        E_all['%s' % gammax] = list()
        index_all['%s' % gammax] = list()
        step_all['%s' % gammax] = list()
    
        G_temp = G[gammax]
        for k in range(25):
            input_data.append([G_temp,N,alpha,beta,gammax])
    print('输入数据个数，l=%s' % len(input_data))       
    with Pool(40) as p:
        result = list(tqdm(p.imap(main_function, input_data), total=len(input_data)))
    

    for i in range(len(result)):
        E_temp = E_all['%s' % result[i][0]]
        E_temp.append(result[i][1])
        E_all['%s' % result[i][0]] = E_temp
        
        Q_temp = Q_all['%s' % result[i][0]]
        Q_temp.append(result[i][2])
        Q_all['%s' % result[i][0]] = Q_temp
        
        index_temp = index_all['%s' % result[i][0]]
        index_temp.append(result[i][3])
        index_all['%s' % result[i][0]] = index_temp
        
        step_temp = index_all['%s' % result[i][0]]
        step_temp.append(result[i][4])
        step_all['%s' % result[i][0]] = step_temp
    
    return E_all,Q_all,index_all,step_all


def calculate_all_index(index_all):
    AUC = pd.DataFrame(None)
    AUC_baseline = pd.DataFrame(None)
    AUPRC = pd.DataFrame(None)
    AUPRC_baseline = pd.DataFrame(None)
    for gammax in list(index_all.keys()):
        temp = index_all[gammax]
        c = pd.DataFrame(columns=['auc', 'auc_baseline', 'auprc', 'auprc_baseline'])
        for k in range(0,len(temp),2):
            c.loc[k] = temp[k]
        
        AUC['%s' % gammax] = c.auc
        AUC_baseline['%s' % gammax] = c.auc_baseline
        AUPRC['%s' % gammax] = c.auprc
        AUPRC_baseline['%s' % gammax] = c.auprc_baseline
        
        print('gamma=%s 已完成' % gammax)
    return AUC,AUC_baseline,AUPRC,AUPRC_baseline

## Properties
def degree_heterogeneity(x):
    d = []
    for gammax in list(x.keys()):
        z = x[gammax]
        de = nx.degree(z)
        unique_degrees = [v for k, v in de]
        de = np.array(unique_degrees)
        d.append(np.var(de)/(np.mean(de)))
    return d

def estimate_gamma(x):
    import sys
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
    
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout
    d = []
    for gammax in list(x.keys()):
        z = x[gammax]
        degree_dict = dict(z.degree())
        degree_sequence = sorted([d for n, d in degree_dict.items()], reverse=True)
        with HiddenPrints():
            fit = powerlaw.Fit(degree_sequence)
        d.append(fit.alpha)
    return d

def calculate_nodes(x):
    d = []
    for gammax in list(x.keys()):
        z = x[gammax]
        degree_dict = dict(z.degree())
        degree_sequence = sorted([d for n, d in degree_dict.items()], reverse=True)
        
        d.append(len(degree_sequence))
    return d

def read_mean_degree(x):
    nn = []
    for gammax in list(x.keys()):
        z = x[gammax]
        nn.append(2*z.number_of_edges()/z.number_of_nodes())
    return np.array(nn)

def read_num_node(g):
    nn = []
    for xx in list(g.keys()):
        nn.append(g[xx].number_of_nodes())
    return np.array(nn)




def q_gamma(N,ga,a_z):
    phi_a = np.zeros(len(a_z))
    for j in range(len(a_z)):
        if a_z[j]<=(1/N**(2/(ga-1))):
            phi_a[j] = 0

        elif a_z[j] >=N**(2/(ga-1)):
            phi_a[j] = part22(ga,a_z[j])

        elif (a_z[j] <N**(2/(ga-1)))&(a_z[j]>=1):
            phi_a[j] = part32(ga,a_z[j])

        elif (a_z[j] >=(1/N**(2/(ga-1))))&(a_z[j]<1):
            phi_a[j] = part42(ga,a_z[j])
        phi_a = phi_a 
    return phi_a

def part22(ga,a):
    
    return 1

def part32(ga,a):
    
    return 1-a**((-ga+2)) * ((ga-1)/(2*ga-3)) 

def part42(ga,a):

    return a**((ga-1)) * ((ga-2)/(2*ga-3))



def modBin(k, n, p):
    from scipy import stats
    if k <= n:
        return stats.binom.pmf(k, n, p)
    else:
        return 0

def pz(z, n, p1, p2):
    prob = 0
    if z > 0:
        for i in range(n):
            prob += modBin(i + z, n, p1) * modBin(i, n, p2)
    else:
        for i in range(n):
            prob += modBin(i + z, n, p1) * modBin(i, n, p2)
    return prob


### overlapping
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
    
def ROC(label, y_prob):
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point
    
def reconstruction_error(d,d1,d2):
    fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(d, d1)
    d1_hat_temp = np.where(d1 > optimal_th, 1, 0)
    error = np.linalg.norm(d - d1_hat_temp, ord=1)
    # error = 1 - np.linalg.norm(d - d1, ord=1) / np.linalg.norm(d + d1, ord=1)
    

    fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(d, d2)
    d2_hat_temp = np.where(d2 > optimal_th, 1, 0)
    error_base = np.linalg.norm(d - d2_hat_temp, ord=1)
    # error_base = 1 - np.linalg.norm(d - d2, ord=1) / np.linalg.norm(d + d2, ord=1)

    cons = len(d)
    return [error/cons, error_base/cons]
    
def calculate_reconstruction_error(gg,e_q):
    r_e = pd.DataFrame(None)
    r_e_baseline = pd.DataFrame(None)
    
    for gammax in list(gg.keys()):
        A_ij = nx.to_numpy_array(gg[gammax])
        row_idx,col_idx = np.tril_indices(n=A_ij.shape[0], k=-1)
        d = A_ij[row_idx,col_idx]
        c = pd.DataFrame(columns=['reconstruction_error', 'reconstruction_error_baseline'])
        
        for k in range(len(e_q[1][gammax])):
            d1 = e_q[1][gammax][k][row_idx,col_idx]
            d2 = e_q[0][gammax][k][row_idx,col_idx]/10
            
            c.loc[k] = reconstruction_error(d,d1,d2)
        
        r_e['%s' % gammax] = c.reconstruction_error
        r_e_baseline['%s' % gammax] = c.reconstruction_error_baseline
        
        print('gamma=%s 已完成' % gammax)
    return r_e,r_e_baseline


def binary_matrix(gg,q):
    A_ij = nx.to_numpy_array(gg)
    row_idx,col_idx = np.tril_indices(n=A_ij.shape[0], k=-1)
    d1 = np.sum(A_ij,1)
    d = A_ij[row_idx,col_idx]
    matrix = np.maximum.outer(d1, d1)[row_idx,col_idx]
    d1_hat = 0
    Q_hat = 0
    for i in range(len(q)):
        Q_ij = q[i]
        d1 = Q_ij[row_idx,col_idx]

        fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(d, d1)
        d1_hat_temp = np.where(d1 > optimal_th, 1, 0)
        Q_hat_temp = np.where(Q_ij > optimal_th, 1, 0)
        d1_hat = d1_hat + d1_hat_temp
        
        Q_hat = Q_hat + Q_hat_temp
    d_id_j = np.dot(np.sum(A_ij,1).reshape(A_ij.shape[0],1),np.sum(A_ij,1).reshape(1,A_ij.shape[0]))[row_idx,col_idx]
    Q_hat = Q_hat.astype(float)
    Q_hat[A_ij == 0] = np.nan
    np.fill_diagonal(Q_hat,np.nan)
    return [d1_hat,d_id_j,Q_hat,np.sum(A_ij,1),d,matrix,Q_hat[row_idx,col_idx]]