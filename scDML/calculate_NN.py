#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 03:02:45 2021

@author: xiaokangyu
"""
import numpy as np
import hnswlib
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
import itertools
from joblib import Parallel, delayed
import time, math

def nn_approx(ds1, ds2, names1, names2, knn=50, return_distance=False,metric="cosine",flag="in"):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    if(metric=="euclidean"):
        tree = hnswlib.Index(space="l2", dim=dim)
    elif(metric=="cosine"):
        tree = hnswlib.Index(space="cosine", dim=dim)
    #square loss: 'l2' : d = sum((Ai - Bi) ^ 2)
    #Inner  product 'ip': d = 1.0 - sum(Ai * Bi)
    #Cosine similarity: 'cosine':d = 1.0 - sum(Ai * Bi) / sqrt(sum(Ai * Ai) * sum(Bi * Bi))
    tree.init_index(max_elements=num_elements, ef_construction=200, M=32) # refer to https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md for detail
    tree.set_ef(50)
    tree.add_items(ds2)
    ind, distances = tree.knn_query(ds1, k=knn)
    if(flag=="in"):
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[1:]:## 
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b):
                    match[(names1[a], names2[b_i])] = distances[a, b_ind]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match
    else:
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[0:]:## 
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b):
                    match[(names1[a], names2[b_i])] = distances[a, b_ind]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match

def nn(ds1, ds2, names1, names2, knn=50, metric_p=2, return_distance=False,metric="cosine",flag="in"):
    # Find nearest neighbors of first dataset.
    if(flag=="in"):
        nn_ = NearestNeighbors(n_neighbors=knn, metric=metric)  # remove self
        nn_.fit(ds2)
        nn_distances, ind = nn_.kneighbors(ds1, return_distance=True)
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[1:]:
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b):
                    match[(names1[a], names2[b_i])] = nn_distances[a, b_ind]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match
    else:
        nn_ = NearestNeighbors(n_neighbors=knn, metric=metric)  # remove self
        nn_.fit(ds2)
        nn_distances, ind = nn_.kneighbors(ds1, return_distance=True)
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[0:]:
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b):
                    match[(names1[a], names2[b_i])] = nn_distances[a, b_ind]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match
        
def nn_annoy(ds1, ds2, names1, names2, knn=20,save=True, return_distance=False,metric="cosine",flag="in"):
    """ Assumes that Y is zero-indexed. """
    # Build index.
    if(metric=="cosine"):
        tree = AnnoyIndex(ds2.shape[1], metric="angular")#metric
        tree.set_seed(100)
    else:
        tree = AnnoyIndex(ds2.shape[1], metric=metric)#metric
        tree.set_seed(100)
    if save:
        tree.on_disk_build('annoy.index')
    for i in range(ds2.shape[0]):
        tree.add_item(i, ds2[i, :])
    tree.build(60)#n_trees=50
    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(tree.get_nns_by_vector(ds1[i, :], knn, search_k=-1)) #search_k=-1 means extract search neighbors
    ind = np.array(ind)
    # Match.
    if(flag=="in"):
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[1:]:
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            # get distance
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b:
                    match[(names1[a], names2[b_i])] = tree.get_distance(a, b_i)
            return match
    else:
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[0:]:
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            # get distance
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b:
                    match[(names1[a], names2[b_i])] = tree.get_distance(a, b_i)
            return match

def mnn(ds1, ds2, names1, names2, knn=20, save=False, approx=True,approx_method="hnswlib", return_distance=False,metric="cosine",flag="in"):
    # Find nearest neighbors in first direction.

    if approx:
        if approx_method=="hnswlib":
            #hnswlib
            match1 = nn_approx(ds1, ds2, names1, names2, knn=knn,return_distance=return_distance,metric=metric,flag=flag)  # save_on_disk = save_on_disk)
            # Find nearest neighbors in second direction.
            match2 = nn_approx(ds2, ds1, names2, names1, knn=knn,return_distance=return_distance,metric=metric,flag=flag)  # , save_on_disk = save_on_disk)
        else:
            #annoy
            match1 = nn_annoy(ds1, ds2, names1, names2, knn=knn,save=save,return_distance=return_distance,metric=metric,flag=flag)  # save_on_disk = save_on_disk)
            # Find nearest neighbors in second direction.
            match2 = nn_annoy(ds2, ds1, names2, names1, knn=knn,save=save,return_distance=return_distance,metric=metric,flag=flag)  # , save_on_disk = save_on_disk)

    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn, return_distance=return_distance,metric=metric,flag=flag)
        match2 = nn(ds2, ds1, names2, names1, knn=knn, return_distance=return_distance,metric=metric,flag=flag)
    # Compute mutual nearest neighbors.
    if(flag=="in"):
        if not return_distance:
            # mutal are set
            mutual = match1 | set([(b, a) for a, b in match1])
            return mutual
        else:
            # mutal are set
            mutual = set([(a, b) for a, b in match1.keys()]) | set([(b, a) for a, b in match2.keys()])
            #distance list of numpy array
            distances = []
            for element_i in mutual:
                distances.append(match1[element_i])  # distance is sys so match1[element_i]=match2[element_2]
            return mutual, distances
    else:
        if not return_distance:
            # mutal are set
            mutual = match1 & set([(b, a) for a, b in match2])
            ####################################################
            # change mnn pair to symmetric
            mutual = mutual | set([(b,a) for (a,b) in mutual])
            ####################################################
            return mutual
        else:
            # mutal are set
            mutual = set([(a, b) for a, b in match1.keys()]) & set([(b, a) for a, b in match2.keys()])
            #distance list of numpy array
            distances = []
            for element_i in mutual:
                distances.append(match1[element_i])  # distance is sys so match1[element_i]=match2[element_2]
            return mutual, distances


## calculate KNN and MNN from data_matrix(embedding matrix) not anndata
def get_dict_mnn(data_matrix, batch_index, k=5, save=True, approx=True,approx_method="hnswlib", verbose=False, return_distance=False,metric="cosine",flag="in",log=None):

    #ipdb.set_trace()
    #assert type(adata) == sc.AnnData, "Please make sure `adata` is sc.AnnData"
    cell_names = np.array(range(len(data_matrix)))
    #batch_list = adata.obs[batch_key] if batch_key in adata.obs.columns else np.ones(adata.shape[0], dtype=str)
    batch_unique = np.unique(batch_index)
    cells_batch = []
    for i in batch_unique:
        cells_batch.append(cell_names[batch_index == i])
    mnns = set()
    mnns_distance = []
    if(flag=="in"):
        num_KNN=0
        if(verbose):
            log.info("Calculate KNN pair intra batch...........")
            log.info("K={}".format(k))
            log.info("metric={}".format(metric))
        for comb in list(itertools.combinations(range(len(cells_batch)), 1)):
            # comb=(2,3)
            i = comb[0]  # ith batch
            j = comb[0]  # ith batch
            if verbose:
                i_batch = batch_unique[i]
                j_batch = batch_unique[j]
                log.info("Processing datasets: {} = {}".format((i, j), (i_batch, j_batch)))
            target = list(cells_batch[j])
            ref = list(cells_batch[i])
            #ds1 = adata[target].obsm[dr_name]
            ds1=data_matrix[target]
            ds2=data_matrix[ref]
            names1 = target
            names2 = ref
            match = mnn(ds1, ds2, names1, names2, knn=k, save=save, approx=approx,approx_method=approx_method, return_distance=return_distance,metric=metric,flag=flag)
            mnns=mnns|match
            #mnns_distance.append(distances) # not need
            if verbose:
                log.info("There are ({}) KNN pairs when processing {}={}".format(len(match),(i, j), (i_batch, j_batch)))
                num_KNN=num_KNN+len(match)
        if(verbose):
            log.info("scDML finds ({}) KNN pairs in dataset finally".format(num_KNN)) 
        #print("done")        
        if not return_distance:
            return mnns
        else:
            return mnns, mnns_distance
    else:
        num_MNN=0
        if(verbose):
            log.info("Calculate MNN pair inter batch...........")
            log.info("K={}".format(k))
            log.info("metric={}".format(metric))
        for comb in list(itertools.combinations(range(len(cells_batch)), 2)):
            # comb=(2,3)
            i = comb[0]  # i batch
            j = comb[1]  # jth batch
            if verbose:# if verbose
                i_batch = batch_unique[i]
                j_batch = batch_unique[j]
                log.info("Processing datasets: {} = {}".format((i, j), (i_batch, j_batch)))

            target = list(cells_batch[j])
            ref = list(cells_batch[i])
            ds1 = data_matrix[target]
            ds2 = data_matrix[ref]
            names1 = target
            names2 = ref
            match = mnn(ds1, ds2, names1, names2, knn=k, save=save, approx=approx,approx_method=approx_method, return_distance=return_distance,metric=metric,flag=flag)
            mnns=mnns|match
            #mnns_distance.append(distances)
            if verbose:
                log.info("There are ({}) MNN pairs when processing {}={}".format(len(match),(i, j), (i_batch, j_batch)))
                num_MNN=num_MNN+len(match)
        if(verbose):
            log.info("scDML finds ({}) MNN pairs in dataset finally".format(num_MNN))
        #print("done")
        if not return_distance:
            return mnns
        else:
            return mnns, mnns_distance

        
## calculate KNN and MNN from data_matrix(embedding matrix) not anndata in parallel mode
def get_dict_mnn_para(data_matrix, batch_index, k=5, save=True, approx=True,approx_method="hnswlib", verbose=False, return_distance=False,metric="cosine",flag="in",njob=8,log=None):

    #ipdb.set_trace()
    #assert type(adata) == sc.AnnData, "Please make sure `adata` is sc.AnnData"
    cell_names = np.array(range(len(data_matrix)))
    #batch_list = adata.obs[batch_key] if batch_key in adata.obs.columns else np.ones(adata.shape[0], dtype=str)
    batch_unique = np.unique(batch_index)
    cells_batch = []
    for i in batch_unique:
        cells_batch.append(cell_names[batch_index == i])
    mnns = set()
    mnns_distance = []
    if(flag=="in"):
        num_KNN=0
        if(verbose):
            log.info("Calculate KNN pair intra batch...........")
            log.info("K={}".format(k))
            log.info("metric={}".format(metric))
     
        res = Parallel(n_jobs=njob)(delayed(mnn)(data_matrix[list(cells_batch[comb[0]])],data_matrix[list(cells_batch[comb[0]])] , list(cells_batch[comb[0]]), list(cells_batch[comb[0]]), knn=k, save=save, approx=approx,approx_method=approx_method, return_distance=return_distance,metric=metric,flag=flag) for comb in list(itertools.combinations(range(len(cells_batch)), 1)))

        mnns=list(itertools.chain(*res))
        if(verbose):
            log.info("scDML finds ({}) KNN pairs in dataset finally".format(len(mnns))) 
        #print("done")        
        if not return_distance:
            return mnns
        else:
            return mnns, mnns_distance
    else:
        num_MNN=0
        if(verbose):
            log.info("Calculate MNN pair inter batch...........")
            log.info("K={}".format(k))
            log.info("metric={}".format(metric))

        res = Parallel(n_jobs=njob)(delayed(mnn)(data_matrix[list(cells_batch[comb[1]])],data_matrix[list(cells_batch[comb[0]])] , list(cells_batch[comb[1]]), list(cells_batch[comb[0]]), knn=k, save=save, approx=approx,approx_method=approx_method, return_distance=return_distance,metric=metric,flag=flag) for comb in list(itertools.combinations(range(len(cells_batch)), 2)))

        mnns=list(itertools.chain(*res))
        if(verbose):
            log.info("scDML finds ({}) MNN pairs in dataset finally".format(len(mnns)))
        #print("done")
        if not return_distance:
            return mnns
        else:
            return mnns, mnns_distance





