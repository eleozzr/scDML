#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 23:42:14 2021
@author: xiaokangyu
"""
from pandas import value_counts
import scanpy as sc
import scipy
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans,MiniBatchKMeans

def Normalization(adata, batch_key ="BATCH",n_high_var = 1000,hvg_list=None, 
                     normalize_samples = True,target_sum=1e4,log_normalize = True, 
                     normalize_features = True,scale_value=10.0,verbose=True,log=None):
    """
    Normalization of raw dataset 
    ------------------------------------------------------------------
    Argument:
        - adata: raw adata to be normalized

        - batch_key: `str`, string specifying the name of the column in the observation dataframe which identifies the batch of each cell. If this is left as None, then all cells are assumed to be from one batch.
    
        - n_high_var: `int`, integer specifying the number of genes to be idntified as highly variable. E.g. if n_high_var = 1000, then the 1000 genes with the highest variance are designated as highly variable.
       
        - hvg_list: 'list',  a list of highly variable genes for seqRNA data
        
        - normalize_samples: `bool`, If True, normalize expression of each gene in each cell by the sum of expression counts in that cell.
        
        - target_sum: 'int',default 1e4,Total counts after cell normalization,you can choose 1e6 to do CPM normalization
            
        - log_normalize: `bool`, If True, log transform expression. I.e., compute log(expression + 1) for each gene, cell expression count.
        
        - normalize_features: `bool`, If True, z-score normalize each gene's expression.

    Return:
        Normalized adata
    ------------------------------------------------------------------
    """
    
    n, p = adata.shape
    
    if(normalize_samples):
        if(verbose):
            log.info("Normalize counts per cell(sum={})".format(target_sum))
        sc.pp.normalize_total(adata, target_sum=target_sum)
        
    if(log_normalize):
        if(verbose):
            log.info("Log1p data")
        sc.pp.log1p(adata)
    
    if hvg_list is None:      
        if(verbose):
            log.info("Select HVG(n_top_genes={})".format(n_high_var))
        sc.pp.highly_variable_genes(adata,n_top_genes=n_high_var,subset=True)
    else:
        log.info("Select HVG from given highly variable genes list")
        adata = adata[:, hvg_list]
    
    adata.obs["batch"]="1"    
    if normalize_features:
        if(len(adata.obs[batch_key].value_counts())==1): # single batch
            if(verbose):
                log.info("Scale batch(scale_value={})".format(scale_value))
            sc.pp.scale(adata,max_value=scale_value)
            adata.obs["batch"]=1
        else:
            if(verbose):
                log.info("Scale batch(scale_value={})".format(scale_value))
            adata_sep=[]
            for batch in np.unique(adata.obs[batch_key]):
                sep_batch=adata[adata.obs[batch_key]==batch]
                sc.pp.scale(sep_batch,max_value=scale_value)
                adata_sep.append(sep_batch)
            adata=sc.AnnData.concatenate(*adata_sep)

    #adata.layers["normalized_input"] = adata.X        
    return adata
  
def dimension_reduction(adata,dim=100,verbose=True,log=None):
    """
    apply dimension reduction with normalized dataset 
    ------------------------------------------------------------------
    Argument:
        - adata: normazlied adata  

        - dim: ’int‘, default:100, number of principal components in PCA dimension reduction
    
        - verbose: print additional infomation

    Return:
        diemension reudced adata
    ------------------------------------------------------------------
    """
    if(verbose):
        log.info("Calculate PCA(n_comps={})".format(dim))
        
    if(adata.shape[0]>300000):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    sc.tl.pca(adata,n_comps=dim)
    emb=sc.AnnData(adata.obsm["X_pca"])
    return emb
    
def init_clustering(emb,reso=3.0,cluster_method="louvain",num_cluster=50,verbose=False,log=None):
    """
    apply clustering algorithm in PCA embedding space, defualt: louvain clustering
    ------------------------------------------------------------------
    Argument:
        - emb: 'AnnData',embedding data of adata(PCA)

        - reso: ’float‘, default:3.0, resolution defined in louvain(or leiden) algorithm
        
        - cluster_method: 'str', clustering algorothm to initize scDML cluster
        
        - num_cluster: 'int', default:40, parameters for kmeans(or minibatch-kmeans) clustering algorithm
    ------------------------------------------------------------------
    """
    if(cluster_method=="louvain"):
        sc.pp.neighbors(emb,random_state=0)
        sc.tl.louvain(emb,resolution=reso,key_added="init_cluster")
        if(verbose):
            log.info("Apply louvain clustring(resolution={}) initization".format(reso))
            log.info("Number of Cluster ={}".format(len(emb.obs["init_cluster"].value_counts())))
            log.info("clusters={}".format([i for i in range(len(emb.obs["init_cluster"].value_counts()))]))
    elif(cluster_method=="leiden"):
        sc.pp.neighbors(emb,random_state=0)
        sc.tl.leiden(emb,resolution=reso,key_added="init_cluster")
        if(verbose):
            log.info("Apply leiden clustring(resolution={})  initization".format(reso))
            log.info("Number of Cluster ={}".format(len(emb.obs["init_cluster"].value_counts())))
            log.info("clusters={}".format([i for i in range(len(emb.obs["init_cluster"].value_counts()))]))
    elif(cluster_method=="kmeans"):
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(emb.X) 
        emb.obs['init_cluster'] = kmeans.labels_.astype(str)
        emb.obs['init_cluster'] = emb.obs['init_cluster'].astype("category")   
        if(verbose):
            log.info("Apply kmeans clustring(num_cluster={}) initization".format(num_cluster))
    elif(cluster_method=="minibatch-kmeans"): # this cluster method will reduce time and memory but less accuracy
        kmeans = MiniBatchKMeans(init='k-means++',n_clusters=num_cluster,random_state=0,batch_size=64).fit(emb.X)
        emb.obs['init_cluster'] = kmeans.labels_.astype(str)
        emb.obs['init_cluster'] = emb.obs['init_cluster'].astype("category")
        if(verbose):
            log.info("Apply minibatch-kmeans clustring(num_cluster={}) initization".format(num_cluster))
    else:
        if(verbose):
            log.info("Not implemented!!!")
        raise IOError

        



