from .data_preprocess import *
from .calculate_NN import get_dict_mnn,get_dict_mnn_para
from .utils import *
from .network import EmbeddingNet
from .logger import create_logger       ## import logger
from .pytorchtools import EarlyStopping ## import earlytopping

import os
from time import time
from scipy.sparse import issparse
from numpy.linalg import matrix_power
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners,reducers,distances

## create scDMLmodel
class scDMLModel:
    def __init__(self,verbose=True,save_dir="./results/"):
        """                               
        create scDMLModel object
        Argument:
        ------------------------------------------------------------------
        - verbose: 'str',optional, Default,'True', write additional information to log file when verbose=True
        - save_dir: folder to save result and log information
        ------------------------------------------------------------------
        """     
        self.verbose=verbose
        self.save_dir=save_dir 
 
        if not os.path.exists(self.save_dir): 
            os.makedirs(self.save_dir+"/")
        
        self.log = create_logger('',fh=self.save_dir+'log.txt')# create log file
        if(self.verbose):
            self.log.info("Create log file....") # write log information
            self.log.info("Create scDMLModel Object Done....") 
    
    ###  preprocess raw data to generate init cluster label 
    def preprocess(self,adata,cluster_method="louvain",resolution=3.0,batch_key="BATCH",n_high_var = 1000,hvg_list=None,normalize_samples = True,target_sum=1e4,log_normalize = True,
                   normalize_features = True,pca_dim=100,scale_value=10.0,num_cluster=50,mode="unsupervised"):
        """
        Preprocessing raw dataset
        Argument:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.

        - cluster_method: "str", the clustering algorithm to initizite celltype label["louvain","leiden","kmeans","minibatch-kmeans"]

        - resolution:'np.float', default:3.0, the resolution of louvain algorthm for scDML to initialize clustering

        - batch_key: `str`, string specifying the name of the column in the observation dataframe which identifies the batch of each cell. If this is left as None, then all cells are assumed to be from one batch.
    
        - n_high_var: `int`, integer specifying the number of genes to be idntified as highly variable. E.g. if n_high_var = 1000, then the 1000 genes with the highest variance are designated as highly variable.
       
        - hvg_list: 'list',  a list of highly variable genes for seqRNA data
        
        - normalize_samples: `bool`, If True, normalize expression of each gene in each cell by the sum of expression counts in that cell.
        
        - target_sum: 'int',default 1e4,Total counts after cell normalization,you can choose 1e6 to do CPM normalization
            
        - log_normalize: `bool`, If True, log transform expression. I.e., compute log(expression + 1) for each gene, cell expression count.
        
        - normalize_features: `bool`, If True, z-score normalize each gene's expression.

        - pca_dim: 'int', number of principal components

        - scale_value: parameter used in sc.pp.scale() which uses to truncate the outlier

        - num_cluster: "np.int", K parameter of kmeans

        Return:
        - normalized adata suitable for integration of scDML in following stage.  
        """
        if(mode=="unsupervised"):
            batch_key=checkInput(adata,batch_key,self.log)
            self.batch_key=batch_key
            self.reso=resolution
            self.cluster_method=cluster_method 
            self.nbatch=len(adata.obs[batch_key].value_counts())
            if(self.verbose):        
                self.log.info("Running preprocess() function...")
                self.log.info("mode={}".format(mode))
                self.log.info("clustering method={}".format(cluster_method))
                self.log.info("resolution={}".format(str(resolution)))
                self.log.info("BATCH_key={}".format(str(batch_key)))

            self.norm_args = (batch_key,n_high_var,hvg_list,normalize_samples,target_sum,log_normalize, normalize_features,scale_value,self.verbose,self.log)
            normalized_adata = Normalization(adata,*self.norm_args)
            emb=dimension_reduction(normalized_adata,pca_dim,self.verbose,self.log)
            init_clustering(emb,reso=self.reso,cluster_method=cluster_method,verbose=self.verbose,log=self.log)
            
            self.batch_index=normalized_adata.obs[batch_key].values
            normalized_adata.obs["init_cluster"]=emb.obs["init_cluster"].values.copy()
            self.num_init_cluster=len(emb.obs["init_cluster"].value_counts())
            if(self.verbose):
                self.log.info("Preprocess Dataset Done...")
            return normalized_adata
        elif(mode=="supervised"):
            batch_key=checkInput(adata,batch_key,self.log)
            self.batch_key=batch_key
            self.reso=resolution
            self.cluster_method=cluster_method 
            self.nbatch=len(adata.obs[batch_key].value_counts())
            self.norm_args = (batch_key,n_high_var,hvg_list,normalize_samples,target_sum,log_normalize, normalize_features,scale_value,self.verbose,self.log)
            normalized_adata = Normalization(adata,*self.norm_args)
            if(self.verbose):
                self.log.info("mode={}".format(mode))
                self.log.info("BATCH_key={}".format(str(batch_key)))
                self.log.info("Preprocess Dataset Done...")
            return normalized_adata

    ### convert normalized adata to training data for scDML    
    def convertInput(self,adata,batch_key="BATCH",celltype_key=None,mode="unsupervised"): 
        """
            convert normalized adata to training data
            Argument:
            ------------------------------------------------------------------
            - adata: `anndata.AnnData`, normalized adata
            - batch_key: `str`, string specifying the batch name in adata.obs
            - mode : "str", "unsupervised" or "supervised"
        """
        if(mode=="unsupervised"):
            checkInput(adata,batch_key=batch_key,log=self.log)# check batch
            if("X_pca" not in adata.obsm.keys()): # check pca
                sc.tl.pca(adata)
            if("init_cluster" not in adata.obs.columns): # check init clustering
                sc.pp.neighbors(adata,random_state=0)
                sc.tl.louvain(adata,key_added="init_cluster",resolution=3.0)   #

            if(issparse(adata.X)):  
                self.train_X=adata.X.toarray()
            else:
                self.train_X=adata.X.copy()
            self.nbatch=len(adata.obs[batch_key].value_counts())
            self.train_label=adata.obs["init_cluster"].values.copy()
            self.emb_matrix=adata.obsm["X_pca"].copy()
            self.batch_index=adata.obs[batch_key].values
            self.merge_df=pd.DataFrame(adata.obs["init_cluster"])
            if(self.verbose):
                self.merge_df.value_counts().to_csv(self.save_dir+"cluster_distribution.csv")

            if(celltype_key is not None):
                self.celltype=adata.obs[celltype_key].values#
            else:
                self.celltype=None
        elif(mode=="supervised"): 
            if(celltype_key is None): # check celltype_key
                self.log.info("please provide celltype key in supervised mode!!!")
                raise IOError
            if(issparse(adata.X)):  
                self.train_X=adata.X.toarray()
            else:
                self.train_X=adata.X.copy()

            self.celltype=adata.obs[celltype_key].values#
            self.ncluster=len(adata.obs[celltype_key].value_counts())
            self.merge_df=pd.DataFrame()
            self.merge_df["nc_"+str(self.ncluster)]=self.celltype
            self.merge_df["nc_"+str(self.ncluster)]=self.merge_df["nc_"+str(self.ncluster)].astype("category").cat.codes
            
    ### calculate connectivity      
    def calculate_similarity(self,K_in=5,K_bw=10,K_in_metric="cosine",K_bw_metric="cosine"):
        """
        calculate connectivity of cluster with KNN and MNN pair for scDML
        
         Argument:
        ------------------------------------------------------------------
        - K_in: `int`,default:5 ,select K_in neighbours around each cell intra batch with KNN pair
        - K_bw: 'int',default:10,select K_bw neighbours to calculate mutual neartest neighbours 
        - K_in_metric: 'str',default:"cosine", select type of distance to calculate KNN
        - K_bw_metric: 'str',default:"cosine", select type of distance to calculate MNN
        ------------------------------------------------------------------
        """
        self.K_in=K_in
        self.K_bw=K_bw
        if(self.verbose):
            self.log.info("K_in={},K_bw={}".format(K_in,K_bw))
            self.log.info("Calculate similarity of cluster with KNN and MNN")
        if(self.nbatch<10):
            if(self.verbose):
                self.log.info("appoximate calculate KNN Pair intra batch...")
            knn_intra_batch_approx=get_dict_mnn(data_matrix=self.emb_matrix,batch_index=self.batch_index,k=K_in,flag="in",metric=K_in_metric,approx=True,return_distance=False,verbose=self.verbose,log=self.log)
            knn_intra_batch=np.array([list(i)for i in knn_intra_batch_approx])
            if(self.verbose):
                self.log.info("appoximate calculate MNN Pair inter batch...")
            mnn_inter_batch_approx=get_dict_mnn(data_matrix=self.emb_matrix,batch_index=self.batch_index,k=K_bw,flag="out",metric=K_bw_metric,approx=True,return_distance=False,verbose=self.verbose,log=self.log)
            mnn_inter_batch=np.array([list(i)for i in mnn_inter_batch_approx])
            if(self.verbose):
                self.log.info("Find All Nearest Neighbours Done....")   
        else:
            if(self.verbose):
                self.log.info("calculate KNN and MNN pair in parallel mode to accelerate!")
                self.log.info("appoximate calculate KNN Pair intra batch...")
            knn_intra_batch_approx=get_dict_mnn_para(data_matrix=self.emb_matrix,batch_index=self.batch_index,k=K_in,flag="in",metric=K_in_metric,approx=True,return_distance=False,verbose=self.verbose,log=self.log)
            knn_intra_batch=np.array(knn_intra_batch_approx)
            if(self.verbose):
                self.log.info("appoximate calculate MNN Pair inter batch...")
            mnn_inter_batch_approx=get_dict_mnn_para(data_matrix=self.emb_matrix,batch_index=self.batch_index,k=K_bw,flag="out",metric=K_bw_metric,approx=True,return_distance=False,verbose=self.verbose,log=self.log)
            mnn_inter_batch=np.array(mnn_inter_batch_approx)
            if(self.verbose):
                self.log.info("Find All Nearest Neighbours Done....")   

        if(self.verbose):
            self.log.info("calculate similarity matrix between cluster")
        self.cor_matrix,self.nn_matrix=cal_sim_matrix(knn_intra_batch,mnn_inter_batch,self.train_label,self.verbose,self.log)
        if(self.verbose):
            self.log.info("save cor matrix to file....")
            self.cor_matrix.to_csv(self.save_dir+"cor_matrix.csv")
            self.log.info("save nn pair matrix to file")
            self.nn_matrix.to_csv(self.save_dir+"nn_matrix.csv")
            self.log.info("Calculate Similarity Matrix Done....")

        if(self.celltype is not None):
            same_celltype=self.celltype[mnn_inter_batch[:,0]]==self.celltype[mnn_inter_batch[:,1]]
            equ_pair=sum(same_celltype)
            self.log.info("the number of mnnpair which link same celltype is {}".format(equ_pair))
            equ_ratio=sum(self.celltype[mnn_inter_batch[:,1]]==self.celltype[mnn_inter_batch[:,0]])/same_celltype.shape[0]
            self.log.info("the ratio of mnnpair which link same celltype is {}".format(equ_ratio))
            df=pd.DataFrame({"celltype_pair1":self.celltype[mnn_inter_batch[:,0]],"celltype_pair2":self.celltype[mnn_inter_batch[:,1]]})
            num_info=pd.crosstab(df["celltype_pair1"],df["celltype_pair2"],margins=True,margins_name="Total")
            ratio_info_row=pd.crosstab(df["celltype_pair1"],df["celltype_pair2"]).apply(lambda r: r/r.sum(), axis=1)
            ratio_info_col=pd.crosstab(df["celltype_pair1"],df["celltype_pair2"]).apply(lambda r: r/r.sum(), axis=0)
            num_info.to_csv(self.save_dir+"mnn_pair_num_info.csv")
            ratio_info_row.to_csv(self.save_dir+"mnn_pair_ratio_info_raw.csv")
            ratio_info_col.to_csv(self.save_dir+"mnn_pair_ratio_info_col.csv")
            self.log.info(num_info)
            self.log.info(ratio_info_row)
            self.log.info(ratio_info_col)
            #self.log.info("the number of mnnpair which link same celltype is {}".format(np.sum(num_info.values[:-1,:-1].diagonal())))
            #if(do_plot):
            #  plotNNpair(self.pca_emb.obsm["X_umap"],mnn_inter_batch,self.BATCH,flag="out",save_dir=self.save_dir)
        return knn_intra_batch,mnn_inter_batch,self.cor_matrix,self.nn_matrix
 
        
    def merge_cluster(self,ncluster_list=[3],merge_rule="rule2"):
        """
        merge small cluster to larger cluster and reassign cluster label
        Argument:
        ------------------------------------------------------------------
        - ncluster_list: 'list', you can set a list of fixed_ncluster to observe the merging producre of scDML.
        
        - merger_rule : 'str', default:"rule2", scDML implemented two type of merge rule, in mose case, two rules will generate same result.
        ------------------------------------------------------------------
        Return:
        merge_df: "pd.DataFrame", label of merge cluster with differnt number of cluster.
        """
        self.nc_list=pd.DataFrame()
        dis_cluster=[str(i) for i in ncluster_list]
        df=self.merge_df.copy()
        df["value"]=np.ones(self.train_X.shape[0])
        if(self.verbose):
            self.log.info("scDML merge cluster with "+merge_rule+"....")
        if(merge_rule=="rule1"):
            for n_cluster in ncluster_list:
                map_set=merge_rule1(self.cor_matrix.copy(),self.num_init_cluster,n_cluster=n_cluster,save_dir=self.save_dir)
                map_dict={}
                for index,item in enumerate(map_set):
                    for c in item:
                        map_dict[str(c)]=index
                self.merge_df["nc_"+str(n_cluster)]=self.merge_df["init_cluster"].map(map_dict)
                df[str(n_cluster)]=str(n_cluster)+"("+self.merge_df["nc_"+str(n_cluster)].astype(str)+")"
                if(self.verbose):
                    self.log.info("merging cluster set:"+str(map_set)) #

        if(merge_rule=="rule2"):
            for n_cluster in ncluster_list:
                map_set=merge_rule2(self.cor_matrix.copy(),self.nn_matrix.copy(),self.merge_df["init_cluster"].value_counts().values.copy(),n_cluster=n_cluster,verbose=self.verbose,log=self.log)
                map_dict={}
                for index,item in enumerate(map_set):
                    for c in item:
                        map_dict[str(c)]=index
                self.merge_df["nc_"+str(n_cluster)]=self.merge_df["init_cluster"].map(map_dict)
                df[str(n_cluster)]=str(n_cluster)+"("+self.merge_df["nc_"+str(n_cluster)].astype(str)+")"
                if(self.verbose):
                    self.log.info("merging cluster set:"+str(map_set)) #
        return df
     
    def build_net(self, in_dim=1000,out_dim=32,emb_dim=[256],projection=False,project_dim=2,use_dropout=False,
            dp_list=None,use_bn=False, actn=nn.ReLU(),seed=1029):
        """
        Build Network for scDML training
         Argument:
        ------------------------------------------------------------------
        - in_dim:default:1000, the input dimension of embedding net,it should be equal to the number of hvg of adata
        - out_dim: default:32, the output dimension of embedding net(i.e. embedding dimension )
        - emb_dim: default:[256], the dimension of hidden layers
        - projection: default:False, construct the projection network whose embedding dimension is project_dim(2 or 3) if True
        - project_dim: default:2,the output dimension of projection network
        - use_drop:default:False,the embedding net will add DropOut Layer if use_drop=True
        - dp_list:default:None, when use_drop=True and embedding net have more than two hidden layers,you can set a list of dropout value consistent to the DropOut Layer
        - use_bn:default:False, embedding net will apply batchNormalization when use_bn=True
        - actn: default:nn.ReLU(), the activation function of embedding net
         -seed: default 1029, It's random seed for pytorch to reproduce result
        ------------------------------------------------------------------
        """
        if(in_dim != self.train_X.shape[1]):
            in_dim = self.train_X.shape[1]
        if(self.verbose):
            self.log.info("Build Embedding Net for scDML training")
        seed_torch(seed)
        self.model=EmbeddingNet(in_sz=in_dim,out_sz=out_dim,emb_szs=emb_dim,projection=projection,project_dim=project_dim,dp_list
                                =dp_list,use_bn=use_bn,actn=actn)
        if(self.verbose):
            self.log.info(self.model)
            self.log.info("Build Embedding Net Done...")
    
    def train(self,expect_num_cluster=None,merge_rule="rule2",num_epochs=50,batch_size=64,early_stop=False,patience=5,delta=50,
              metric="euclidean",margin=0.2,triplet_type="hard",device=None,save_model=False,mode="unsupervised"):
        """
        training scDML with triplet loss
        Argument:
        ------------------------------------------------------------------
        -expect_num_cluster: default None, the expected number of cluster you want to scDML to merge to. If this parameters is None,
          scDML will use merge to the number of cluster which is identified by default threshold.
          
        -merge_rule: default:"relu2", merge rule of scDML to merge cluster when expect_num_cluster is None

        -num_epochs :default 50. maximum iteration to train scDML
        
        -early_stop: embedding net will stop to train with consideration of the rule of early stop when early top is true
        
        -patience:  default 5. How long to wait after last time validation loss improved.

        -delta: default 50, Minimum change in the monitored quantity(number of hard triplet) to qualify as an improvement.
    
        -metric: default(str) "euclidean", the type of distance to be used to calculate triplet loss
        
        -margin: the hyperparmeters which is used to calculate triplet loss
        
        -triplet_type: the type of triplets will to used to be mined and optimized the triplet loss
        
        -save model: whether to save model after scDML training
        
        -mode: if mode=="unsupervised", scDML will use the result of merge rule with similarity matrix
               if mode=="supervised", scDML will use the true label of celltype to integrate dataset
        ------------------------------------------------------------------

        Return:
        ------------------------------------------------------------------
        -embedding: default:AnnData, anndata with batch effect removal after scDML training
        ------------------------------------------------------------------
        """
        if(mode=="unsupervised"):
            if(expect_num_cluster is None): 
                if(self.verbose):
                    self.log.info("expect_num_cluster is None, use eigen value gap to estimate the number of celltype......")
                cor_matrix=self.cor_matrix.copy()
                for i in range(len(cor_matrix)):
                    cor_matrix.loc[i,i]=0.0
                
                    A=cor_matrix.values/np.max(cor_matrix.values)# normalize similarity matrix to [0,1]
    
                    # enhance the similarity structure
                    norm_A=A+matrix_power(A,2) # 
                    
                    for i in range(len(A)):
                        norm_A[i,i]=0.0
                    #cor_matrix

                k, _,  _ = eigenDecomposition(norm_A,save_dir=self.save_dir)
                self.log.info(f'Optimal number of clusters {k}')
                ## dafault to select the top one
                expect_num_cluster=k[0]
            if("nc_"+str(expect_num_cluster) not in self.merge_df):
                self.log.info("scDML can't find the mering result of cluster={} ,you can run merge_cluster(ncluster_list=[{}]) function to get this".format(expect_num_cluster,expect_num_cluster))
                raise IOError
            self.train_label=self.merge_df["nc_"+str(expect_num_cluster)].values.astype(int)
        elif(mode=="supervised"):
            expect_num_cluster=self.ncluster
            self.train_label=self.merge_df["nc_"+str(expect_num_cluster)].values.astype(int)
        else:
            self.log.info("Not implemented!!!")
            raise IOError
        if os.path.isfile(os.path.join(self.save_dir,"scDML_model.pkl")):
            self.log.info("Loading trained model...")
            self.model=torch.load(os.path.join(self.save_dir,"scDML_model.pkl"))
        else:
            if(self.verbose):
                self.log.info("train scDML(expect_num_cluster={}) with Embedding Net".format(expect_num_cluster))
                self.log.info("expect_num_cluster={}".format(expect_num_cluster))
            if(device is None):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if(self.verbose):
                    if(torch.cuda.is_available()):
                        self.log.info("using GPU to train model")
                    else:
                        self.log.info("using CPU to train model")        
            train_set = torch.utils.data.TensorDataset(torch.FloatTensor(self.train_X), torch.from_numpy(self.train_label).long())
            train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0,shuffle=True)
            self.model=self.model.to(device)
            optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            if(metric=="cosine"):
                distance = distances.CosineSimilarity()# use cosine_similarity()
            elif(metric=="euclidean"):
                distance=distances.LpDistance(p=2,normalize_embeddings=False) # use euclidean distance
            else:
                self.log.info("Not implemented,to be updated")
                raise IOError
            #reducer: reduce the loss between all triplet(mean)
            reducer = reducers.ThresholdReducer(low = 0)
            #Define Loss function
            loss_func = losses.TripletMarginLoss(margin = margin, distance = distance, reducer = reducer)
            #Define miner_function
            mining_func = miners.TripletMarginMiner(margin = margin, distance = distance, type_of_triplets = triplet_type)
            if(self.verbose):        
                self.log.info("use {} distance and {} triplet to train model".format(metric,triplet_type))
            mined_epoch_triplet=np.array([])#
            if(not early_stop):
                if(self.verbose):
                    self.log.info("not use earlystopping!!!!")
                for epoch in range(1, num_epochs+1):
                    temp_epoch_loss=0
                    temp_num_triplet=0
                    self.model.train()
                    for batch_idx, (train_data, training_labels) in enumerate(train_loader):
                        train_data, training_labels = train_data.to(device), training_labels.to(device)
                        optimizer.zero_grad()
                        embeddings = self.model(train_data)
                        indices_tuple = mining_func(embeddings, training_labels)
                        loss = loss_func(embeddings, training_labels, indices_tuple)
                        temp_num_triplet=temp_num_triplet+indices_tuple[0].size(0)
                        loss.backward()
                        optimizer.step()

                    mined_epoch_triplet=np.append(mined_epoch_triplet,temp_num_triplet)
                    if(self.verbose):
                        self.log.info("epoch={},number_hard_triplet={}".format(epoch,temp_num_triplet))
            else:
                if(self.verbose):
                    self.log.info("use earlystopping!!!!")
                early_stopping = EarlyStopping(patience=patience, delta=delta,verbose=True,path=self.save_dir+"checkpoint.pt",trace_func=self.log.info)
                for epoch in range(1, num_epochs+1):
                    temp_epoch_loss=0
                    temp_num_triplet=0
                    self.model.train()
                    for batch_idx, (train_data, training_labels) in enumerate(train_loader):
                        train_data, training_labels = train_data.to(device), training_labels.to(device)
                        optimizer.zero_grad()
                        embeddings = self.model(train_data)
                        indices_tuple = mining_func(embeddings, training_labels)
                        loss = loss_func(embeddings, training_labels, indices_tuple)
                        temp_num_triplet=temp_num_triplet+indices_tuple[0].size(0)
                        loss.backward()
                        optimizer.step()
                    early_stopping(temp_num_triplet, self.model)#
        
                    if early_stopping.early_stop:
                        self.log.info("Early stopping")
                        break

                    mined_epoch_triplet=np.append(mined_epoch_triplet,temp_num_triplet)
                    if(self.verbose):
                        self.log.info("epoch={},number_hard_triplet={}".format(epoch,temp_num_triplet))
            if(self.verbose):
                self.log.info("scDML training done....")
            ##### save embedding model
            if(save_model):
                if(self.verbose):
                    self.log.info("save model....")
                torch.save(self.model.to(torch.device("cpu")),os.path.join(self.save_dir,"scDML_model.pkl"))
            self.loss=mined_epoch_triplet
        ##### generate embeding

        features=self.predict(self.train_X)
        return features


    def predict(self,X,batch_size=128):
        """
        prediction for data matrix(produce embedding)
        Argument:
        ------------------------------------------------------------------
        X: data matrix fo dataset
        batch_size: batch_size for dataloader
        ------------------------------------------------------------------
        """
        if(self.verbose):
            self.log.info("extract embedding for dataset with trained network")
        device=torch.device("cpu")    
        dataloader = DataLoader(
            torch.FloatTensor(X), batch_size=batch_size, pin_memory=False, shuffle=False
        )
        data_iterator = tqdm(dataloader, leave=False, unit="batch")
        self.model=self.model.to(device)
        with torch.no_grad():
            self.model.eval()
            features = []
            for batch in data_iterator:
                batch = batch.to(device)
                output = self.model(batch)
                features.append(
                    output.detach().cpu()
                )  # move to the CPU to prevent out of memory on the GPU
            features=torch.cat(features).cpu().numpy()
        return features

    ##### integration for scDML
    def integrate(self,adata,batch_key="BATCH",ncluster_list=[3],expect_num_cluster=None,K_in=5,K_bw=10,K_in_metric="cosine",K_bw_metric="cosine",merge_rule="rule2",num_epochs=50,
                  projection=False,early_stop=False,batch_size=64,metric="euclidean",margin=0.2,triplet_type="hard",device=None,seed=1029,out_dim=32,emb_dim=[256],save_model=False,celltype_key=None,mode="unsupervised"):
        """
        batch alignment for integration with scDML
        Argument:
        ------------------------------------------------------------------
        adata: normalized adata
        celltype_key: evaluate the ratio of mnn pair or used in supervised mode
        mode : default str,"unsupervised", user can choose  "unsupervised" or "supervised"
        ....
        ....
        ....
        ------------------------------------------------------------------
        """
        self.log.info("mode={}".format(mode))
        #start_time=time()
        if(mode=="unsupervised"):
            # covert adata to training data
            self.convertInput(adata,batch_key=batch_key,celltype_key=celltype_key,mode=mode)
            #print("convert input...cost time={}s".format(time()-start_time))  
            # calculate similarity between cluster
            self.calculate_similarity(K_in=K_in,K_bw=K_bw,K_in_metric=K_in_metric,K_bw_metric=K_bw_metric)
            #print("calculate similarity matrix done...cost time={}s".format(time()-start_time))    
            # merge cluster and reassign cluster label
            self.merge_cluster(ncluster_list=ncluster_list,merge_rule=merge_rule)
            #print("reassign cluster label done...cost time={}s".format(time()-start_time))    
            # build Embeddding Net for scDML
            self.build_net(out_dim=out_dim,emb_dim=emb_dim,projection=projection,seed=seed)
            #print("construct network done...cost time={}s".format(time()-start_time))    
            # train scDML to remove batch effect
            features=self.train(expect_num_cluster=expect_num_cluster,num_epochs=num_epochs,early_stop=early_stop,batch_size=batch_size,metric=metric,margin=margin,triplet_type=triplet_type,device=device,save_model=save_model,mode=mode)
            #print("train neural network done...cost time={}s".format(time()-start_time)) 
            # save result
        elif(mode=="supervised"):
            self.convertInput(adata,batch_key=batch_key,celltype_key=celltype_key,mode=mode)
            # build Embeddding Net for scDML
            self.build_net(out_dim=out_dim,emb_dim=emb_dim,projection=projection,seed=seed)
            #print("construct network done...cost time={}s".format(time()-start_time))    
            # train scDML to remove batch effect
            features=self.train(expect_num_cluster=expect_num_cluster,num_epochs=num_epochs,early_stop=early_stop,batch_size=batch_size,metric=metric,margin=margin,triplet_type=triplet_type,device=device,save_model=save_model,mode=mode)
            #print("train neural network done...cost time={}s".format(time()-start_time)) 
            # save result
        else:
            self.log.info("Not implemented!!!")
            raise IOError

        adata.obsm["X_emb"]=features
        adata.obs["reassign_cluster"]=self.train_label.astype(int).astype(str)
        adata.obs["reassign_cluster"]=adata.obs["reassign_cluster"].astype("category")


        

        
    

    
    
