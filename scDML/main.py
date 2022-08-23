from .data_preprocess import *
from .calculate_NN import get_dict_mnn
from .utils import *
from .network import EmbeddingNet
import scanpy as sc 
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners,reducers,distances
from time import time
from scipy.sparse import issparse
from tqdm import tqdm

class scDMLModel:
    def __init__(self,verbose=True,save_dir="./results/"):
        """                               
        create scDMLModel object
        Argument:
        ------------------------------------------------------------------
        - verbose: 'str',optional, Default,'False', It will print some additional information when verbose=True

        - save_dir: folder to save result
        ------------------------------------------------------------------
        """     
        self.verbose=verbose
        self.save_dir=save_dir 
 
        if(os.path.exists(self.save_dir)):
            print("Saving file folder exists")

        if not os.path.exists(self.save_dir): 
            print("Create file folder to save results")
            os.makedirs(self.save_dir+"/")
        print("==========Create scDMLModel Object Done....     ===================")
        
    def preprocess(self,adata,preprocessed=False,resolution=3.0,batch_key="BATCH",n_high_var = 1000,hvg_list=None,normalize_samples = True,target_sum=1e4,log_normalize = True,
                   normalize_features = True,pca_dim=100,scale_value=10.0,cluster_method="louvain"):
        """
        Preprocessing raw dataset 
        Argument:
        ------------------------------------------------------------------
        - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.

        - preprocessed: `bool`, default, False, It indicate  whether the adata have been preprocesed(normalized).

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
                
        """
        batch_key=checkInput(adata,batch_key,preprocessed)
        self.batch_key=batch_key
        self.reso=resolution
        self.cluster_method=cluster_method
        self.method="scDML cluster_method=({})".format(cluster_method)   

        if(preprocessed):
            print("you have preprocessed the data which is suitable to scDML training")
            if(issparse(adata.X)):  
                self.train_X=adata.X.toarray()
            else:
                self.train_X=adata.X.copy()
            self.train_label=adata.obs["init_cluster"].values.copy()
            self.emb_matrix=adata.obsm["X_pca"].copy()
            self.batch_index=adata.obs[batch_key].values
            self.merge_df=pd.DataFrame(adata.obs["init_cluster"])
        else:
            print("dataset is not preprocessed,run 'preprocess()' function to preprocess")
            print("==========Running preprocess() function         ===================")
            if(batch_key is None):
                batch_key=self.batch_key

            self.norm_args = (batch_key,n_high_var,hvg_list,normalize_samples,target_sum,log_normalize, normalize_features,scale_value,self.verbose)
            normalized_adata = Normalization(adata,*self.norm_args)
            self.batch_index=normalized_adata.obs[batch_key].values
            
            emb=dimension_reduction(normalized_adata,pca_dim,self.verbose)
            init_clustering(emb,self.reso,cluster_method)
            self.emb_matrix=emb.X
            self.train_X= normalized_adata.X.copy()
            self.train_label= emb.obs["init_cluster"].values.copy()
            normalized_adata.obs["init_cluster"]=emb.obs["init_cluster"].values.copy()
            self.merge_df=pd.DataFrame(emb.obs["init_cluster"])
            self.num_init_cluster=len(emb.obs["init_cluster"].value_counts())
            self.norm_data=normalized_adata
            print("Preprocess Dataset Done...")
            return normalized_adata
            # sc.pp.neighbors(norm_data)
            # sc.tl.umap(norm_data)
            # sc.pl.umap(norm_data,color=["BATCH"])
           
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
        print("==========Calculate similarity of cluster with KNN and MNN=========")
        print("appoximate calculate KNN Pair intra batch...")
        knn_intra_batch_approx,_ =get_dict_mnn(data_matrix=self.emb_matrix,batch_index=self.batch_index,k=K_in,flag="in",metric=K_in_metric,approx=True,return_distance=False,verbose=self.verbose)
        knn_intra_batch=np.array([list(i)for i in knn_intra_batch_approx])
        
        print("appoximate calculate MNN Pair inter batch...")
        mnn_inter_batch_approx,_=get_dict_mnn(data_matrix=self.emb_matrix,batch_index=self.batch_index,k=K_bw,flag="out",metric=K_bw_metric,approx=True,return_distance=False,verbose=self.verbose)
        mnn_inter_batch=np.array([list(i)for i in mnn_inter_batch_approx])
        print("Find All Nearest Neighbours Done....")   

        print("calculate similarity matrix between cluster")
        self.cor_matrix,self.nn_matrix=cal_sim_matrix(knn_intra_batch,mnn_inter_batch,self.train_label,self.verbose)
        print("Calculate Similarity Matrix Done....")
        return knn_intra_batch,mnn_inter_batch,self.cor_matrix,self.nn_matrix
        # knn_diff_set=knn_set[norm_data.obs["init_cluster"][knn_set[:,0]].values!=norm_data.obs["init_cluster"][knn_set[:,1]].values]
        # mnn_diff_set=mnn_set[norm_data.obs["init_cluster"][mnn_set[:,0]].values!=norm_data.obs["init_cluster"][mnn_set[:,1]].values]
        
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
        print("=================scDML merge cluster with "+merge_rule+"                 ===================")
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
                    print("merging cluster set:",map_set) #

        if(merge_rule=="rule2"):
            for n_cluster in ncluster_list:
                map_set=merge_rule2(self.cor_matrix.copy(),self.nn_matrix.copy(),self.merge_df["init_cluster"].value_counts().values.copy(),n_cluster=n_cluster,verbose=self.verbose)
                map_dict={}
                for index,item in enumerate(map_set):
                    for c in item:
                        map_dict[str(c)]=index
                self.merge_df["nc_"+str(n_cluster)]=self.merge_df["init_cluster"].map(map_dict)
                df[str(n_cluster)]=str(n_cluster)+"("+self.merge_df["nc_"+str(n_cluster)].astype(str)+")"
                if(self.verbose):
                    print("merging cluster set:",map_set) #
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
            print("the input dimension of Embedding net is not equal to the features of input data")
            raise IOError
        print("=================Build Embedding Net for scDML training         ===================")
        seed_torch(seed)
        self.model=EmbeddingNet(in_sz=in_dim,out_sz=out_dim,emb_szs=emb_dim,projection=projection,project_dim=project_dim,dp_list
                                =dp_list,use_bn=use_bn,actn=actn)
        if(self.verbose):
            print(self.model)
        print("=================Build Embedding Net Done...                    ===================")
    
    def train(self,expect_num_cluster=None,merge_rule="rule2",num_epochs=50,batch_size=64,early_stop=False,
              metric="euclidean",margin=0.2,triplet_type="hard",device=None,save_model=True):
        """
        training scDML with triplet loss
        Argument:
        ------------------------------------------------------------------
        -expect_num_cluster: default None, the expected number of cluster you want to scDML to merge to. If this parameters is None,
          scDML will use merge to the number of cluster which is identified by default threshold.
          
        -merge_rule: default:"relu2", merge rule of scDML to merge cluster when expect_num_cluster is None

        -num_epochs :default 50. maximum iteration to train scDML
        
        -early_stop: embedding net will stop to train with consideration of the rule of early stop when early top is true
        
        -metric: default(str) "euclidean", the type of distance to be used to calculate triplet loss
        
        -margin: the hyperparmeters which is used to calculate triplet loss
        
        -triplet_type: the type of triplets will to used to be mined and optimized the triplet loss
        
        -do_umap: default True, do umap visulization for embedding
        ------------------------------------------------------------------

        Return:
        ------------------------------------------------------------------
        -embedding: default:AnnData, anndata with batch effect removal after scDML training
        ------------------------------------------------------------------
        """
        if(expect_num_cluster is None): 
            print("expect_num_cluster is None, use default threshold...")
            threshold=self.nn_matrix.to_numpy().sum()/(self.K_bw+self.K_in)/self.train_X.shape[0]
            if(merge_rule == "rule2"):
                map_set=merge_rule2(self.cor_matrix.copy(),self.nn_matrix.copy(),self.merge_df["init_cluster"].value_counts().values.copy(),verbose=self.verbose,threshold=threshold)
                expect_num_cluster=len(map_set)
                map_dict={}
                for index,item in enumerate(map_set):
                    for c in item:
                        map_dict[str(c)]=index
                self.merge_df["nc_"+str(expect_num_cluster)]=self.merge_df["init_cluster"].map(map_dict)
            elif(merge_rule == "rule1"):
                map_set=merge_rule1(self.cor_matrix.copy(),self.num_init_cluster,threshold=threshold)
                expect_num_cluster=len(map_set)
                map_dict={}
                for index,item in enumerate(map_set):
                    for c in item:
                        map_dict[str(c)]=index
                self.merge_df["nc_"+str(expect_num_cluster)]=self.merge_df["init_cluster"].map(map_dict)
        
        if("nc_"+str(expect_num_cluster) not in self.merge_df):
            print("scDML can't find the mering result of cluster={} ,you can run merge_cluster(fixed_ncluster={}) function to get this".format(expect_num_cluster,expect_num_cluster))
            raise IOError
        self.train_label=self.merge_df["nc_"+str(expect_num_cluster)].values.astype(int)

        if os.path.isfile(os.path.join(self.save_dir,"scDML_model.pkl")):
            print("Loading trained model...")
            self.model=torch.load(os.path.join(self.save_dir,"scDML_model.pkl"))
        else:
            print("=================train scDML(expect_num_cluster={}) with Embedding Net==============".format(expect_num_cluster))
            if(device is None):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if(self.verbose):
                    if(torch.cuda.is_available()):
                        print("using GPU to train model")
                    else:
                        print("using CPU to train model")
                    
            train_set = torch.utils.data.TensorDataset(torch.FloatTensor(self.train_X), torch.from_numpy(self.train_label).long())
            train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0,shuffle=True)
            self.model=self.model.to(device)
            optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            if(metric=="cosine"):
                distance = distances.CosineSimilarity()# use cosine_similarity()
            elif(metric=="euclidean"):
                distance=distances.LpDistance(p=2,normalize_embeddings=False) # use euclidean distance
            else:
                pass
            reducer = reducers.ThresholdReducer(low = 0)#reducer: reduce the loss between all triplet(mean)
            #Define Loss function
            loss_func = losses.TripletMarginLoss(margin = margin, distance = distance, reducer = reducer)
            #Define miner_function
            mining_func = miners.TripletMarginMiner(margin = margin, distance = distance, type_of_triplets = triplet_type)

            if(self.verbose):        
                print("use {} distance and {} triplet to train model".format(metric,triplet_type))
            train_epoch_loss=np.array([])#
            mined_epoch_triplet=np.array([])#
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
                    temp_epoch_loss=temp_epoch_loss+loss.item()
                    temp_num_triplet=temp_num_triplet+indices_tuple[0].size(0)
                    loss.backward()
                    optimizer.step()
                # if epoch % 100==0:
                #     with torch.no_grad():
                #         self.model.eval()
                #         train_embeddings = self.model(torch.FloatTensor(self.train_X).to(device)).cpu().numpy()
                #         train_labels=self.train_label.astype(int)
                #         visulize_encode(train_embeddings,train_labels,self.celltype,self.BATCH,epoch,"../figures/",False,"Full")
                mined_epoch_triplet=np.append(mined_epoch_triplet,temp_num_triplet)
                train_epoch_loss=np.append(train_epoch_loss,temp_epoch_loss) 
                if(self.verbose):
                    print("epoch={},triplet_loss={},number_hard_triplet={}".format(epoch,temp_epoch_loss,temp_num_triplet))

            print("=================scDML training done...               ==============================")
            if(self.verbose):
                print("plot number of mined triplets in all epochs")
                plt.figure(figsize=(6,4))
                plt.plot(range(1,len(mined_epoch_triplet)+1),mined_epoch_triplet,c="r")
                plt.title("scDML loss(epoch)")
                plt.xlabel("epoch")
                plt.ylabel("mined hard triplets")
                plt.savefig(self.save_dir+"/mined_num_triplet_epoch.png")
                plt.show()
            
            ##### save embedding model
            if(save_model):
                torch.save(self.model.to(torch.device("cpu")),os.path.join(self.save_dir,"scDML_model.pkl"))
        ##### generate embeding
        features=self.predict(self.train_X)
        embedding=sc.AnnData(features)
        embedding.obsm["X_emb"]=features
        embedding.obs=self.norm_data.obs
        embedding.obs["reassign_cluster"]=self.train_label.astype(int).astype(str)
        embedding.obs["reassign_cluster"]=embedding.obs["reassign_cluster"].astype("category")
        return embedding
        # if(do_umap):
        #     print("calulate umap visulization for scDML embedding...")
        #     sc.pp.neighbors(embedding,random_state=0)
        #     sc.tl.umap(embedding)

    def predict(self,X):
        """
        prediction for data matrix(produce embedding)
        Argument:
        ------------------------------------------------------------------
        X: data matrix fo dataset
        ------------------------------------------------------------------
        """
        if(self.verbose):
            print("============do prediction for dataset ==============")
        device=torch.device("cpu")    
        dataloader = DataLoader(
            torch.FloatTensor(X), batch_size=128, pin_memory=False, shuffle=False
        )
        data_iterator = tqdm(dataloader, leave=False, unit="batch")
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

    ##### integrate all step into one function #####
    def full_run(self,adata,resolution=3.0,ncluster_list=[3],expect_num_cluster=None,batch_key="BATCH",celltype_key="celltype",K_in=5,K_bw=10,cluster_method="louvain",merge_rule="rule2"):
        # preprocess dataset
        #start_time=time()
        self.preprocess(adata=adata,resolution=resolution,cluster_method=cluster_method)
        #print("preprocess done...cost time={}s".format(time()-start_time))    
        # calculate similarity between cluster
        self.calculate_similarity(K_in=K_in,K_bw=K_bw)
        #print("calculate similarity matrix done...cost time={}s".format(time()-start_time))    
        # merge cluster and reassign cluster label
        self.merge_cluster(ncluster_list=ncluster_list,merge_rule=merge_rule)
        #print("reassign cluster label done...cost time={}s".format(time()-start_time))    
        # build Embeddding Net for scDML
        self.build_net()
        #print("construct network done...cost time={}s".format(time()-start_time))    
        # train scDML to remove batch effect
        embedding=self.train(expect_num_cluster=expect_num_cluster)
        #print("train neural network done...cost time={}s".format(time()-start_time)) 
        # evaluate scDML correction result
        #scdml_eva=self.evaluate(ncelltype=3)
        #print("all done")
        return embedding

        

        
    

    
    