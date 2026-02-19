# DCMDA
 DCMDA is an innovative model designed for metabolite-disease association prediction, 
 DCMDA extracts deep features of both metabolites and diseases using Dual-network Cross-learning. 
 DCMDA consists of three parts. 
 The data processing module integrates similarity networks with association networks to construct a heterogeneous network. 
 The feature extraction module extracts features from the metabolite-disease association network based on the non-negative matrix factorization method and from the heterogeneous network using graph autoencoder technique.
 The feature fusion module combines the association matrix feature with the heterogeneous network feature through a Cross-Attention mechanism, thereby obtaining deep representations of metabolites and diseases.

## Novel Enhancements

This version includes several novel improvements over the base DCMDA model:

1. **Multi-Head Cross-Attention Fusion** (`classifiers.py`): Replaces single-head attention with a multi-head cross-attention mechanism (4 heads, key_dim=16) that captures diverse interaction patterns between GAE and NMF feature streams, improving robustness of feature fusion.

2. **Regularization via Dropout and Batch Normalization** (`classifiers.py`): Adds BatchNormalization after dense layers and Dropout (rate=0.3) to prevent overfitting, with EarlyStopping callback for training stability.

3. **Graph-Based Edge Dropout Augmentation** (`main.py`): Implements edge dropout (5% rate) on the heterogeneous network during training, acting as data augmentation to improve model generalization and reduce sensitivity to specific graph edges.

4. **NMF Convergence Monitoring** (`NMF.py`): Adds periodic convergence checking during NMF iterations, enabling early stopping when the objective function stabilizes, improving computational efficiency.

5. **GCN.py Bug Fix**: Fixed indentation errors where `build()` and `call()` methods were outside the `GraphConvolution` class, and added missing `activation` attribute and `K` import.

# Code
The DCMDA source code is accessible for academic use at https://github.com/ChenYanxin999/DCMDA/tree/master.

# Datasets
+ 2262 metabolites
+ 216 diseases
+ 4536 metabolite-disease pairs

* seven files:
 
> disease_information_entropy_similarity.csv --- information entropy similarity of diseases
>
> disease_GIP_similarity.csv --- gaussian kernel similarity of diseases
> 
> disease_semantic_similarity.csv --- semantic similarity of diseases 
> 
> M_D.csv --- metabolite-disease association matrix
> 
> metabolite_GIP_similarity.csv --- information entropy similarity of metabolites
> 
> metabolites_information_entropy_similarity.csv --- gaussian kernel similarity of metabolites
> 
> metabolites_structure_similarity.csv --- structural similarity of metabolites

# Prepare conda enviroment
+ conda create -n your_env_name python=3.6.13

dependencies:

+ numpy==1.19.2
+ pandas==1.1.5
+ scipy==1.5.2
+ tensorflow==2.6.2

Or install via: `pip install -r requirements.txt`

# Step-by-step running for DCMDA
### 1. Integration of similarity networks for metabolites or diseases.

Related code is located at the following location:

+ similarity_fusion.py 

The relevant parameters are as follows：
+ k1 = 226
+ k2 = 21

Then you will get:

+ m_fusion_sim
+ d_fusion_sim

### 2. Construct a heterogeneous network of metabolite-disease associations, disease-disease similarity information, and metabolite-metabolite similarity information.

Related code is located at the following location:

+ main.py

Then you will get:

+ related_matrix

### 3. Extract association matrix features through non-negative matrix factorization. 

Related code is located at the following location:

+ NMF.py 

The relevant parameters are as follows：

+ D = 90
+ lam = 0.01

Then you will get:

+ NMF_mfeature
+ NMF_dfeature 

### 4. Extract heterogeneous network features through graph autoencoders. 

Related code is located at the following location:

+ GAE_trainer.py 
+ GAE.py 
+ GCN.py 

The relevant parameters are as follows：

+ m_threshold = 0.035 
+ epochs = 100
+ lr = 0.001
+ hidden-dims = [128,64]
+ dropout = 0.5
+ gradient_clipping = 5.0

Then you will get:
+ md_features 

### 5. Employing the Cross-Attention mechanism effectively integrates association matrix features and heterogeneous network features. 

Related code is located at the following location:

+ classifiers.py

Then you will get:

+ result 


### 6. Train and test model under 5fold CV train-test scheme. 
Running:
> python main.py 

