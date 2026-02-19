import itertools
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from numpy import interp
from classifiers import *
from GAE_trainer import *
from GAE import *
from NMF import *
from metric import *
from similarity_fusion import *
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# Disable GPU
tf.config.set_visible_devices([], 'GPU')


# Novelty: Graph-based edge dropout augmentation for training robustness
def edge_dropout(adjacency_matrix, drop_rate=0.05):
    """Randomly drops edges from the adjacency matrix for augmentation.

    Args:
        adjacency_matrix: Binary adjacency matrix.
        drop_rate: Fraction of edges to drop (default 0.05).

    Returns:
        Augmented adjacency matrix with some edges removed.
    """
    aug = adjacency_matrix.copy()
    edge_indices = np.argwhere(aug > 0)
    n_drop = int(len(edge_indices) * drop_rate)
    if n_drop > 0:
        drop_indices = edge_indices[
            np.random.choice(len(edge_indices), n_drop, replace=False)
        ]
        for idx in drop_indices:
            aug[idx[0], idx[1]] = 0
    return aug


if __name__ == '__main__':
    # parameter
    n_splits = 5
    classifier_epochs = 50
    m_threshold = [0.035]
    epochs=[100]
    fold = 0
    result = np.zeros((1, 7), float)
    tprs=[]
    aucs=[]
    mean_fpr=np.linspace(0,1,100)
    all_auc=0
    all_aupr =0
    all_f1 =0
    all_accuracy =0
    all_recall =0
    all_precision =0
    for s in itertools.product(m_threshold,epochs):

            association = pd.read_csv("datasets/M_D.csv", index_col=0).to_numpy()
            samples = get_all_samples(association)


            k1 = 226
            k2 = 21

            m_fusion_sim = pd.read_csv("datasets/m_fusion_sim2.csv", index_col=0).to_numpy()
            d_fusion_sim = pd.read_csv("datasets/d_fusion_sim2.csv", index_col=0).to_numpy()

            kf = KFold(n_splits=n_splits, shuffle=True)

            # Metabolite and disease features extraction from NMF
            D = 90
            NMF_mfeature, NMF_dfeature = get_low_feature(D, 0.001, pow(10, -4), association)

            for train_index, val_index in kf.split(samples):
                fold += 1
                train_samples = samples[train_index, :]
                val_samples = samples[val_index, :]
                new_association = association.copy()
                for i in val_samples:
                    new_association[i[0], i[1]] = 0

                # Similarity thresholding for metabolite and disease networks
                m_network = sim_thresholding(m_fusion_sim, s[0])
                d_network = sim_thresholding(d_fusion_sim, 0.0035)
                d_association = new_association.T

                # Construct heterogeneous network
                DA = np.zeros((2262 + 216, 2262 + 216))
                DA[:2262, :2262] = m_network
                DA[2262:, :2262] = d_association
                DA[:2262, 2262:] = new_association
                DA[2262:, 2262:] = d_network

                # Novelty: Apply edge dropout augmentation to heterogeneous network
                DA = edge_dropout(DA, drop_rate=0.05)

                D_adj, D_features = generate_adj_and_feature(DA, DA)

                # Extract GAE features
                md_features = get_gae_feature(D_adj, DA, s[1], 1)

                # Get feature and label
                train_gae_feature,train_nmf_feature, train_label = generate_f1(D, train_samples, md_features, NMF_mfeature, NMF_dfeature)
                val_gae_feature,val_nmf_feature, val_label = generate_f1(D, val_samples, md_features, NMF_mfeature, NMF_dfeature)

                # MLP classifier with multi-head cross-attention
                model = BuildModel(train_gae_feature,train_nmf_feature, train_label)
                test_N = val_samples.shape[0]
                y_score = np.zeros(test_N)
                y_score = model.predict([val_gae_feature,val_nmf_feature])[:, 0]

                y_pred = [1 if score >= 0.3 else 0 for score in y_score]
                result += get_metrics(val_label, y_score)
                print('[aupr, auc, f1_score, accuracy, recall, specificity, precision]',
                      get_metrics(val_label, y_score))
                from sklearn.metrics import roc_auc_score, average_precision_score
                from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

            print("==================================================")
            print(result/n_splits)







