import itertools
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from scipy import interp
from classifiers import *
from GAE_trainer import *
from GAE import *
from NMF import *
from metric import *
from similarity_fusion import *
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# 禁用 GPU
tf.config.set_visible_devices([], 'GPU')

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
#        m_fusion_sim, d_fusion_sim = get_fusion_sim(k1, k2)  # Integration of similarity networks for metabolites or diseases
#
#        np.save('datasets/m_fusion_sim.npy', m_fusion_sim)
#        np.save('datasets/d_fusion_sim.npy', d_fusion_sim)

        # print("文件已保存到指定文件夹中。")
        m_fusion_sim = pd.read_csv("datasets/m_fusion_sim2.csv", index_col=0).to_numpy()
        d_fusion_sim = pd.read_csv("datasets/d_fusion_sim2.csv", index_col=0).to_numpy()
#        m_fusion_sim = np.load('datasets/m_fusion_sim2.npy')
#        d_fusion_sim = np.load('datasets/d_fusion_sim2.npy')
        

#        print(d_fusion_sim[0])
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




#            DA = np.zeros((2262+216, 2262+216))
#            # 将矩阵 A 放入 D 的左上角
#            DA[:2262, :2262] = m_fusion_sim
#
#            # 将矩阵 B 的转置放入 D 的左下角
#            DA[2262:, :2262] = new_association.T
#
#            # 将矩阵 B 放入 D 的右上角
#            DA[:2262, 2262:] = new_association
#
#            # 将矩阵 C 放入 D 的右下角
#            DA[2262:, 2262:] = d_fusion_sim
#            # Metabolite features extraction from GAE
#            md_network = sim_thresholding(DA, 0.0035)
##            md_network=related_matrix
#            md_adj, meta_features = generate_adj_and_feature(md_network, DA)
#            print("===")
#            with tf.device('/CPU:0'):
#                md_features = get_gae_feature(md_adj, meta_features, s[1], 1)
#


# 对 m_fusion_sim 进行相似度阈值处理得到网络
            m_network = sim_thresholding(m_fusion_sim, s[0])

# 对 d_fusion_sim 进行相似度阈值处理得到网络
            d_network = sim_thresholding(d_fusion_sim, 0.0035)  # 0 1阈值矩阵

# 将 new_association 转置，用于后续计算
            d_association = new_association.T

# 创建一个全零矩阵 DA，大小为 (2262 + 216, 2262 + 216)
            DA = np.zeros((2262 + 216, 2262 + 216))

# 将 m_network 放入 DA 的左上角
            DA[:2262, :2262] = m_network

# 将 d_association 放入 DA 的左下角
            DA[2262:, :2262] = d_association

# 将 new_association 放入 DA 的右上角
            DA[:2262, 2262:] = new_association

# 将 d_network 放入 DA 的右下角
            DA[2262:, 2262:] = d_network

# 使用 np.tile 函数将 new_association 和 d_association 复制到合适的大小
#            resultm_matrix = np.tile(new_association, (1, 2262))
#            resultd_matrix = np.tile(d_association, (1, 216))
#
## 将 resultm_matrix 和 resultd_matrix 拼接成 result_matrix
#            result_matrix = np.concatenate((resultm_matrix, resultd_matrix), axis=0)

# 生成 D_adj 和 D_features
            D_adj, D_features = generate_adj_and_feature(DA, DA)

# 获取 GAE 特征
#             with tf.device('/CPU:0'):
            md_features = get_gae_feature(D_adj, DA, s[1], 1)


            

           



            # # Disease features extraction from five-layer auto-encoder
            # d_features = five_AE(d_fusion_sim)

            # get feature and label
            train_gae_feature,train_nmf_feature, train_label = generate_f1(D, train_samples, md_features, NMF_mfeature, NMF_dfeature)
            val_gae_feature,val_nmf_feature, val_label = generate_f1(D, val_samples, md_features, NMF_mfeature, NMF_dfeature)

            # MLP classfier
            model = BuildModel(train_gae_feature,train_nmf_feature, train_label)
            # with tf.device('/CPU:0'):
            test_N = val_samples.shape[0]
            y_score = np.zeros(test_N)
            y_score = model.predict([val_gae_feature,val_nmf_feature])[:, 0]

            # # calculate metrics
            # fpr, tpr, thresholds = roc_curve(val_label, y_score)
            # tprs.append(interp(mean_fpr, fpr, tpr))
            # tprs[-1][0] = 0.0
            # roc_auc = auc(fpr, tpr)
            # aucs.append(roc_auc)
            # 假设阈值为0.5, 转换预测分数为二分类结果
            y_pred = [1 if score >= 0.3 else 0 for score in y_score]
            result += get_metrics(val_label, y_score)
            print('[aupr, auc, f1_score, accuracy, recall, specificity, precision]',
                  get_metrics(val_label, y_score))
            from sklearn.metrics import roc_auc_score, average_precision_score
            from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score




            # # 计算 AUC
            # auc = roc_auc_score(val_label, y_pred)
            # print(f"AUC: {auc:.4f}")
            #
            # # 计算 AUPR
            # aupr = average_precision_score(val_label, y_pred)
            # print(f"AUPR: {aupr:.4f}")
            #
            # # 计算 F1-Score
            # f1 = f1_score(val_label, y_pred)
            # print(f"F1-Score: {f1:.4f}")
            #
            # # 计算 Accuracy
            # accuracy = accuracy_score(val_label, y_pred)
            # print(f"Accuracy: {accuracy:.4f}")
            #
            # # 计算 Recall
            # recall = recall_score(val_label, y_pred)
            # print(f"Recall: {recall:.4f}")
            #
            # # 计算 Precision
            # precision = precision_score(val_label, y_pred)
            # print(f"Precision: {precision:.4f}")
            #
            # all_auc+=auc
            # all_aupr+=aupr
            # all_f1+=f1
            # all_accuracy+=accuracy
            # all_recall+=recall
            # all_precision+=precision
        print("==================================================")
        print(result/n_splits)
        # print(all_auc/n_splits,all_aupr/n_splits,all_f1/n_splits,all_accuracy/n_splits,all_recall/n_splits,all_precision/n_splits)

       







