from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from dataset import MyDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import Autoencoder
import pandas as pd
import os


def get_subdirectories(folder_path: str):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories


# 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(device)

# 加载数据位置
root_dir = 'contact_626_diagvector3'

model_path = '75.pth'

# 加载数据集

test_dataset = MyDataset(root_dir=root_dir, is_shuffle=False, is_mask=False,
                         random_mask=False, update_mask=False, is_train=False, mask_rate=0.1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

data_size = test_dataset.datasize

# 创建模型实例
model = Autoencoder(data_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

label_dirs = get_subdirectories(root_dir)
str2dig = {}
x = []
y = []

for i, label_name in enumerate(label_dirs):
    str2dig[label_name] = i

print(str2dig)

total = 0.0
sum_similarity = 0.0
cycles = 55
with torch.no_grad():
    for i, test_data in enumerate(test_loader):

        file_name = list(test_dataset.datas.keys())[i]

        datas, flag = test_data
        if isinstance(datas, list):
            # 此时datas是由[original_datas,masked_datas]组成
            original_datas = datas[0]
            masked_datas = datas[1]
        else:
            original_datas = datas
            masked_datas = datas

        original_datas = original_datas.view(original_datas.size(0), -1).to(device)
        masked_datas = masked_datas.view(masked_datas.size(0), -1).to(device)

        embedding = model.encoder(original_datas).to(device)

        x.append(original_datas)

        # x.append(np.copy(embedding.numpy()))
        print(file_name)
        print(flag[0])
        y.append(str2dig[flag[0]])

        i += 1
        if i > cycles:
            break

# 假设 X_train 是训练后的样本数据，y_train 是样本标签
# 这里假设 X_train 是一个二维的特征数据
# 假设我们选择将数据映射到二维空间进行观察
X_train = np.concatenate(x, axis=0)
y_train = np.array(y)

# 使用 K-means 聚类算法进行聚类
kmeans = KMeans(n_clusters=4, random_state=42)  # 假设分为3个簇
cluster_labels = kmeans.fit_predict(X_train)

# 使用 PCA 进行数据降维
pca = PCA(n_components=2)  # 将数据降至2维
X_pca = pca.fit_transform(X_train)

# 绘制散点图来观察聚类结果
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-means Clustering Results')
plt.show()

k = 0.0
for i in range(len(cluster_labels)):
    if cluster_labels[i] != y_train[i]:
        k += 1.0
print(k / len(cluster_labels))

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 计算调整兰德指数和归一化互信息
ari = adjusted_rand_score(y_train, cluster_labels)
nmi = normalized_mutual_info_score(y_train, cluster_labels)

print("Adjusted Rand Index (ARI):", ari)
print("Normalized Mutual Information (NMI):", nmi)
