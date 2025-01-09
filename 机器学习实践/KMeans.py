from sklearn.discriminant_analysis import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
from dlframe import Logger
from sklearn import datasets

class KMeansClustering:
    def __init__(self, k=3, max_iters=100):
        self.logger = Logger.get_logger("KMneansModel")
        self.k = k
        self.max_iters = max_iters
        self.centroids = []
        self.cluster_centers_ = None
        self.clusters = []

    def _initialize_centroids(self, X):
        # KMeans++ initialization
        self.centroids = [X[np.random.choice(range(len(X)))]]
        for _ in range(1, self.k):
            distances = np.array([min([np.inner(x - centroid, x - centroid) for centroid in self.centroids]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    self.centroids.append(X[j])
                    break

    def fit(self, X):
        self._initialize_centroids(X)
        for _ in range(self.max_iters):
            clusters = {i: [] for i in range(self.k)}
            for x in X:
                distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(x)
            
            new_centroids = []
            for cluster in clusters.values():
                if cluster:
                    new_centroid = np.mean(cluster, axis=0)
                    new_centroids.append(new_centroid)
                else:
                    # 如果某个聚类中没有点，则重新初始化聚类中心
                    new_centroids.append(X[np.random.choice(range(len(X)))])
            self.centroids = new_centroids
            self.cluster_centers_ = self.centroids
            self.logger.print("new logger!!")

            # 如果聚类中心不再变化，则提前结束
            if np.all(self.centroids == new_centroids):
                break

    def predict(self, X):
        clusters = {i: [] for i in range(self.k)}
        for x in X:
            distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(x)
        self.clusters = list(clusters.values())
        return self.clusters

    def load_data(self):
        # 加载鸢尾花数据集
        iris = datasets.load_iris()
        X = iris.data
        return X

    def standardize_data(self, X):
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    
    def plot_data(self, X, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], s=50)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.show()

    def plot_clusters(self, X, clusters, centroids):
        plt.figure(figsize=(8, 6))
        for i, cluster in enumerate(clusters):
            plt.scatter([x[0] for x in cluster], [x[1] for x in cluster], s=50, label=f'Cluster {i}')
        plt.scatter([c[0] for c in centroids], [c[1] for c in centroids], s=200, c='red', label='Centroids')
        plt.title('KMeans Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def run_kmeans(self):
        # 加载数据
        X = self.load_data()
        # 标准化数据
        X_scaled = self.standardize_data(X)
        # 绘制原始数据
        self.plot_data(X_scaled, 'Original Data')
        # 训练模型
        self.fit(X_scaled)
        # 预测聚类
        self.predict(X_scaled)
        # 打印聚类结果
        for i, cluster in enumerate(self.clusters):
            self.logger.print(f"Cluster {i}: {len(cluster)} points")
            print(f"Cluster {i}: {len(cluster)} points")
        # 绘制聚类结果
        self.plot_clusters(X_scaled, self.clusters, self.centroids)
        
def K_Means(x):

    kmeans = KMeansClustering(k=3)
    kmeans.run_kmeans()
