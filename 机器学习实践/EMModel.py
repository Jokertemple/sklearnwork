from math import log
import operator
from dlframe import WebManager, Logger
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
from regex import splititer
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import numpy as np




# 鸢尾花数据集加载
class IrisDataset:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('IrisDataset')
        self.logger.print("Loading Iris dataset...")
        iris = datasets.load_iris()  # 改数据集位置
        self.data = iris.data
        self.target = iris.target

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.target[idx]


class TrainTestDataset:
    def __init__(self, data, target) -> None:
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.target[idx]

 
# 数据集切分器
class IrisSplitter:
    def __init__(self, test_size) -> None:
        super().__init__()
        self.test_size = test_size
        self.logger = Logger.get_logger('IrisSplitter')
        self.logger.print("Test size: {}".format(self.test_size))

    def split(self, dataset):
        # data = [dataset[i][0] for i in range(len(dataset))]
        # target = [dataset[i][1] for i in range(len(dataset))]
        x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=self.test_size, random_state=42)

        trainingSet = TrainTestDataset(x_train, y_train)
        testingSet = TrainTestDataset(x_test, y_test)

        self.logger.print("Split complete!")
        self.logger.print("Training set size: {}".format(len(trainingSet)))
        self.logger.print("Testing set size: {}".format(len(testingSet)))
        return trainingSet, testingSet
    


# EM 模型
class EMModel:
    def __init__(self, n_components=3) -> None:
        super().__init__()
        self.n_components = n_components
        self.logger = Logger.get_logger('EMModel')
        self.name = 'EMModel'
        self.model = GaussianMixture(n_components=self.n_components, random_state=42)

    def train(self, trainDataset) -> None:
        X_train = [trainDataset[i][0] for i in range(len(trainDataset))]
        self.logger.print("Training EM with n_components = {}...".format(self.n_components))
        self.model.fit(X_train)

    def test(self, testDataset):
        X_test = [testDataset[i][0] for i in range(len(testDataset))]
        y_pred = self.model.predict(X_test)
        self.logger.print("Testing complete.")
        # 转换为NumPy数组
        X_test = np.array(X_test)

        # 绘图
        plt.figure(figsize=(10, 6))

        # 绘制数据点，按预测的类别着色
        for component_id in range(self.n_components):
            cluster_points = X_test[y_pred == component_id]
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                label=f"Component {component_id}",
                s=50
            )

        # 绘制高斯分布的均值和协方差椭圆
        means = self.model.means_
        covariances = self.model.covariances_
        for i in range(self.n_components):
            mean = means[i]
            covariance = covariances[i]

            # 计算协方差椭圆的特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

            # 提取前两个特征值
            width, height = 2 * np.sqrt(eigenvalues[:2])

            # 计算椭圆旋转角
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            # 绘制椭圆
            ellipse = Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=angle,
                edgecolor='red',
                facecolor='none'
            )
            plt.gca().add_patch(ellipse)

        # 图形细节
        plt.scatter(means[:, 0], means[:, 1], c='red', s=200, marker='x', label="Means")
        plt.title("Gaussian Mixture Model Results")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True)
        plt.show()

        return y_pred



# 结果判别器
class EMJudger:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('EMJudger')

    def judge(self, y_hat, test_dataset) -> None:
        y_true = [test_dataset[i][1] for i in range(len(test_dataset))]
        accuracy = np.mean(np.array(y_hat) == np.array(y_true))
        self.logger.print("Predicted labels: {}".format(y_hat))
        self.logger.print("True labels: {}".format(y_true))
        self.logger.print("Accuracy: {:.2f}".format(accuracy))

def EM(x):
    data = IrisDataset()
    splititer = IrisSplitter(0.3)
    train_data_test_data = splititer.split(data)
    train_data, test_data = train_data_test_data[0], train_data_test_data[1]
    model = EMModel(n_components=3)
    model.train(train_data)
    y_hat = model.test(test_data)
    judger =  EMJudger()
    judger.judge(y_hat, test_data)
