from dlframe import Logger
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# SVM模型
class SVMModel:
    def __init__(self, kernel='linear', C=1.0) -> None:
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.logger = Logger.get_logger('SVMModel')
        self.model = SVC(kernel=self.kernel, C=self.C)

    def train(self, trainDataset) -> None:
        X_train = [trainDataset[i][0] for i in range(len(trainDataset))]
        y_train = [trainDataset[i][1] for i in range(len(trainDataset))]
        self.logger.print("Training SVM with kernel = {}, C = {}...".format(self.kernel, self.C))
        self.model.fit(X_train, y_train)

    def test(self, testDataset):
        X_test = [testDataset[i][0] for i in range(len(testDataset))]
        y_pred = self.model.predict(X_test)
        self.logger.print("Testing complete.")
        return y_pred


# 结果判别器
class SVMJudger:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('SVMJudger')

    def judge(self, y_hat, test_dataset) -> None:
        y_true = [test_dataset[i][1] for i in range(len(test_dataset))]
        accuracy = np.mean(np.array(y_hat) == np.array(y_true))
        self.logger.print("Predicted labels: {}".format(y_hat))
        self.logger.print("True labels: {}".format(y_true))
        self.logger.print("Accuracy: {:.2f}".format(accuracy))

def SVM(x):
    data = IrisDataset()
    splititer = IrisSplitter(0.3)
    train_data_test_data = splititer.split(data)
    train_data, test_data = train_data_test_data[0], train_data_test_data[1]
    model = SVMModel(kernel='linear', C=1.0)
    model.train(train_data)
    y_hat = model.test(test_data)
    judger =  SVMJudger()
    judger.judge(y_hat, test_data)