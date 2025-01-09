import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from dlframe import  Logger


# 数据集
class BreastCancerDataset:
    def __init__(self, data=None, target=None):
        self.logger = Logger.get_logger('BreastCancerDataset')
        if data is None and target is None:
            self.data, self.target = load_breast_cancer(return_X_y=True)
        else:
            self.data = data
            self.target = target
        self.logger.print("Breast Cancer dataset loaded")
    def test(self, testDataset):
        probabilities = self.model.predict_proba(testDataset.data)
        predictions = self.model.predict(testDataset.data)
        return probabilities, predictions
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.target[idx]

# 数据集切分器
class DatasetSplitter:
    def __init__(self, ratio):
        self.ratio = ratio
        self.logger = Logger.get_logger('DatasetSplitter')

    def split(self, dataset):
        train_data, test_data, train_target, test_target = train_test_split(
            dataset.data, dataset.target, test_size=1-self.ratio, random_state=42
        )
        return BreastCancerDataset(train_data, train_target), BreastCancerDataset(test_data, test_target)

# 模型
class KNNModel:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.logger = Logger.get_logger('KNNModel')
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, trainDataset):
        self.model.fit(trainDataset.data, trainDataset.target)
        self.logger.print("Training completed")

    def test(self, testDataset):
        predictions = self.model.predict(testDataset.data)
        return predictions

# 结果判别器
class ResultJudger:
    def __init__(self):
        self.logger = Logger.get_logger('ResultJudger')

    def judge(self,y_hat, test_dataset):
        y_true = test_dataset.target
        self.logger.print("Predictions: {}".format(y_hat))
        self.logger.print("Ground truth: {}".format(test_dataset.target))
        # 计算评价指标
        accuracy = accuracy_score(y_true, y_hat)
        precision = precision_score(y_true, y_hat)
        recall = recall_score(y_true, y_hat)
        f1 = f1_score(y_true, y_hat)
        roc_auc = roc_auc_score(y_true, y_hat)
        fpr, tpr, thresholds = roc_curve(y_true, y_hat)
        conf_matrix = confusion_matrix(y_true, y_hat)
        # 打印评价指标
        self.logger.print(f"Accuracy: {accuracy:.2f}")
        self.logger.print(f"Precision: {precision:.2f}")
        self.logger.print(f"Recall: {recall:.2f}")
        self.logger.print(f"F1 Score: {f1:.2f}")
        self.logger.print(f"AUC: {roc_auc:.2f}")
        self.logger.print("Confusion Matrix:")
        self.logger.print(conf_matrix)

         # 绘制ROC曲线和AUC值
        self.plot_roc_curve(y_true, y_hat)

        # 绘制混淆矩阵
        self.plot_confusion_matrix(y_true, y_hat)
        # 绘制混淆矩阵
        self.plot_confusion_matrix(y_true, y_hat)

    def plot_roc_curve(self, y_true, y_hat):
        fpr, tpr, _ = roc_curve(y_true, y_hat)
        roc_auc = roc_auc_score(y_true, y_hat)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self, y_true, y_hat):
        conf_matrix = confusion_matrix(y_true, y_hat)
        plt.figure()
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_true)))
        plt.xticks(tick_marks, np.unique(y_true), rotation=45)
        plt.yticks(tick_marks, np.unique(y_true))
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

def Knearly(x):
    dataset = BreastCancerDataset()
    
    # 分割数据集
    splitter = DatasetSplitter(ratio=0.2)
    train_dataset, test_dataset = splitter.split(dataset)
    
    # 训练模型
    model = KNNModel(n_neighbors=5)
    model.train(train_dataset)
    
    # 测试模型
    predictions = model.test(test_dataset)
    
    # 判断结果
    judger = ResultJudger()
    judger.judge(predictions, test_dataset)  # 确保传递了预测结果和测试数据集

