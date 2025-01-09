from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
class LogisticRegressionModel:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.logger = Logger.get_logger('LogisticRegressionModel')
        self.model = LogisticRegression(solver='liblinear')

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

    def judge(self, y_hat, test_dataset):
        self.logger.print("Predictions: {}".format(y_hat))
        self.logger.print("Ground truth: {}".format(test_dataset.target))
        y_true =   test_dataset.target
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
def logistic(x):
    # 加载数据集
    dataset = BreastCancerDataset()
    
    # 分割数据集
    splitter = DatasetSplitter(ratio=0.2)
    train_dataset, test_dataset = splitter.split(dataset)
    
    # 训练模型
    model = LogisticRegressionModel(learning_rate=0.01)
    model.train(train_dataset)
    
    # 测试模型
    predictions = model.test(test_dataset)
    
    # 判断结果
    judger = ResultJudger()
    judger.judge(predictions, test_dataset)

        