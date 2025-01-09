#导入第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from dlframe import Logger



class Boosting:
    def __init__(self,value):
        self.logger = Logger.get_logger('Boosting')
        self.value = value
        pass


    def boosting(self,xX):
    #读取数据集
        wine = pd.read_csv("C:\\Users\\123\\Desktop\\合并版本\\机器学习实践\\wine_data.csv",header=None)
    #查看前五行数据
        wine.head()

    #给定列名
        wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols',
    'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
    #类别数据数据查看
        self.logger.print("class labels",np.unique(wine['Class label']))
        wine.head() #查看此时数据格式

    # 数据预处理
    # 仅仅考虑2，3类葡萄酒，去除1类
        wine = wine[wine['Class label']!= 1]
        y = wine['Class label'].values #种类标签
        X = wine[['Alcohol','OD280/OD315 of diluted wines']].values #酒精含量和稀释酒

    # 将分类标签变成二进制编码(当标签是两类的时候可以直接使用二进制编码)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        #print(y)
        #print( wine['Class label'].values)

    # 按8：2分割训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)  # stratify参数代表了按照y的类别等比例抽样

    # 使用单一决策树建模
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=1)
        from sklearn.metrics import accuracy_score
        tree = tree.fit(X_train,y_train)
        y_train_pred = tree.predict(X_train)
        y_test_pred = tree.predict(X_test)
        tree_train = accuracy_score(y_train,y_train_pred)
        tree_test = accuracy_score(y_test,y_test_pred)
        #print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train,tree_test))

    # 使用sklearn实现Adaboost(基分类器为决策树)
        '''
    AdaBoostClassifier相关参数：
    base_estimator：基本分类器，默认为DecisionTreeClassifier(max_depth=1)
    n_estimators：终止迭代的次数
    learning_rate：学习率
    algorithm：训练的相关算法，{'SAMME'，'SAMME.R'}，默认='SAMME.R'
    random_state：随机种子
        '''
        from sklearn.ensemble import AdaBoostClassifier
        ada = AdaBoostClassifier(estimator=tree,n_estimators=500,learning_rate=0.1,random_state=1)
        ada = ada.fit(X_train,y_train)
        y_train_pred = ada.predict(X_train)
        y_test_pred = ada.predict(X_test)
        ada_train = accuracy_score(y_train,y_train_pred)
        ada_test = accuracy_score(y_test,y_test_pred)
        self.logger.print('Adaboost train/test accuracies %.3f/%.3f' % (ada_train,ada_test))

# 画出单层决策树与Adaboost的决策边界：
        x_min = X_train[:, 0].min() - 1
        x_max = X_train[:, 0].max() + 1
        y_min = X_train[:, 1].min() - 1
        y_max = X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
        f, axarr = plt.subplots(nrows=1, ncols=2,sharex='col',sharey='row',figsize=(12, 6))
        for idx, clf, tt in zip([0, 1],[tree, ada],['Decision tree', 'Adaboost']):
            clf.fit(X_train, y_train)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            axarr[idx].contourf(xx, yy, Z, alpha=0.3)
            axarr[idx].scatter(X_train[y_train==0, 0],X_train[y_train==0, 1],c='blue', marker='^')
            axarr[idx].scatter(X_train[y_train==1, 0],X_train[y_train==1, 1],c='red', marker='o')
            axarr[idx].set_title(tt)
        axarr[0].set_ylabel('Alcohol', fontsize=12)
        plt.tight_layout()
        plt.text(0, -0.2,s='OD280/OD315 of diluted wines',ha='center',va='center',fontsize=12,transform=axarr[1].transAxes)
        plt.show()
def boosting_Run(x):
    B=Boosting(x)
    B.boosting(x)





