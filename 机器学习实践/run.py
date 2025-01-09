
from dlframe import WebManager
from DECISION_TREE import decision
from HMMModel import HMMRun
from SVMModel import SVM
from EMModel import EM
from NaiveBayesModel import NativeBayes
from Max import maxL
from Boosting import boosting_Run
from KMeans import K_Means
from Knearly import Knearly
from logistic import logistic
with WebManager() as manager:
    # 定义元素可选的 python 对象
    
    manager.register('Selection_Algorithm', '决策树',decision)
    manager.register('Selection_Algorithm', 'EMModel',EM)
    manager.register('Selection_Algorithm', 'NativeBayesModel',NativeBayes)
    manager.register('Selection_Algorithm', 'SVMModel',SVM)
    manager.register('Selection_Algorithm', 'Max',maxL)
    manager.register('Selection_Algorithm', 'Boosting',boosting_Run)
    manager.register('Selection_Algorithm', 'KMeans',K_Means)
    manager.register('Selection_Algorithm', 'Knearly',Knearly)
    manager.register('Selection_Algorithm', 'Logistic',logistic)
    manager.register('Selection_Algorithm','隐马尔可夫模型',HMMRun)
    #func是作为函数的输入
    # 定义元素
    func = manager['Selection_Algorithm']

    # 定义框架执行逻辑
    func(1)

