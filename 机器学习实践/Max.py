import math
from copy import deepcopy
from dlframe import Logger

class MaxEntropy:
    def __init__(self, EPS=0.005):
        self.logger = Logger.get_logger('MaxEntropy')
        self._samples = []
        self._label_y = set()  # 标签集合，相当去去重后的y
        self._numXY = {}  # key为(x,y)，value为出现次数
        self._samples_num = 0  # 样本数
        self._Ep_ = []  # 样本分布的特征期望值
        self._xyID = {}  # key记录(x,y),value记录id号
        self._xy_num = 0  # 特征键值(x,y)的个数
        self._max_feature_num = 0  # 最大特征数
        self._IDxy = {}  # key为(x,y)，value为对应的id号
        self._weights = []
        self._EPS = EPS  # 收敛条件
        self._last_weights = []  # 上一次w参数值

    def loadData(self):
        
        self._samples = deepcopy(dataset)
        for items in self._samples:
            y = items[0]
            X = items[1:]
            self._label_y.add(y)  # 集合中y若已存在则会自动忽略
            for x in X:
                if (x, y) in self._numXY:
                    self._numXY[(x, y)] += 1
                else:
                    self._numXY[(x, y)] = 1

        self._samples_num = len(self._samples)
        self._xy_num = len(self._numXY)
        self._max_feature_num = max([len(sample) - 1 for sample in self._samples])
        self._weights = [0] * self._xy_num
        self._last_weights = self._weights[:]

        self._Ep_ = [0] * self._xy_num
        for i, xy in enumerate(self._numXY):  # 计算特征函数fi关于经验分布的期望
            self._Ep_[i] = self._numXY[xy] / self._samples_num
            self._xyID[xy] = i
            self._IDxy[i] = xy
    
    # 计算每个Z(x)值
    def _calc_zx(self, X):
        zx = 0
        for y in self._label_y:
            temp = 0
            for x in X:
                if (x, y) in self._numXY:
                    temp += self._weights[self._xyID[(x, y)]]
            zx += math.exp(temp)
        return zx
    
    # 计算每个P(y|x)
    def _calu_model_pyx(self, y, X):  
        zx = self._calc_zx(X)
        temp = 0
        for x in X:
            if (x, y) in self._numXY:
                temp += self._weights[self._xyID[(x, y)]]
        pyx = math.exp(temp) / zx
        return pyx
    
    # 计算特征函数fi关于模型的期望
    def _calc_model_ep(self, index):  
        x, y = self._IDxy[index]
        ep = 0
        for sample in self._samples:
            if x not in sample:
                continue
            pyx = self._calu_model_pyx(y, sample)
            ep += pyx / self._samples_num
        return ep
    
     # 判断是否全部收敛
    def _convergence(self): 
        for last, now in zip(self._last_weights, self._weights):
            if abs(last - now) >= self._EPS:
                return False
        return True
     
    #计算预测概率
    def predict(self, X):  
        Z = self._calc_zx(X)
        result = {}
        for y in self._label_y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._weights[self._xyID[(x, y)]]
            pyx = math.exp(ss) / Z
            result[y] = pyx
        return result
    
    #训练
    def train(self, maxiter=1000): 
        for loop in range(maxiter):
            if loop % 10 == 0:
                self.logger.print("迭代次数:%d" % loop)
            #print("迭代次数:%d" % loop)
            self._last_weights = self._weights[:]
            for i in range(self._xy_num):
                ep = self._calc_model_ep(i)  # 计算第i个特征的模型期望
                self._weights[i] += math.log(self._Ep_[i] / ep) / self._max_feature_num  # 更新参数
            if loop % 10 == 0:
                self.logger.print("权值:", self._weights)
            #print("权值:", self._weights)
            if self._convergence():  # 判断是否收敛
                break
    
    def Bool(self,predict):
        self.logger.print("预测结果：",predict)

dataset = [['no', 'sunny', 'hot', 'high', 'FALSE'],
           ['no', 'sunny', 'hot', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'high', 'FALSE'],
           ['yes', 'rainy', 'mild', 'high', 'FALSE'],
           ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
           ['no', 'rainy', 'cool', 'normal', 'TRUE'],
           ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
           ['no', 'sunny', 'mild', 'high', 'FALSE'],
           ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
           ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
           ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
           ['yes', 'overcast', 'mild', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
           ['no', 'rainy', 'mild', 'high', 'TRUE']]#

def maxL(y):
    MaxEner= MaxEntropy() 
    x = ['overcast', 'mild', 'high', 'FALSE']
    MaxEner.loadData()
    MaxEner.train(maxiter=100)
    MaxEner.Bool(MaxEner.predict(x))
    #print('预测结果:', MaxEner.predict(x))

