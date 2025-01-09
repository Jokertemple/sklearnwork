import json
import jieba
import re
from dlframe import Logger

text = "道法自然，无为而治。心静如水，观世间万物之变。顺应天时，不争不抢，以柔克刚。修炼内心，清净无为，方得长生久视之道。道家精髓，在于悟道，悟透生死，超脱世俗。"

jieba_result = jieba.lcut(text)

def load_dict(file_path):
    """
    从文件中加载单词到字典。
    
    :file_path: 字典文件的路径
    :return: 包含所有单词的集合
    """
    word_dict = set()
    with open('机器学习实践\dict_cn.txt', 'r', encoding='gbk') as file:
        for line in file:
            word = line.strip()
            word = re.sub(r'[^\u4e00-\u9fff]+', '', line)#替换频率及词性为''（空）
            if word:  # 确保行不为空
                word_dict.add(word)

    return word_dict
dict_path = '机器学习实践\dict_cn.txt'
dictionary = load_dict(dict_path)
sentence = text


class HMM:
    def __init__(self, states_matrix, observation_matrix, init_states, states):
        self.logger = Logger.get_logger('HMMModel')
        self.states_matrix = states_matrix  # 状态转移概率矩阵
        self.observation_matrix = observation_matrix  # 观测状态概率矩阵
        self.init_states = init_states  # 初始状态概率分布
        self.states = states  # 状态集合
    def viterbi(self, sentence):
        weight = [{}]  # 动态规划表
        path = {}

        # 初始化第一个词
        if sentence[0] not in self.observation_matrix['B']:
            for state in self.states:
                if state == 'S':
                    self.observation_matrix[state][sentence[0]] = 0
                else:
                    self.observation_matrix[state][sentence[0]] = -3.14e+100

        for state in self.states:
            weight[0][state] = self.init_states[state] + self.observation_matrix[state][sentence[0]]
            path[state] = [state]

        # 置分词开始和结束标志
        for state in self.states:
            if state == 'B':
                self.observation_matrix[state]['begin'] = 0
            else:
                self.observation_matrix[state]['begin'] = -3.14e+100
        for state in self.states:
            if state == 'E':
                self.observation_matrix[state]['end'] = 0
            else:
                self.observation_matrix[state]['end'] = -3.14e+100

        for i in range(1, len(sentence)):
            weight.append({})
            new_path = {}

            for state0 in self.states:  # state0表示sentence[i]的状态
                items = []
                for state1 in self.states:  # states1表示sentence[i-1]的状态
                    prob = weight[i - 1][state1] + self.states_matrix[state1][state0] + self.observation_matrix[state0][sentence[i]]
                    items.append((prob, state1))
                best = max(items)
                weight[i][state0] = best[0]
                new_path[state0] = path[best[1]] + [state0]
            path = new_path

        prob, state = max([(weight[len(sentence) - 1][state], state) for state in self.states])
        return path[state]

    def tag_seg(self, sentence, tag):
        word_list = []
        start = -1
        started = False

        if len(tag) != len(sentence):
            return None
        if len(tag) == 1:
            word_list.append(sentence[0])  # 语句只有一个字，直接输出
        else:
            if tag[-1] == 'B' or tag[-1] == 'M':  
                if tag[-2] == 'B' or tag[-2] == 'M':
                    tag[-1] = 'E'
                else:
                    tag[-1] = 'S'
            for i in range(len(tag)):
                if tag[i] == 'S':
                    word_list.append(sentence[i])
                elif tag[i] == 'B':
                    if started:
                        word_list.append(sentence[start:i])
                    start = i
                    started = True
                elif tag[i] == 'E':
                    started = False
                    word = sentence[start:i + 1]
                    word_list.append(word)
                elif tag[i] == 'M':
                    continue
        return word_list

    def segment(self, sentence):
        tag = self.viterbi(sentence)
        seg = self.tag_seg(sentence, tag)
        return seg
    
    def printmesege(self,hmm_precision,hmm_recall):
        self.logger.print("JieBa分词结果:", jieba_result)
        self.logger.print("Prefix dict has been built successfully.")
        self.logger.print(f"HMM算法准确率: {hmm_precision}, 召回率: {hmm_recall}")



def evaluate_segmentation(algorithm_result, jieba_result):
    # 将分词结果转化为集合以便比较
    set_algorithm = set(algorithm_result)
    set_jieba = set(jieba_result)
    
    # 计算交集和并集
    intersection = set_algorithm.intersection(set_jieba)
    union = set_algorithm.union(set_jieba)
    
    # 计算准确率和召回率
    precision = len(intersection) / len(set_algorithm) if set_algorithm else 0
    recall = len(intersection) / len(set_jieba) if set_jieba else 0
    
    return precision, recall

def HMMRun(x):


    with open('C:\\Users\\123\\Desktop\\合并版本\\机器学习实践\\hmm_states.txt', encoding='utf-8') as f:
        prameter = json.load(f)
    array_A = prameter['states_matrix']
    array_B = prameter['observation_matrix']
    array_Pi = prameter['init_states']
    STATES = ['B', 'M', 'E', 'S']

    hmm_model = HMM(array_A, array_B, array_Pi, STATES)

    hmm_result = hmm_model.segment(text)
    print('/ '.join(hmm_result))
    hmm_precision, hmm_recall = evaluate_segmentation(hmm_result, jieba_result)
    hmm_model.printmesege(hmm_precision,hmm_recall)
