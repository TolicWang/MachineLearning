# -*- coding: utf-8 -*-
# Author: wangchengo@126.com
import numpy as np
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
class Metrics():
    y=[]
    y_pre=[]
    def __init__(self,y_true,y_pre):
        self.y=y_true
        self.y_pre=y_pre

    def getFscAccNmiAri(self):
        y=self.y
        y_pre=self.y_pre
        Fscore,Accuracy=self.getFscoreAndAcc()
        NMI = normalized_mutual_info_score(y, y_pre)
        ARI = adjusted_rand_score(y, y_pre)
        return Fscore,Accuracy,NMI,ARI

    def getFscoreAndAcc(self):
        y=np.array(self.y)
        y_pre=np.array(self.y_pre)
        n = len(y_pre)
        p = np.unique(y)
        c = np.unique(y_pre)
        p_size = len(p)
        c_size = len(c)

        a = np.ones((p_size, 1), dtype=int) * y  # p_size by 1  *  1 by n   ==> p_size by n
        b = p.reshape(p_size, 1) * np.ones((1, n), dtype=int)  # p_size by 1 * 1 by n ==> p_size by n
        pid = (a == b) * 1  # p_size by n

        a = np.ones((c_size, 1), dtype=int) * y_pre  # c_size by 1 * 1 by n ==> c_size by n
        b = c.reshape(c_size, 1) * np.ones((1, n))
        cid = (a == b) * 1  # c_size by n

        cp = np.dot(cid, pid.T)
        pj = np.sum(cp, axis=0)
        ci = np.sum(cp, axis=1)

        precision = cp / (ci.reshape(len(ci), 1) * np.ones((1, p_size), dtype=float))
        recall = cp / (np.ones((c_size, 1), dtype=float) * pj.reshape(1, len(pj)))

        F = 2 * precision * recall / (precision + recall)

        F = np.nan_to_num(F)

        temp = (pj / float(pj.sum())) * np.max(F, axis=0)
        Fscore = np.sum(temp, axis=0)

        temp = np.max(cp, axis=1)
        Accuracy = np.sum(temp, axis=0) / float(n)
        return (Fscore, Accuracy)