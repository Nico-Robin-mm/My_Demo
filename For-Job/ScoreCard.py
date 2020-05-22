# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:28:36 2020

@author: 木木
"""


import pandas as pd
import scipy.stats.stats as stats
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix
from math import log


# 利用随机森林填充缺失值
def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    known = np.array(df[df.iloc[:, 5].notna()])
    unknown = np.array(df[df.iloc[:, 5].isna()])
    X_train = known[:, [0, 1, 2, 3, 4, 6, 7, 8, 9]]
    y_train = known[:, 5]
    rfr = RandomForestRegressor(random_state=0, n_estimators=500, max_depth=3, n_jobs=-1)
    rfr.fit(X_train, y_train)
    X_pred = unknown[:, [0, 1, 2, 3, 4, 6, 7, 8, 9]]
    unknown[:, 5] = rfr.predict(X_pred).round(0)
    return df


# 决策树CART分箱
def decision_tree_bin(x: pd.Series, y: pd.Series) -> list:
    # 分7箱，最小分箱数5%，return分割点threshold
    clf = tree.DecisionTreeClassifier(criterion="entropy",
                                      max_leaf_nodes=7,
                                      min_samples_leaf=0.05)
    X = np.array(x).reshape(-1, 1)
    clf.fit(X, y)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    boundary = []
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:
            boundary.append(threshold[i])
    sort_boundary = sorted(boundary)
    sort_boundary = [float("-inf")] + sort_boundary + [float("inf")]
    return sort_boundary


# 利用斯皮尔曼相关系数给出单调性指标
def monoto(x: pd.Series, y: pd.Series, boundary: list) -> float:
    df = pd.concat([x, y], axis=1)
    df.columns = ["x", "y"]
    df["bins"] = pd.cut(x=x, bins=boundary)
    grouped = df.groupby("bins")
    r, p = stats.spearmanr(grouped.mean().x, grouped.mean().y)
    return r


# 计算woe和iv值
def woe_iv(x: pd.Series, y: pd.Series, boundary: list) -> pd.DataFrame:
    df = pd.concat([x, y], axis=1)
    df.columns = ["x", "y"]
    df["bins"] = pd.cut(x=x, bins=boundary)
    grouped = df.groupby("bins")
    result_df = pd.DataFrame()
    result_df["bad"] = grouped.y.sum()
    result_df["total"] = grouped.y.count()
    result_df["good"] = result_df["total"] - result_df["bad"]
    result_df["bad_rate"] = result_df["bad"] / result_df["bad"].sum()
    result_df["good_rate"] = result_df["good"] / result_df["good"].sum()
    result_df["woe"] = np.log(result_df["bad_rate"] / result_df["good_rate"])
    result_df["iv"] = (result_df["bad_rate"] - result_df["good_rate"]) * result_df["woe"]
    return result_df


# 替换woe值
def replace_woe(x: pd.Series, boundary: list, woe: list) -> np.array:
    lst = []
    i = 0
    while i < len(x):
        value = x.values[i]
        j = len(boundary) - 2
        m = len(boundary) - 2
        while j >= 0:
            if value > boundary[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        lst.append(woe[m])
        i += 1
    return np.array(lst)


# 计算分数函数
def get_score(coef: float, woe: list, B: float) -> list:
    scores = []
    for w in woe:
        score = round(coef * w * B, 0)
        scores.append(score)
    return scores


# 根据变量计算分数
def compute_score(x: pd.Series, boundary: list, score: int) -> list:
    lst = []
    i = 0
    while i < len(x):
        value = x.values[i]
        j = len(boundary) - 2
        m = len(boundary) - 2
        while j >= 0:
            if value > boundary[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        lst.append(score[m])
        i += 1
    return lst


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 数据读取
    data = pd.read_csv("./cs-training.csv")
    print(data.isna().sum())
    # MonthlyIncome存在29731缺失值， NumberOfDependents存在3924缺失值
    # 删除重复项
    data = data.drop_duplicates()
    
    # 异常值分析
    fig1 = plt.figure(figsize=(20, 15))
    ax1 = fig1.add_subplot(2, 2, 1)
    ax2 = fig1.add_subplot(2, 2, 2)
    ax3 = fig1.add_subplot(2, 2, 3)
    ax4 = fig1.add_subplot(2, 2, 4)
    ax1.boxplot([data.iloc[:, 1], data.iloc[:, 4]])
    ax1.set_xticklabels(["可用额度比值", "负债率"], fontsize=20)
    sns.distplot(data.iloc[:, 2], ax=ax2, color="b")
    ax2.set_xlabel("年龄", fontsize=20)
    ax3.boxplot([data.iloc[:, 3], data.iloc[:, 9], data.iloc[:, 7]])
    ax3.set_xticklabels(["逾期30-59天笔数", "逾期60-89天笔数", "逾期90天笔数"], fontsize=20)
    sns.distplot(data[data.iloc[:, 5] < 30000].iloc[:, 5], ax=ax4)
    ax4.set_xlabel("月收入", fontsize=20)
    plt.show()
    
    # 删除age为0的项
    data = data[data.age > 0]
    
    # 三种日期逾期天数的箱型图 明显存在个别离群值 选择剔除
    data = data[data["NumberOfTime30-59DaysPastDueNotWorse"] < 90]
    data = data[data["NumberOfTime60-89DaysPastDueNotWorse"] < 90]
    data = data[data["NumberOfTimes90DaysLate"] < 90]
    
    # 随机森林回归填充MonthlyIncome中的空缺值
    data = fill_missing(data)
    
    # 选择直接删除NumberOfDependents中数量较少的空缺值
    data = data.dropna()
    
    # 各变量相关性分析 Pearson相关系数 热力图
    corr = data.corr()
    fig2 = plt.figure(figsize=(10, 10))
    ax5 = sns.heatmap(corr, annot=True, cmap="Greens")
    ax5.set_xticklabels(["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"], 
                        rotation=0, fontsize=10)
    ax5.set_yticklabels(corr.index, fontsize=10)
    plt.show()
    # 各变量间相关性系数较小，不需要操作

    # binning
    for i in range(1, 11):
        locals()["boundary_{0}".format(i)] = decision_tree_bin(data.iloc[:, i],
                                                               data.iloc[:, 0])
    # 用Spearman相关系数表示单调性程度。取值[-1, 1]，绝对值越大，单调性越强
    for i in range(1, 11):
        locals()["r_{0}".format(i)] = monoto(data.iloc[:, i], data.iloc[:, 0],
                                             locals()["boundary_{0}".format(i)])
    # woe, iv值
    for i in range(1, 11):
        locals()["result_df_{0}".format(i)] = woe_iv(data.iloc[:, i], data.iloc[:, 0],
                                                     locals()["boundary_{0}".format(i)])
    # 求每个变量总的iv值.（0,0.03）无预测能力，(0.03,0.09)低预测能力，(0.1,0.29)中，
    # (0.3,0.49)高，(0.5,inf)极高且可疑
    # 一般应大于0.02，默认选IV大于0.1的变量进模型
    for i in range(1, 11):
        locals()["iv_{0}".format(i)] = locals()["result_df_{0}".format(i)].iv.sum()
    
    # 各个特征变量的iv值柱状图可视化
    iv = []
    for i in range(1, 11):
        iv.append(locals()["iv_{0}".format(i)])
    iv_df = pd.DataFrame(
        iv,
        index=["可用额度比值", "年龄", "逾期30-59天笔数", "负债率", "月收入", "信贷数量",
               "逾期90天笔数", "固定资产贷款量", "逾期60-89天笔数", "家属数量"],
        columns=["IV"])
    fig3 = plt.figure(figsize=(10, 6))
    ax6 = fig3.add_subplot(1, 1, 1)
    iv_bar = iv_df.plot.bar(color="b", alpha=0.3, rot=30, fontsize=(10), ax=ax6)
    iv_bar.set_title("特征变量与IV值分布图", fontsize=(20))
    iv_bar.set_xlabel("特征变量", fontsize=(15))
    iv_bar.set_ylabel("IV", fontsize=(15))
    plt.show()
    # 选择 1, 2, 3, 7, 9 这五组features

    # 将result_df中的woe值提取出来 -> list
    for i in range(1, 11):
        locals()["woe_list_{0}".format(i)] = list(locals()["result_df_{0}".format(i)].woe.round(3))
    # 替换woe值
    for i in range(1, 11):
        locals()["woe_{0}".format(i)] = replace_woe(data.iloc[:, i], locals()["boundary_{0}".format(i)],
                                                    locals()["woe_list_{0}".format(i)])
    # 创建一个新的DataFrame，放入woe数据
    data_new = pd.DataFrame(columns=["y", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"])
    for i in range(1, 11):
        data_new.iloc[:, i] = locals()["woe_{0}".format(i)]
    data_new.iloc[:, 0] = pd.Series(np.array(data.SeriousDlqin2yrs))
    
    # 划分训练集和测试集
    X = data_new.iloc[:, [1, 2, 3, 7, 9]]
    y = data_new.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=666)
    
    # logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    coef = log_reg.coef_
    intercept = log_reg.intercept_
    decision_scores = log_reg.decision_function(X_test)
    
    # roc_curve and roc_auc_score
    fprs, tprs, thresholds = roc_curve(y_test, decision_scores)
    rocauc_score = roc_auc_score(y_test, decision_scores)
    fig4 = plt.figure(figsize=(10, 10))
    plt.plot(fprs, tprs, label="ROC Curve(area=%.2f)" % rocauc_score)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC_Curve")
    plt.legend(loc="best")
    plt.show()
    
    # ks curve
    fig5 = plt.figure(figsize=(10, 10))
    ks_value = max(abs(fprs - tprs))
    plt.plot(fprs, label="fpr")
    plt.plot(tprs, label="tpr")
    plt.plot(abs(fprs - tprs), label="diff")
    index = np.argwhere(abs(fprs - tprs) == ks_value)[0, 0]
    plt.plot((index, index), (0, ks_value), label="ks-%.2f" % ks_value, marker="o", color="k")
    plt.scatter((index, index), (0, ks_value), color="k")
    plt.legend()
    plt.title("KS Curve")
    plt.show()
    
    # 选择最优的阈值，同时给出混淆矩阵，精准率，回收率，f1-score
    y_pred_original = log_reg.predict(X_test)
    print("原混淆矩阵:\n", confusion_matrix(y_test, y_pred_original))
    print("原precision_score:\n", precision_score(y_test, y_pred_original))
    print("原recall_score:\n", recall_score(y_test, y_pred_original))
    print("原f1_score:\n", recall_score(y_test, y_pred_original))
    y_pred = np.array(decision_scores >= thresholds[index], dtype="int")
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
    print("precision_score:\n", precision_score(y_test, y_pred))
    print("recall_score:\n", recall_score(y_test, y_pred))
    print("f1_score:\n", f1_score(y_test, y_pred))
    
    # score card
    """
    odds = p / (1 - p)
    score = A - B * log(odds)
    由 P0 = A - B * log(α)， P0 - PDO = A - B * log(2α) 得
    A = P0 + B * log(α)
    B = PDO / log(2)
    当好坏比为20时，对应的评分600分，令PDO=20
    而本例中的positive实际对应的是违约（坏好率为1/20）。即α=1/20
    也就是说：坏好率1/20时为600分，坏好率1/40时为620
    600 = A - B * log(1/20)， 620 = A - B * log(1/40)
    B = PDO / log(2) = 20 / log(2)
    A = 600 - 20 * log(20) / log(2) = 600 - 20 * log(20) / log(2)
    """
    A = 600 - 20 * log(20) / log(2)
    B = 20 / log(2)
    BASESCORE = round(A - B * float(intercept), 0) # result is 588
    
    # 得到对应分数
    score_corr_1 = get_score(coef[0, 0], woe_list_1, B)
    score_corr_2 = get_score(coef[0, 1], woe_list_2, B)
    score_corr_3 = get_score(coef[0, 2], woe_list_3, B)
    score_corr_7 = get_score(coef[0, 3], woe_list_7, B)
    score_corr_9 = get_score(coef[0, 4], woe_list_9, B)
    
    # 计算test样本中每个人的分数值
    test_data = pd.concat([y_test, X_test], axis=1)
    test_data.to_csv("./test_data.csv", index=False)
    score_card = pd.read_csv("./test_data.csv")
    score_card["x1_score"] = pd.Series(compute_score(score_card.iloc[:, 1], boundary_1, score_corr_1))
    score_card["x2_score"] = pd.Series(compute_score(score_card.iloc[:, 2], boundary_2, score_corr_2))
    score_card["x3_score"] = pd.Series(compute_score(score_card.iloc[:, 3], boundary_3, score_corr_3))
    score_card["x7_score"] = pd.Series(compute_score(score_card.iloc[:, 4], boundary_7, score_corr_7))
    score_card["x9_score"] = pd.Series(compute_score(score_card.iloc[:, 5], boundary_9, score_corr_9))
    score_card["BaseScore"] = pd.Series(np.zeros(len(y_test))) + BASESCORE
    score_card["Score"] = score_card["BaseScore"] - score_card["x1_score"] - score_card["x2_score"] \
        - score_card["x3_score"] - score_card["x7_score"] - score_card["x9_score"]
    score_card.to_csv("./Score_Card.csv", index=False)
