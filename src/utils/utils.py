import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, hamming_loss, label_ranking_loss, coverage_error
from collections import defaultdict
from rex.metrics import calc_p_r_f1_from_tp_fp_fn

# **
def classification_auc(gold_labels:np, pred_sources:np):
    '''
    parameter:
        gold_labels: 金标标签 (num_simple, num_class)
        pred_sources: 预测概率(num_simple, num_class)
    return: 
        AUC
    '''
    roc_auc = dict()
    roc_auc['micro'] = roc_auc_score(gold_labels, pred_sources, average='micro')
    # macro: 以label为单位, 计算每个label的auc再取均值
    try:
        roc_auc['macro'] = roc_auc_score(gold_labels, pred_sources, average='macro')
    except ValueError:
        roc_auc['macro'] = 0
    
    return roc_auc
   

def classification_p_at_k_one_instance(gold_label:list, pred_source:list, k:int, threshold:float):
    '''
    parameter:
        gold_label:  金标01串 (num_class)
        pred_source: 预测概率 (num_class)
        k: top_k
        threshold: 阈值
    return: 
        p_at_k: top k 的精确率
    '''
    if k == 0:
        return 0
    num_class = len(gold_label)
    if k > len(gold_label):
        raise ValueError(f'k值大于类别数量 k:{k}, num_class:{num_class}')
    pred_gold = []
    for gold, pred in zip(gold_label, pred_source):
        pred_gold.append((gold, pred))
    # 排序
    sorted_pred_gold = list(sorted(pred_gold, key=lambda x:x[1], reverse=True))
    tp = 0
    for i in range(k):
        if sorted_pred_gold[i][0] == 1 and sorted_pred_gold[i][1] >= threshold:
            tp += 1
    return tp/k


def classification_p_at_k(gold_labels:list, pred_sources:list, k:int, threshold:float):
    '''
    parameter:
        gold_labels: 金标标签 (num_simple, num_class)
        pred_sources: 预测概率(num_simple, num_class)
        k: top_k 
        threshold: 阈值
    return: 
        p_at_k: top k 的精确率
    '''
    all_p_at_k = 0
    for gold_label, pred_source in zip(gold_labels, pred_sources):
        all_p_at_k += classification_p_at_k_one_instance(gold_label, pred_source, k, threshold)
    return all_p_at_k/len(gold_labels)


def classification_average_p_at_k(gold_label:list, pred_source:list):
    '''
    parameter:
        gold_label:  金标01串 (num_class)
        pred_source: 预测概率 (num_class)
    return: 
        ap: average_p
    '''
    pred_gold = []
    for gold, pred in zip(gold_label, pred_source):
        pred_gold.append((gold, pred))
    # 排序
    sorted_pred_gold = list(sorted(pred_gold, key=lambda x:x[1], reverse=True))
    
    all_ap = 0
    gold_count = 0
    for idx, item in enumerate(sorted_pred_gold):
        if item[0] == 1:
            gold_count += 1
            all_ap += (gold_count/(idx + 1))
    return all_ap/gold_count


def classification_mean_average_p_at_k(gold_labels:list, pred_sources:list):
    '''
    parameter:
        gold_labels: 金标标签 (num_simple, num_class)
        pred_sources: 预测概率(num_simple, num_class)
    return: 
        map: mean_average_p
    '''
    all_ap = 0
    for gold, pred in zip(gold_labels, pred_sources):
        all_ap += classification_average_p_at_k(gold, pred)
    return all_ap/(len(gold_labels))

# **
def mcml_prf1(preds, golds, id2label: dict):
    few_labels = ['关闭分支机构', '澄清辟谣', '第一大股东变化', '产品质量问题', '非法集资', '停产停业', '实际控制人违规', '引入新股东', '重大安全事故', '资金紧张', '债务融资失败', '承担赔偿责任', '吊销资质牌照']
    if len(preds) == 0 or len(golds) == 0:
        raise ValueError("Preds or golds is empty.")
    measure_results = defaultdict(dict)
    supports = defaultdict(dict)

    for value in id2label.values():
        measure_results[value] = {"p": 0.0, "r": 0.0, "f1": 0.0}
        supports[value] = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    measure_results['few_label'] = {"p": 0.0, "r": 0.0, "f1": 0.0}
    supports['few_label'] = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    all_tp, all_fp, all_fn = 0, 0, 0
    for pred, gold in zip(preds, golds):
        if len(pred) != len(gold):
            raise ValueError(
                f"Pred: {pred} cannot be matched with gold: {gold} in length."
            )
        for type_idx, (p, g) in enumerate(zip(pred, gold)):
            label = id2label[type_idx]
            if p == 1 and g == 1:
                supports[label]["tp"] += 1
                if label in few_labels:
                    supports['few_label']["tp"] += 1
                all_tp += 1
            elif p == 1 and g == 0:
                supports[label]["fp"] += 1
                if label in few_labels:
                    supports['few_label']["fp"] += 1
                all_fp += 1
            elif p == 0 and g == 1:
                supports[label]["fn"] += 1
                if label in few_labels:
                    supports['few_label']["fn"] += 1
                all_fn += 1

    for label in supports:
        measure_results[label] = calc_p_r_f1_from_tp_fp_fn(
            supports[label]["tp"], supports[label]["fp"], supports[label]["fn"]
        )

    measure_results["micro"] = calc_p_r_f1_from_tp_fp_fn(all_tp, all_fp, all_fn)

    return measure_results

# **
def evaluation(y_true, y_prob, y_pred):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)
    class_freq = np.sum(y_true, axis=0)
    Hamming = hamming_loss(y_true, y_pred)
    Ranking = label_ranking_loss(y_true, y_prob)
    Coverage = (coverage_error(y_true, y_prob) - 1) / y_prob.shape[1]
    mAP = average_precision_score(y_true[:, class_freq != 0], y_prob[:, class_freq != 0], average='macro')

    return {'Hamming': Hamming, 'Ranking': Ranking, 'Coverage': Coverage, 'mAP': mAP, }


def ndcg_k(y_true:list, y_prob:list, k=[1, 3, 5]):
    batch_size = len(y_true)
    ndcg = []
    for _k in k:
        score = 0
        for i in range(batch_size):
            one_true = y_true[i]
            one_prob = y_prob[i]
            idcg_true = sorted(one_true, key=lambda x:x, reverse=True)
            dcg_true_prob = sorted(zip(one_true, one_prob), key=lambda x:x[1], reverse=True)
            dcg_val = 0
            idcg_val = 0
            for j in range(_k):
                idcg_val = idcg_val + idcg_true[j]/np.log2(j+2)
                dcg_val = dcg_val + dcg_true_prob[j][0]/np.log2(j+2)
            score += (dcg_val / idcg_val)

        ndcg.append(score * 100 / batch_size)
    return ndcg
