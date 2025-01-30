import numpy as np
import torch


def confusion_matrix(scores_negatives, scores_positives, threshold):
    false_positives = np.sum(scores_negatives > threshold)
    true_postives = np.sum(scores_positives > threshold)
    true_negatives = np.sum(scores_negatives <= threshold)
    false_negatives = np.sum(scores_positives <= threshold)
    return false_positives, true_postives, true_negatives, false_negatives


def tpr_fpr(scores_negatives, scores_positives, threshold):
    fp, tp, tn, fn = confusion_matrix(scores_negatives, scores_positives, threshold)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr


# MLS Score
def mls(logits):
    scores = -torch.max(logits, dim=1)[0]
    return scores.cpu().numpy()


# Choix du meilleur seuil
def compute_threshold(scores, target_tpr=0.95):
    sorted_scores = np.sort(scores)
    target_index = int(np.ceil((1 - target_tpr) * len(sorted_scores))) - 1

    target_index = max(0, target_index)
    target_index = min(len(sorted_scores) - 1, target_index)

    threshold = sorted_scores[target_index]

    return threshold
