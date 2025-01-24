import os
import numpy as np
import csv
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve

def evaluate(all_labels, all_predictions, threshold=None):
    auroc = roc_auc_score(all_labels, all_predictions)
    auprc = average_precision_score(all_labels, all_predictions)

    if threshold is None:
        # finding the best threshold and f1 score
        precision, recall, thresholds = precision_recall_curve(all_labels, all_predictions)
        # Calculate F-score
        fscore = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
        # Find the index of the largest F-score
        ix = np.argmax(fscore)

        thresh = thresholds[ix]
        predictions_binary = [1 if prob > thresh else 0 for prob in all_predictions]
        acc = accuracy_score(all_labels, predictions_binary)
        f1 = fscore[ix]
        precision = precision[ix]
        recall = recall[ix]

    if threshold is not None:
        predictions_binary = [1 if prob > threshold else 0 for prob in all_predictions]
        acc = accuracy_score(all_labels, predictions_binary)
        f1 = f1_score(all_labels, predictions_binary)
        precision = precision_score(all_labels, predictions_binary)
        recall = recall_score(all_labels, predictions_binary)
        thresh = threshold


    metrics = {
        'AUROC': auroc,
        'AUPRC': auprc,
        'ACC': acc,
        'F1': f1,
        'Precision': precision,
        'Recall':recall,
        'Treshold': thresh,
    }

    return metrics

def log_results(metrics, experiment_name, filepath):
    """
    Log the results to a CSV file.

    Args:
        metrics (dict): A dictionary of evaluation metrics.
        experiment_name (str): The name of the experiment.
        filepath (str): The path to the log file.
    """
    fieldnames = ['experiment', 'AUROC', 'AUPRC', 'ACC', 'F1', 'Precision', 'Recall', 'Treshold']

    # check if file exists
    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow({'experiment': experiment_name, **metrics})