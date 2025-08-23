import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import torch
import torch.nn.functional as F
from models.LMGNN import BertGGCN
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.figure.plot import plot_confusion_matrix, plot_roc_curve


def test(model, device, test_loader, figure_save_path = ''): 
    """
    Tests the model using the provided test data loader.

    :param model: The model to be tested.
    :param device: The device to perform testing on (e.g., 'cpu' or 'cuda').
    :param test_loader: The data loader containing the test data.
    :return: Tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []

    for batch_idx, batch in enumerate(test_loader):
        batch.to(device)
        with torch.no_grad():
            y_ = model(batch)

        batch.y = batch.y.squeeze().float()
        # BCE on probabilities
        test_loss += F.binary_cross_entropy(y_.squeeze(-1), batch.y).item()

        probs = y_.squeeze(-1)
        pred = (probs > 0.5).long()
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Plots moved to utils.figure.plot
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(figure_save_path, 'confusion_matrix.png'), labels=['benign', 'malware'])
    plot_roc_curve(y_true, y_probs, save_path=os.path.join(figure_save_path, 'roc_curve.png'))

    results_array = np.column_stack((y_true, y_pred, y_probs))
    header_text = "True label, Predicted label, Predicted Probability"
    np.savetxt(os.path.join(figure_save_path, 'results.txt'), results_array, fmt='%1.6f', delimiter='\t', header=header_text)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1