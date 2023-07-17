import torch


def get_accuracy_from_logits(logits, labels):
    # Convert logits to probabilties
    probabilties = torch.sigmoid(logits.unsqueeze(-1))
    # Convert probabilities to predictions (1: positive, 0: negative)
    predictions = (probabilties > 0.5).long().squeeze()

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(labels)):
        if predictions[i] == 1 and labels[i] == 1:
            TP += 1
        elif predictions[i] == 0 and labels[i] == 1:
            FN += 1
        elif predictions[i] == 0 and labels[i] == 0:
            TN += 1
        else:
            FP += 1
    if TP != 0:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
    else:
        precision = 0
        recall = 0

    # Calculate qnd return accuracy
    return (predictions == labels).float().mean(), precision, recall
