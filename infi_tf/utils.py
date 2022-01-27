
def eval_filter(y_true, y_pred, threshold=0.5):
    """
    Args:
        y_true (list): true execution labels
        y_pred (list): predicted execution probabilities
        threshold (float): if p<threshold: skip the execution

    Return:
        label_acc (float): inference accuracy, only punishes False Negative cases
        filtered_rate (float): ratio of filtered data
    """
    wrong_count = 0
    filtered_count = 0
    for y1, y2 in zip(y_true, y_pred):
        # FN case brings wrong label
        if y1[0] == 1 and y2[0] <= threshold:
            wrong_count += 1
        if y2[0] <= threshold:
            filtered_count += 1
    
    total_num = len(y_true)
    filtered_rate = filtered_count / total_num
    inference_acc = 1. - wrong_count / total_num
    
    return inference_acc, filtered_rate