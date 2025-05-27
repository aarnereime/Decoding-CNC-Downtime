def calculate_pos_weight(labels):
    positive_ratio = labels.sum() / len(labels)
    return (1 - positive_ratio) / positive_ratio