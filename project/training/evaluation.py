import torch
from project.utils.seed import set_seed
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

set_seed(42)


def evaluate_model(model, loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for num_seq, cat_seq, label, lengths in loader:
            num_seq, cat_seq, label, lengths = num_seq.to(device), cat_seq.to(device), label.to(device), lengths.to(device)
            logits = model(num_seq, cat_seq, lengths)
            
            probs = torch.sigmoid(logits)
            predictions = (probs >= 0.5).float()
            correct += (predictions == label.unsqueeze(1)).sum().item()
            total += label.size(0)
            
            all_preds.extend(predictions.cpu().view(-1).numpy())
            all_labels.extend(label.cpu().view(-1).numpy())
            all_probs.extend(probs.cpu().view(-1).numpy())

        accuracy = correct / total 
        
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = roc_auc_score(all_labels, all_probs)
        
    print(f'{model.__class__.__name__} Model Evaluation on Test Data:')
    print(f'- Validation Accuracy: {accuracy:.4f}')
    print(f'- Validation Precision: {precision:.4f}')
    print(f'- Validation Recall: {recall:.4f}')
    print(f'- Validation F1: {f1:.4f}')
    print(f'- ROC AUC: {roc_auc:.4f}')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }