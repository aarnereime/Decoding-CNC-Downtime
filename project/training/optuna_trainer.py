import torch
import torch.nn as nn
import copy

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from project.utils.seed import set_seed


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for num_seq, cat_seq, label, lengths in train_loader:
        num_seq, cat_seq, label, lengths = num_seq.to(device), cat_seq.to(device), label.to(device), lengths.to(device)
        optimizer.zero_grad()
        
        logits = model(num_seq, cat_seq, lengths)
        label = label.unsqueeze(1)
        loss = criterion(logits, label)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # Gradient clipping to prevent exploding gradients
        optimizer.step()
        
        running_loss += loss.item()
        
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        correct += (predictions == label).sum().item()
        total += label.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return epoch_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_val_preds = []
    all_val_labels = []
    all_val_probs = []
    
    with torch.no_grad():
        for num_seq, cat_seq, label, lengths in val_loader:
            num_seq, cat_seq, label, lengths = num_seq.to(device), cat_seq.to(device), label.to(device), lengths.to(device)
            
            logits = model(num_seq, cat_seq, lengths)
            label = label.unsqueeze(1)
            loss = criterion(logits, label)
            
            running_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            predictions = (probs >= 0.5).float()
            correct += (predictions == label).sum().item()
            total += label.size(0)
            
            all_val_preds.extend(predictions.cpu().view(-1).numpy())
            all_val_labels.extend(label.cpu().view(-1).numpy())
            all_val_probs.extend(probs.cpu().view(-1).numpy())
    
    epoch_loss = running_loss / len(val_loader)
    accuracy = correct / total
    
    precision = precision_score(all_val_labels, all_val_preds, zero_division=0)
    recall = recall_score(all_val_labels, all_val_preds, zero_division=0)
    f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
    fpr, tpr, _ = roc_curve(all_val_labels, all_val_probs)
    roc_auc = roc_auc_score(all_val_labels, all_val_probs)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }
    return epoch_loss, accuracy, metrics


def optuna_train_model(model, train_loader, val_loader, device, optimizer, num_epochs):
    set_seed(42)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss().to(device) 

    best_val_acc = float('-inf')
    best_metrics = {}
    best_wts = copy.deepcopy(model.state_dict())
    early_stop_count = 0
    patience = 15
    delta = 0.001

    losses_train, losses_val = [], []

    # Quick class balance check on a few batches
    batch_labels = []
    for i, (_, _, y, _) in enumerate(train_loader):
        batch_labels += y.cpu().tolist()
        if i == 10: break
    print(f'Class balance in 10 batches: {sum(batch_labels)/len(batch_labels):.3f}')

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, metrics = validate_epoch(model, val_loader, criterion, device)

        losses_train.append(train_loss)
        losses_val.append(val_loss)

        print(f'Epoch {epoch}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | '
              f'Val Precision: {metrics['precision']:.4f}, '
              f'Val Recall: {metrics['recall']:.4f}, '
              f'Val F1: {metrics['f1_score']:.4f}, '
              f'ROC AUC: {metrics['roc_auc']:.4f}'
        )

        # Early stopping logic
        if val_acc > best_val_acc + delta:
            best_val_acc = val_acc
            best_metrics = metrics
            best_wts = copy.deepcopy(model.state_dict())
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f'Early stopping at epoch {epoch}. Best val acc: {best_val_acc:.4f}')
                break

    # Load best model weights
    model.load_state_dict(best_wts)
    return model, best_val_acc, best_metrics, losses_train, losses_val