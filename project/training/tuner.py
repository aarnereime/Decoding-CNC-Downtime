import optuna
import torch

from project.models.MODEL_REGISTRY import MODEL_REGISTRY, initialize_model
from project.utils.dataset_and_dataloader import get_data_loader
from project.training.optuna_trainer import optuna_train_model
from project.utils.seed import set_seed
from project.config.model_config import NUM_NUMERICAL_FEATURES, VOCAB_SIZES, CATEGORICAL_EMBEDDING_DIMS

set_seed(42)

NUM_EPOCHS = 150
NUM_TRIALS = 50


def bayesian_search(train_dataset, val_dataset, device, num_trials=NUM_TRIALS):
    
    def print_model_info(model_name, hidden_size, num_layers, batch_size, dropout, learning_rate, weight_decay, optimizer_name):
        print('----------------------Hyperparameter tuning model----------------------')
        print('Model:', model_name)
        print('Hyperparameters:', {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'optimizer': optimizer_name
        })
    
    def objective(trial):
        # Hyperparameters to search 
        model_name = trial.suggest_categorical('model_name', MODEL_REGISTRY.keys())
        hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        batch_size = trial.suggest_categorical('batch_size', [64, 128])
        dropout = trial.suggest_float('dropout', 0.2, 0.7, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop'])

        # Create data loaders
        train_loader = get_data_loader(train_dataset, batch_size, use_weighted_sampler=True)
        val_loader = get_data_loader(val_dataset, batch_size)
        
        print_model_info(model_name, hidden_size, num_layers, batch_size, dropout, learning_rate, weight_decay, optimizer_name)
        
        model = initialize_model(
            model_name, 
            num_numerical_features=NUM_NUMERICAL_FEATURES, 
            categorical_vocab_sizes=VOCAB_SIZES, 
            categorical_embedding_dims=CATEGORICAL_EMBEDDING_DIMS, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout
        ).to(device)
        
        optimizer_classes = {
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
        }

        # Optimizer parameters
        optimizer_kwargs = {'lr': learning_rate, 'weight_decay': weight_decay}
        optimizer = optimizer_classes[optimizer_name](model.parameters(), **optimizer_kwargs)
        
        best_model, best_acc, metrics, train_loss, val_loss = optuna_train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            optimizer=optimizer,
            num_epochs=NUM_EPOCHS,
        )

        trial.set_user_attr('model', best_model)
        trial.set_user_attr('val_precision', metrics['precision'])
        trial.set_user_attr('val_recall', metrics['recall'])
        trial.set_user_attr('val_f1', metrics['f1_score'])
        trial.set_user_attr('train_loss', train_loss)
        trial.set_user_attr('val_loss', val_loss)
        trial.set_user_attr('roc_auc', metrics['roc_auc'])
        trial.set_user_attr('fpr', metrics['fpr'])
        trial.set_user_attr('tpr', metrics['tpr'])
        return best_acc
    
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=30,
            interval_steps=1,
            n_min_trials=5
        )  
    )
    study.optimize(objective, n_trials=num_trials)
    
    best = study.best_trial
    best_model = best.user_attrs.get('model', None)
    best_params = best.params
    
    train_loss = best.user_attrs.get('train_loss', [])
    val_loss = best.user_attrs.get('val_loss', [])
    metrics = {
        'accuracy': best.value,
        'precision': best.user_attrs.get('val_precision', 0.0),
        'recall': best.user_attrs.get('val_recall', 0.0),
        'f1': best.user_attrs.get('val_f1', 0.0),
        'roc_auc': best.user_attrs.get('roc_auc', 0.0),
        'fpr': best.user_attrs.get('fpr', []),
        'tpr': best.user_attrs.get('tpr', [])      
    }
    
    print(f'Best model: {best_model.__class__.__name__}')
    print(f'Best hyperparameters: {best_params} \n'
          f'with validation acc: {best.value:.4f}, '
          f'precision: {metrics['precision']:.4f}, '
          f'recall: {metrics['recall']:.4f}, '
          f'f1: {metrics['f1']:.4f}, '
          f'roc_auc: {metrics['roc_auc']:.4f}')
    
    return best_model, best_params, metrics, train_loss, val_loss
