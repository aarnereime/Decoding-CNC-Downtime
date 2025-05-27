import torch
import argparse
import json
from pathlib import Path
from project.config.root_dir import ROOT_DIR
import os

from project.utils.seed import set_seed
from project.training.evaluation import evaluate_model
from project.utils.gpu_selector import GPUSelector
from project.utils.dataset_and_dataloader import make_dataset, get_data_loader
from project.utils.result_manager import SaveResult
from project.training.tuner import bayesian_search
from project.models.logistic_regression import LogisticRegressionModel
from project.models.XGBoost import XGBModel


logreg = LogisticRegressionModel()
xgb = XGBModel()


def select_gpu():
    gpu_selector = GPUSelector()  
    selected_gpu_index = gpu_selector.select_gpu()
    device = torch.device(f'cuda:{selected_gpu_index}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device


def train_logistic_regression():
    logreg_pipeline = logreg.make_pipeline()
    best_model, best_params = logreg.tune_model(logreg_pipeline)
    val_metrics = logreg.evaluate(best_model, 'Validation')
    return best_model, best_params, val_metrics


def train_xgb_model():
    xgb_pipeline = xgb.make_pipeline()
    best_model, best_params = xgb.tune_model(xgb_pipeline)
    val_metrics = xgb.evaluate(best_model, 'Validation')
    return best_model, best_params, val_metrics
    
    
def train_lstm_models(device):
    # Initialize datasets
    train_dataset = make_dataset('train', device)
    class_distribution = train_dataset.labels.cpu().long()
    class_distribution = torch.bincount(class_distribution)
    class_distribution = class_distribution.float() / class_distribution.sum()
    print(f'Class distribution in training set: {class_distribution}')
    
    val_dataset = make_dataset('val', device)
    
    # Perform Bayesian search for hyperparameter tuning
    best_model, best_params, val_metrics, train_loss, val_loss = bayesian_search(train_dataset, val_dataset, device)
    return best_model, best_params, val_metrics, train_loss, val_loss
    

def retrieve_final_model(path):
    tuning_path = Path(path)
    best_model = None
    
    with open(tuning_path / 'stats.json', 'r') as f:
        best_model = json.load(f)
    
    return best_model     
    

if __name__ == '__main__':
    set_seed(42)
    
    # Parse command line arguments making it possible to choose which model to train when using Birget
    parser = argparse.ArgumentParser(description='Training models')
    parser.add_argument('--model', type=str, choices=['lstm', 'xgb', 'logreg', 'all'], default='all',
                        help='Which model(s) to train')
    
    parser.add_argument('--tune_models', action='store_true', help='Whether to tune models or not')
    parser.add_argument('--no_tune_models', dest='tune_models', action='store_false')
    parser.set_defaults(tune_models=True)

    parser.add_argument('--test_best_model', action='store_true', help='Whether to test the best model or not')
    parser.add_argument('--no_test_best_model', dest='test_best_model', action='store_false')
    parser.set_defaults(test_best_model=False)
    args = parser.parse_args()
    
    print(f'Arguments: {args}')
    
    device = select_gpu()
    
    tuned_model_information = {}
    
    save_result = SaveResult()
    
    if args.tune_models:
        if args.model in ['logreg', 'all']:
            best_model, best_params, val_metrics = train_logistic_regression()
            tuned_model_information['logreg'] = {
                'model': best_model,
                'best_model_params': best_params,
                'metrics': val_metrics
            }
        
        if args.model in ['xgb', 'all']:
            best_model, best_params, val_metrics = train_xgb_model()
            tuned_model_information['xgb'] = {
                'model': best_model,
                'best_model_params': best_params,
                'metrics': val_metrics
            }
            
        if args.model in ['lstm', 'all']:
            best_model, best_params, val_metrics, train_loss, val_loss = train_lstm_models(device)
            tuned_model_information['lstm'] = {
                'model': best_model,
                'best_model_params': best_params,
                'metrics': val_metrics,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
        
        save_result.save_tuning(tuned_model_information)
        
        
    if args.test_best_model:
        best_model_input = input('Enter the model you want to test (logreg, xgb, lstm): ')
        if best_model_input not in ['logreg', 'xgb', 'lstm']:
            raise ValueError(f'Invalid model name: {best_model_input}. Choose from logreg, xgb, lstm.')
        best_model = retrieve_final_model(os.path.join(ROOT_DIR, 'project/result/tuning_results', best_model_input))
        
        print(f'The best choosen model is: {best_model_input} '
            f'with validation accuracy: {best_model['accuracy']:.4f}, '
            f'precision: {best_model['precision']:.4f}, '
            f'recall: {best_model['recall']:.4f}, '
            f'f1: {best_model['f1']:.4f}')
        print(f'Best model parameters: {best_model['best_model_params']}')
        
        tuning_path = os.path.join(ROOT_DIR, 'project/result/tuning_results')
        if best_model_input == 'logreg':
            best_logreg_model = logreg.load_model(os.path.join(tuning_path, 'logreg', 'best_model.joblib'))
            metrics = logreg.evaluate(best_logreg_model, label='Test')
            save_result.save_best_model(model_name='logreg', model=best_logreg_model, metrics=metrics)
            
            
        elif best_model_input == 'xgb':
            best_xgb_model = xgb.load_model(os.path.join(tuning_path, 'xgb', 'best_model.joblib'))
            metrics = xgb.evaluate(best_xgb_model, label='Test')
            save_result.save_best_model(model_name='xgb', model=best_xgb_model, metrics=metrics)
            
            
        elif best_model_input == 'lstm':
            model = torch.load(os.path.join(tuning_path, 'lstm', 'best_model.pth'))
            # model.eval() and model.to(device) are done in the evaluate_model function

            test_dataset = make_dataset('test', device)
            test_loader = get_data_loader(test_dataset, batch_size=best_model['best_model_params']['batch_size'])
            test_performance = evaluate_model(model, test_loader, device)
            save_result.save_best_model(model_name='lstm', model=model, metrics=test_performance)
    
    
    torch.cuda.empty_cache()