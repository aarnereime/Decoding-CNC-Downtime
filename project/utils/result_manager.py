from project.config.root_dir import ROOT_DIR
import os
import json
import torch
import joblib


class SaveResult:
    
    def __init__(self):
        self.best_model_dir = os.path.join(ROOT_DIR, 'project/result/best_model')
        self.tuning_dir = os.path.join(ROOT_DIR, 'project/result/tuning_results')

            
    def make_tuning_dir(self, tuning_information):
        os.makedirs(self.tuning_dir, exist_ok=True)
        
        # making subdirectories for each key in tuning_information
        for key in tuning_information.keys():
            os.makedirs(os.path.join(self.tuning_dir, key), exist_ok=True)
            
            
    def unpack_metrics(self, metrics):
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        
        fpr = metrics['fpr'].tolist()
        tpr = metrics['tpr'].tolist()
        roc_auc = float(metrics['roc_auc'])
        cm = metrics.get('cm', None)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'cm': cm
        }


    def save_tuning(self, tuning_information):
        self.make_tuning_dir(tuning_information)
        
        for key, value in tuning_information.items():
            if isinstance(value['model'], torch.nn.Module):
                model_path = os.path.join(self.tuning_dir, key, 'best_model.pth')
                torch.save(value['model'].state_dict(), model_path)
            else:
                model_path = os.path.join(self.tuning_dir, key, 'best_model.joblib')
                joblib.dump(value['model'], model_path)
                print(f'Saved best model to: {model_path}')
            
            stats = {
                'best_model_params': value['best_model_params'],
                **self.unpack_metrics(value['metrics'])
            }
            
            if key == 'lstm':
                stats.update({
                    'train_loss': value['train_loss'],
                    'val_loss': value['val_loss']
                })
                
            with open(os.path.join(self.tuning_dir, key, 'stats.json'), 'w') as f:
                json.dump(stats, f, indent=4)
            
        print(f'Saved tuning information to: {self.tuning_dir}')
        
        
    def make_best_model_dir(self, model_name):
        os.makedirs(self.best_model_dir, exist_ok=True)
        
        # making subdirectory for the best model
        os.makedirs(os.path.join(self.best_model_dir, model_name), exist_ok=True)
        
        
    def save_best_model(self, model_name, model, metrics):
        self.make_best_model_dir(model_name)
        
        # Save the model
        if isinstance(model, torch.nn.Module):
            model_path = os.path.join(self.best_model_dir, model_name, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
        else:
            model_path = os.path.join(self.best_model_dir, model_name, 'best_model.joblib')
            joblib.dump(model, model_path)
            print(f'Saved best model to: {model_path}')
        
        # Save the metrics
        metrics = {
            **self.unpack_metrics(metrics)
        }
        metrics_path = os.path.join(self.best_model_dir, model_name, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f'Saved best model and metrics to: {self.best_model_dir}')
        
    
    