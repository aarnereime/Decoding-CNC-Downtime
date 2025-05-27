import torch
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier,  plot_importance

root = os.getcwd()

class XGBModel:
    
    def __init__(self):
        self.data_dict = torch.load(os.path.join(root, 'project/data/logistic_xgboost_data.pt'))
        
        self.X_train = self.data_dict['X_train']
        self.y_train = self.data_dict['y_train']
        self.X_val = self.data_dict['X_val']
        self.y_val = self.data_dict['y_val']
        self.X_test = self.data_dict['X_test']
        self.y_test = self.data_dict['y_test']
        self.feature_names = self.data_dict['feature_names']
        
        
    def calculate_pos_weight(self):
        pos_class_weight = (len(self.y_train) - np.sum(self.y_train)) / np.sum(self.y_train)
        return pos_class_weight
        
        
    def make_pipeline(self):
        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=self.calculate_pos_weight(),
            random_state=42
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgboost', model)
        ])
        return pipeline
    

    def tune_model(self, pipeline):
        param_dist = {
            'xgboost__eta': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
            'xgboost__gamma': [0, 0.1, 0.2, 0.3],
            'xgboost__n_estimators': [50, 100, 200, 300],
            'xgboost__max_depth': [3, 4, 5, 6],
            'xgboost__subsample': [0.6, 0.8, 1.0],
            'xgboost__colsample_bytree': [0.6, 0.8, 1.0],
            'xgboost__min_child_weight': [1, 3, 5, 10],
            'xgboost__lambda': [1, 2, 5, 7, 10],
            'xgboost__alpha': [1, 2, 5, 10]
        }
        
        random_search = RandomizedSearchCV(
            estimator=pipeline, 
            param_distributions=param_dist, 
            n_iter=5000, 
            scoring='accuracy',
            cv=5, 
            n_jobs=-1, 
            random_state=42,
            verbose=1
        )
        random_search.fit(self.X_train, self.y_train)
        return random_search.best_estimator_, random_search.best_params_
    

    def evaluate(self, pipeline, label):
        assert label in ['Validation', 'Test'], "Label must be either 'Validation' or 'Test'"
        if label == 'Validation':
            X, y = self.X_val, self.y_val
        elif label == 'Test':
            X, y = self.X_test, self.y_test
            
        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        fpr, tpr, _ = roc_curve(y, y_proba)
        metrics.update({
            'fpr': fpr,
            'tpr': tpr
        })
        
        # Confusion matrix for test set (will crash the code if )
        if label == 'Test':
            cm = confusion_matrix(y, y_pred)
            metrics.update({
                'cm': cm.tolist()
            })
            self.plot_feature_importance(pipeline)
        
        for key, value in metrics.items():
            if key not in ('fpr', 'tpr', 'cm'):
                print(f'{label} {key.capitalize()}: {value:.4f}')
        return metrics
    

    def load_model(self, model_path):
        return joblib.load(model_path)

        
    def plot_feature_importance(self, pipeline, importance_type='gain', top_n=20):
        booster = pipeline.named_steps['xgboost'].get_booster()
        score_dict = booster.get_score(importance_type=importance_type)

        data = []
        for fid, score in score_dict.items():
            idx = int(fid.lstrip('f'))
            name = self.feature_names[idx] if idx < len(self.feature_names) else fid
            data.append((name, score))
            
        imp_df = pd.DataFrame(data, columns=['Feature', 'Importance'])
        imp_df = imp_df.sort_values('Importance', ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        plt.barh(imp_df['Feature'], imp_df['Importance'])
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} feature importances ({importance_type})')
        plt.tight_layout()
        save_path = os.path.join(root, 'project/xgb_feature_importance.png')
        plt.savefig(save_path)