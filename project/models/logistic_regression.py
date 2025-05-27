import os
import torch
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

root = os.getcwd()

class LogisticRegressionModel:
    
    def __init__(self):
        self.data_dict = torch.load(os.path.join(root, 'project/data/logistic_xgboost_data.pt'))
        
        self.X_train = self.data_dict['X_train']
        self.y_train = self.data_dict['y_train']
        self.X_val = self.data_dict['X_val']
        self.y_val = self.data_dict['y_val']
        self.X_test = self.data_dict['X_test']
        self.y_test = self.data_dict['y_test']
        self.feature_names = self.data_dict['feature_names']
        
    
    def make_pipeline(self):
        model = LogisticRegression(
            random_state=42,
            max_iter=800,
            class_weight='balanced'
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', model)
        ])
        return pipeline
    
    
    def tune_model(self, pipeline):
        params_grid = [
            {
                'logreg__penalty': ['l2'],
                'logreg__tol': [1e-5, 1e-4, 1e-3],
                'logreg__C': [0.0001, 0.01, 0.1, 1, 10],
                'logreg__solver': ['lbfgs', 'newton-cg']
            },
            {
                'logreg__penalty': ['l1'],
                'logreg__tol': [1e-5, 1e-4, 1e-3],
                'logreg__C': [0.0001, 0.01, 0.1, 1, 10],
                'logreg__solver': ['liblinear']
            }
        ]
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=params_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='accuracy'
        )
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_estimator_, grid_search.best_params_
    
    
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
        
        for key, value in metrics.items():
            if key not in ('fpr', 'tpr'):
                print(f'{label} {key.capitalize()}: {value:.4f}')
        return metrics
    
    
    def load_model(path):
        return joblib.load(path)