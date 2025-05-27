import torch
from torch.utils.data import WeightedRandomSampler

def initialize_weighted_sampler(dataset):
    
    labels = dataset.labels.cpu().long() 
    
    class_counts = torch.bincount(labels)
    
    num_samples = len(labels)
    num_classes = len(class_counts)
    class_weights = num_samples / (num_classes * class_counts.float())
    
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
    