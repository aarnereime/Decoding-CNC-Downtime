from .unidirectional_lstm import UnidirectionalLSTM
from .bidirectional_lstm import BidirectionalLSTM
from .unidirectional_attention_lstm import UnidirectionalAttentionLSTM
from .bidirectional_attention_lstm import BidirectionalAttentionLSTM
from .cnn_lstm_hybrid_1 import CNNLSTMHybrid
from .cnn_lstm_hybrid_2 import CNNLSTMBidirectionalHybrid
from .MALSTM_FCN import MultivariateAttentionLSTMFCN


MODEL_REGISTRY = {
    'unidirectional_lstm': UnidirectionalLSTM,
    'bidirectional_lstm': BidirectionalLSTM, 
    'unidirectional_attention_lstm': UnidirectionalAttentionLSTM, 
    'bidirectional_attention_lstm': BidirectionalAttentionLSTM, 
    'cnn_lstm_hybrid_1': CNNLSTMHybrid, 
    'cnn_lstm_hybrid_2': CNNLSTMBidirectionalHybrid, 
    'multivariate_attention_lstm_fcn': MultivariateAttentionLSTMFCN 
}


def initialize_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Invalid model name: {model_name}')
    return MODEL_REGISTRY[model_name](**kwargs)