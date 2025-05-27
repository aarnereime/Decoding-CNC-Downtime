import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionLSTMBlock(nn.Module):
    def __init__(self, lstm_input_dim, hidden_size, num_layers, dropout):
        """
        Parameters:
            lstm_input_dim: Input dimension for the LSTM, which is the number of numerical features + total embedding dimension.
            hidden_size: Hidden size of the LSTM.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
        """
        super(AttentionLSTMBlock, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0, # Dropout is only applied if num_layers > 1
        )
        
        # Attention mechanism components
        # A linear layer to project the LSTM outputs 
        self.attention_layer = nn.Linear(hidden_size, hidden_size)
        # A learnable parameter vector to compute attention scores
        self.attention_vector = nn.Parameter(torch.randn(hidden_size))
        
        self.dropout = nn.Dropout(0.8) # Set to 80% to match paper's dropout rate
        
        
    def forward(self, x, lengths):
        # Pack the sequences for the LSTM using the original lengths (ignores padding in computation)
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_input)
        
        # Unpack the output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Compute attention scores
        # We first apply the attention layer to the LSTM outputs (using a tanh activation because we want the scores to be between -1 and 1)
        attention_scores = torch.tanh(self.attention_layer(output))
        # We then compute the scores by multiplying the attention vector with the tanh output
        attention_scores = torch.matmul(attention_scores, self.attention_vector)
        
        # Creating a mask based on the true lengths to ignore padding
        max_len = output.size(1)
        mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        attention_scores[~mask] = -float('inf')
        
        # Normalize the scores using a softmax operation
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Computes the weighted sum of the LSTM outputs
        context_vector = torch.sum(output * attention_weights.unsqueeze(2), dim=1)
        
        context = self.dropout(context_vector)
        return context


class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels):
        """
        Parameters:
            channels: Number of input channels (features) for the Squeeze-Excite block.
        """
        super(SqueezeExciteBlock, self).__init__()
        
        self.average_pooling = nn.AdaptiveAvgPool1d(1)
        
        reduction_ratio = 16
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch, channels, seq_len]
        batch_size, channel, seq_len = x.shape
        
        # Squeeze operation
        squeeze = self.average_pooling(x).view(batch_size, channel) # shape: [batch, channels]
        
        # Excitation operation
        excite = self.fc(squeeze).view(batch_size, channel, 1) # shape: [batch, channels, 1]
        
        # Scales the input tensor with the learned excitation weights
        scale = x * excite.expand_as(x) # shape: [batch, channels, seq_len]
        return scale


class FCNBlock(nn.Module):
    def __init__(self, in_channels, dropout):
        """
        Parameters:
            in_channels: Number of input channels (features) for the FCN block.
            dropout: Dropout probability.
        """
        super(FCNBlock, self).__init__()
        fcn_filters = [128, 256, 128]
        fcn_kernel_sizes = [8, 5, 3]
        batch_norm_momentum = 0.99
        batch_norm_epsilon = 0.001
        
        layers = []
        current_in_channels = in_channels
        for idx, (filter, kernel) in enumerate(zip(fcn_filters, fcn_kernel_sizes)):
            layers.append(nn.Conv1d(in_channels=current_in_channels, out_channels=filter, kernel_size=kernel, padding=kernel//2))
            layers.append(nn.BatchNorm1d(num_features=filter, eps=batch_norm_epsilon, momentum=batch_norm_momentum))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            if idx < 2:
                layers.append(SqueezeExciteBlock(filter))
            
            current_in_channels = filter
                
        self.conv_layers = nn.Sequential(*layers) # Creates a sequential container with all the layers defined by the loop above
        self.average_pooling = nn.AdaptiveAvgPool1d(1)
        
        
    def forward(self, x):
        """
        Parameters:
            x: Input tensor of shape [batch, in_channels, seq_len].
        """
        out = self.average_pooling(self.conv_layers(x)) # shape: [batch, in_channels, 1]
        return out.squeeze(-1) # shape: [batch, in_channels]


class MultivariateAttentionLSTMFCN(nn.Module):
    def __init__(self, num_numerical_features, categorical_vocab_sizes, categorical_embedding_dims, hidden_size, num_layers, dropout):
        """
        Parameters:
            num_numerical_features: Number of numerical features per time step.
            categorical_vocab_sizes: List of vocabulary sizes for each categorical feature.
            categorical_embedding_dims: List of embedding dimensions for each categorical feature.
            hidden_size: Hidden size of the LSTM.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
        """
        super(MultivariateAttentionLSTMFCN, self).__init__()
        
        # Creating an embedding layer for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dimension)
            for vocab_size, emb_dimension in zip(categorical_vocab_sizes, categorical_embedding_dims)
        ])

        total_emb_dim = sum(categorical_embedding_dims) # Total embedding dimension after concatenating all categorical embeddings
        combined_input_dim = num_numerical_features + total_emb_dim # LSTM input dimension: number of numerical features + total embedding dimension
        
        self.fcn_block = FCNBlock(
            in_channels=combined_input_dim,
            dropout=dropout
        )
        
        self.lstm_attention_block = AttentionLSTMBlock(
            lstm_input_dim=combined_input_dim,
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + 128, 1)
        )
    
    
    def forward(self, x_num, x_cat, lengths):        
        # Embedding layer
        embedded_features = [embed(x_cat[:, :, i]) for i, embed in enumerate(self.embeddings)]
        x_emb = torch.cat(embedded_features, dim=2) # Concatenate all embeddings along the feature dimension: [batch, max_seq_length, total_emb_dim]
        x_combined = torch.cat((x_num, x_emb), dim=2) # Combine numerical features with embeddings: [batch, max_seq_length, num_numerical + total_emb_dim]

        # FCN block expects shape [batch, num_features (channels), seq_len], currently we have shape [batch, seq_len, num_features]
        x_fcn = x_combined.transpose(1, 2) # [batch, num_features, seq_len]
        fcn_out = self.fcn_block(x_fcn)
        
        # LSTM Attention block
        lstm_attn_out = self.lstm_attention_block(x_combined, lengths)
        
        combined = torch.cat([fcn_out, lstm_attn_out], dim=1) # Concatenate the outputs from FCN and LSTM Attention blocks, shape: [batch, hidden_size + 128]
        
        # final classification layer
        output = self.fc(combined)
        return output