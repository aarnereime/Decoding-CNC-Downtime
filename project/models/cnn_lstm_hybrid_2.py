import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from project.utils.seed import set_seed

set_seed(42)


class CNNLSTMBidirectionalHybrid(nn.Module):
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
        super(CNNLSTMBidirectionalHybrid, self).__init__()
        
        # Creating an embedding layer for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dimension)
            for vocab_size, emb_dimension in zip(categorical_vocab_sizes, categorical_embedding_dims)
        ])

        total_emb_dim = sum(categorical_embedding_dims) # Total embedding dimension after concatenating all categorical embeddings
        combined_input_dim = num_numerical_features + total_emb_dim # LSTM input dimension: number of numerical features + total embedding dimension        
        
        cnn_kernel_size = 3

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=combined_input_dim, 
                out_channels=hidden_size, 
                kernel_size=cnn_kernel_size, 
                padding=cnn_kernel_size // 2
            ),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=cnn_kernel_size,
                padding=cnn_kernel_size // 2
            ),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM that will process the CNN output
        # The LSTM input dimension will be 'hidden_size' (from the CNN output channels)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0, # Dropout is only applied if num_layers > 1
            bidirectional=True
        )
        
        # self.norm = nn.LayerNorm(hidden_size * 2) # Layer normalization on the pooled representation
        self.fc = nn.Linear(hidden_size * 2, 1) # Final classification layer (using the final hidden state)
        
    
    def forward(self, x_num, x_cat, lengths):
        """
        Parameters:
            x_num: Tensor of shape [batch, max_batch_seq_length, num_numerical_features].
            x_cat: Tensor of shape [batch, max_batch_seq_length, num_categorical_features].
            lengths: Tensor of original sequence lengths for each batch item.
        """
        embedded_features = [embed(x_cat[:, :, i]) for i, embed in enumerate(self.embeddings)]
        x_emb = torch.cat(embedded_features, dim=2) # Concatenate all embeddings along the feature dimension: [batch, max_seq_length, total_emb_dim]
        x_combined = torch.cat((x_num, x_emb), dim=2) # Combine numerical features with embeddings: [batch, max_seq_length, num_numerical + total_emb_dim]
        
        # For CNN, transpose to shape: [batch, channels, seq_len]
        x_cnn = x_combined.transpose(1, 2) # shape: [batch, combined_input_dim, max_seq_length]
        
        # Apply the stacked convolutional layers
        cnn_out = self.conv_layers(x_cnn) # shape: [batch, hidden_size, max_seq_length]
        
        # Transpose back to LSTM input shape: [batch, seq_len, hidden_size]
        lstm_input = cnn_out.transpose(1, 2) # shape: [batch, max_seq_length, hidden_size]
        
        # Pack the sequences for the LSTM using the original lengths (ignores padding in computation)
        packed_input = pack_padded_sequence(lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_input)
        
        final_forward = hidden_state[-2] # Second-to-last element = last forward layer
        final_backward = hidden_state[-1] # Last element = last backward layer
        
        # Concatenate the final hidden states from both directions
        final_hidden = torch.cat((final_forward, final_backward), dim=1)
        
        output = self.fc(final_hidden)
        return output
    