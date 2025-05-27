import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from project.utils.seed import set_seed

set_seed(42)


class UnidirectionalLSTM(nn.Module):
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
        super(UnidirectionalLSTM, self).__init__()
        
        # Creating an embedding layer for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dimension)
            for vocab_size, emb_dimension in zip(categorical_vocab_sizes, categorical_embedding_dims)
        ])

        total_emb_dim = sum(categorical_embedding_dims) # Total embedding dimension after concatenating all categorical embeddings
        lstm_input_dim = num_numerical_features + total_emb_dim # LSTM input dimension: number of numerical features + total embedding dimension
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0, # Dropout is only applied if num_layers > 1
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        
    def forward(self, x_num, x_cat, lengths):
        """
        Parameters:
            x_num: Tensor of shape [batch, max_batch_seq_length, num_numerical_features].
            x_cat: Tensor of shape [batch, max_batch_seq_length, num_categorical_features].
            lengths: Tensor of original sequence lengths for each batch item.
        """
        embedded_features = [embed(x_cat[:, :, i]) for i, embed in enumerate(self.embeddings)] # Create embeddings for each categorical feature
        x_emb = torch.cat(embedded_features, dim=2) # Concatenate all embeddings along the feature dimension, shape: [batch, max_seq_length, total_emb_dim]
        x_combined = torch.cat((x_num, x_emb), dim=2) # Combine numerical features with embeddings, shape: [batch, max_seq_length, num_numerical + total_emb_dim]
        
        # Pack the sequences for the LSTM using the original lengths (ignores padding in computation)
        packed_input = pack_padded_sequence(x_combined, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_input)

        # Extract the final hidden state of the LSTM
        final_hidden = hidden_state[-1]  # shape: [batch, hidden_size]
        
        # Apply the final classification layer
        output = self.fc(final_hidden) # shape: [batch, 1]
        return output