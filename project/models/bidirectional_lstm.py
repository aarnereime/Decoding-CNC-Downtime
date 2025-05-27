import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from project.utils.seed import set_seed

set_seed(42)


class BidirectionalLSTM(nn.Module):
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
        super(BidirectionalLSTM, self).__init__()
        
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
            bidirectional=True # Set bidirectional=True to use a bidirectional LSTM
        )
        
        # Final layer accounts for 2x hidden_size (forward + backward) when using a bidirectional LSTM
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 1)
        )
        
    
    def forward(self, x_num, x_cat, lengths):
        """
        Parameters:
            x_num: Tensor of shape [batch, max_batch_seq_length, num_numerical_features].
            x_cat: Tensor of shape [batch, max_batch_seq_length, num_categorical_features].
            lengths: Tensor of original sequence lengths for each batch item.
        """
        embedded_features = [embed(x_cat[:, :, i]) for i, embed in enumerate(self.embeddings)] # Create embeddings for each categorical feature
        x_emb = torch.cat(embedded_features, dim=2) # Concatenate all embeddings along the feature dimension: [batch, max_seq_length, total_emb_dim]
        x_combined = torch.cat((x_num, x_emb), dim=2) # Combine numerical features with embeddings: [batch, max_seq_length, num_numerical + total_emb_dim]
        
        # Pack the sequences for the LSTM using the original lengths (ignores padding in computation)
        packed_input = pack_padded_sequence(x_combined, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_input)
        
        # Pytroch states that the hidden state will contain a concatenation of the final forward and reverse hidden states, respectively,
        # just like in the packed_output tensor. But i think its better to say that the forward and reverse hidden state are interleaved
        # by layer, not all forwards first and then all backwards. So the first layer of the num_layers * num_directions dimension will be
        # the first layer of the forward hidden states, and the second layer will be the first layer of the reverse hidden states, and so on.
        final_forward = hidden_state[-2] # Second-to-last element = last forward layer
        final_backward = hidden_state[-1] # Last element = last backward layer
        
        # Concatenate the final hidden states from both directions
        final_hidden = torch.cat((final_forward, final_backward), dim=1) # shape: [batch, hidden_size * 2]
        
        # Apply the final classification layer
        output = self.fc(final_hidden) # shape: [batch, 1]
        return output
        
        
        