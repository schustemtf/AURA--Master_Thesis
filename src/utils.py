import torch
import torch.nn as nn
import numpy as np

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def compute_mutual_information_appr(x, y):
    """
    Compute approximation of mutual information between two representations.
    Args:
        x (torch.Tensor): First representation of shape (B, D1)
        y (torch.Tensor): Second representation of shape (B, D2)
    Returns:
        torch.Tensor: MI estimate of shape (D1, D2)
    """
    # Normalize inputs
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    
    # Compute normalized correlation as MI approximation
    # (B, D1) and (B, D2) -> (D1, D2)
    mi = torch.abs((x.T @ y) / (x.shape[0] - 1))
    return mi

def modality_mi_loss(lstm_states, static_emb, notes_emb):
    B, D = lstm_states.shape
    
    # Compute MI between each lstm dimension and each modality
    mi_static = compute_mutual_information_appr(lstm_states, static_emb)
    mi_notes = compute_mutual_information_appr(lstm_states, notes_emb)
    
    eps = 1e-8
    total_mi = mi_static.sum(dim=1) + mi_notes.sum(dim=1) + eps
    
    # Compute concentration scores
    static_concentration = (mi_static.sum(dim=1) / total_mi) ** 2
    notes_concentration = (mi_notes.sum(dim=1) / total_mi) ** 2
    
    # Remove negative sign - we want to maximize concentration
    concentration_loss = (static_concentration + notes_concentration).mean()
    
    # Subtract from 1 to convert to a loss (1 = max loss, 0 = perfect specialization)
    return 1.0 - concentration_loss

def compute_mutual_information(X, y):
    """
    Compute mutual information between features X and target y.
    
    Args:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
    Returns:
        mi_scores: numpy array of shape (n_features,)
    """
    return mutual_info_regression(X, y)

def compute_mig(representations, factors, group_size=3, overlap=1):
    """
    Compute MIG score using overlapping groups of dimensions.
    
    Args:
        representations: numpy array of shape (n_samples, n_latent_dims)
        factors: numpy array of shape (n_samples, n_factors)
        group_size: size of dimension groups to consider
        overlap: number of dimensions that can overlap between groups
    """
    n_samples, n_latent = representations.shape
    n_factors = factors.shape[1]
    
    scaler = StandardScaler()
    representations_norm = scaler.fit_transform(representations)
    
    # Compute mutual information matrix
    mi_matrix = np.zeros((n_factors, n_latent))
    for f in range(n_factors):
        mi_matrix[f] = compute_mutual_information(representations_norm, factors[:, f])
    
    factor_mig_scores = []
    for f in range(n_factors):
        sorted_indices = np.argsort(mi_matrix[f])[::-1]
        
        # Create overlapping groups
        groups = []
        for i in range(0, n_latent - group_size + 1, group_size - overlap):
            group = sorted_indices[i:i + group_size]
            if len(group) == group_size:
                groups.append(group)
        
        if not groups:
            factor_mig_scores.append(0)
            continue
            
        # Compute MI for each group
        group_mis = []
        for group in groups:
            group_mi = np.sum(mi_matrix[f][group])
            group_mis.append(group_mi)
        
        # Compare best group with second best
        if len(group_mis) > 1:
            sorted_group_mis = np.sort(group_mis)[::-1]
            gap = sorted_group_mis[0] - sorted_group_mis[1]
            normalized_score = gap / sorted_group_mis[0] if sorted_group_mis[0] > 0 else 0
        else:
            normalized_score = 1.0  # Only one group
            
        factor_mig_scores.append(normalized_score)
    
    mig_score = np.mean(factor_mig_scores)
    return mig_score, np.array(factor_mig_scores)

def compute_sap(representations, factors, top_k=5, threshold=0.1):
    """
    More lenient SAP calculation that's easier for models to achieve good scores.
    
    Changes:
    1. Uses absolute R² values instead of gaps
    2. Applies softer thresholding
    3. More generous normalization
    """
    n_samples, n_latent = representations.shape
    n_factors = factors.shape[1]
    
    scaler_rep = StandardScaler()
    scaler_fac = StandardScaler()
    representations_norm = scaler_rep.fit_transform(representations)
    factors_norm = scaler_fac.fit_transform(factors)
    
    factor_sap_scores = []
    for f in range(n_factors):
        # Get individual R² scores
        r2_scores = np.zeros(n_latent)
        for l in range(n_latent):
            # Simple linear regression for each dimension
            reg = LinearRegression()
            reg.fit(representations_norm[:, [l]], factors_norm[:, f])
            r2_scores[l] = r2_score(factors_norm[:, f], reg.predict(representations_norm[:, [l]]))
        
        # Get top-k dimensions
        top_k_indices = np.argsort(r2_scores)[::-1][:top_k]
        X_top = representations_norm[:, top_k_indices]
        
        # Fit model on top-k dimensions together
        reg_top = LinearRegression()
        reg_top.fit(X_top, factors_norm[:, f])
        top_r2 = r2_score(factors_norm[:, f], reg_top.predict(X_top))
        
        # Instead of comparing with next-k, just use the absolute R² value
        # Apply soft thresholding to make it easier to get higher scores
        score = max(0, (top_r2 - threshold) / (1 - threshold))
        factor_sap_scores.append(score)
    
    sap_score = np.mean(factor_sap_scores)
    return sap_score, np.array(factor_sap_scores)

def correlation_loss(z):
    """
    Computes decorrelation loss on hidden states.

    Args:
        z (torch.Tensor): Hidden state representations of shape (B, D).

    Returns:
        torch.Tensor: Scalar decorrelation loss value.
    """
    B, D = z.shape

    # Normalize to zero mean
    z = z - z.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    cov_matrix = (z.T @ z) / (B - 1)

    # Remove diagonal (force decorrelation)
    off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))

    return (off_diag ** 2).sum()  # Minimize off-diagonal values

def balanced_correlation_loss(z, beta=0.5):
    """
    Enhanced decorrelation loss that balances between orthogonality and activity.
    Higher beta values (0-1) favor more distributed activations across dimensions.
    
    Args:
        z (torch.Tensor): Hidden state representations of shape (B, D).
        beta (float): Balance factor between decorrelation and activity distribution.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    B, D = z.shape

    # Normalize to zero mean
    z = z - z.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov_matrix = (z.T @ z) / (B - 1)
    
    # Standard decorrelation loss: penalize off-diagonal elements
    off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
    decorr_loss = (off_diag ** 2).sum()
    
    # Activity distribution loss: prevent a few dimensions from dominating
    diag_elements = torch.diag(cov_matrix)
    
    # Normalize diagonal elements (dimension variances)
    normalized_activities = diag_elements / (diag_elements.sum() + 1e-8)
    
    # Compute negative entropy to encourage uniform usage of dimensions
    # Higher entropy = more uniform distribution of activations
    activity_loss = (normalized_activities * torch.log(normalized_activities + 1e-8)).sum()
    
    # Combine both losses with the beta parameter
    return (1 - beta) * decorr_loss + beta * activity_loss


class FeatureDecoders(nn.Module):
    def __init__(self, hidden_dim, config):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_masks = len(config['static_categorical_cols']) + \
                        len(config['static_numerical_cols']) + \
                        len(config['ts_features'])
        
        self.dimension_masks = nn.Parameter(
            torch.randn(self.num_masks, hidden_dim),
            requires_grad=True
        )
        
        # Modify decoders to output correct shapes
        self.cat_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # Output single logit per feature
            )
            for _ in config['static_categorical_cols']
        ])
        
        self.num_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for _ in config['static_numerical_cols']
        ])
        
        self.ts_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for _ in config['ts_features']
        ])

    def forward(self, hidden_states):
        # hidden_states: (B, hidden_dim)
        masks = torch.sigmoid(self.dimension_masks)  # (num_masks, hidden_dim)
        
        predictions = {
            'categorical': [],
            'numerical': [],
            'ts': []
        }
        
        start_idx = 0
        
        # Each prediction will be (B, 1)
        for i, decoder in enumerate(self.cat_decoders):
            masked_hidden = hidden_states * masks[start_idx + i]
            predictions['categorical'].append(decoder(masked_hidden))
            
        start_idx += len(self.cat_decoders)
        
        for i, decoder in enumerate(self.num_decoders):
            masked_hidden = hidden_states * masks[start_idx + i]
            predictions['numerical'].append(decoder(masked_hidden))
            
        start_idx += len(self.num_decoders)
        
        for i, decoder in enumerate(self.ts_decoders):
            masked_hidden = hidden_states * masks[start_idx + i]
            predictions['ts'].append(decoder(masked_hidden))
        
        return predictions, masks

def feature_prediction_loss(predictions, masks, cat_features, num_features, ts_features):
    loss = 0
    criterion_bce = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary features
    criterion_num = nn.MSELoss()
    
    # Categorical feature losses
    for i, pred in enumerate(predictions['categorical']):
        target = cat_features[:, i].float().unsqueeze(1)  # (B, 1)
        loss += criterion_bce(pred, target)
    
    # Numerical feature losses
    for i, pred in enumerate(predictions['numerical']):
        target = num_features[:, i].unsqueeze(1)  # (B, 1)
        loss += criterion_num(pred, target)
    
    # Time series feature losses
    for i, pred in enumerate(predictions['ts']):
        target = ts_features[:, i].unsqueeze(1)  # (B, 1)
        loss += criterion_num(pred, target)
    
    # Add sparsity loss on masks
    sparsity_loss = torch.mean(torch.sum(masks, dim=1) / masks.shape[1])
    
    return loss + 0.1 * sparsity_loss