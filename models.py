from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

HOMEWORK_DIR = Path(__file__).resolve().parent
class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,  # Increased number of layers
        dropout_rate: float = 0.3,  # Dropout rate
    ):
        """
        Args:
            n_track (int): Number of points in each side of the track
            n_waypoints (int): Number of waypoints to predict
            hidden_dim (int): Number of hidden units in the fully connected layers
            num_layers (int): Number of layers in the MLP
            dropout_rate (float): Dropout rate
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Input dimension: concatenated left and right boundaries (n_track * 4 for x and y of each boundary)
        input_dim = n_track * 4

        # Output dimension: x and y for each predicted waypoint (n_waypoints * 2)
        output_dim = n_waypoints * 2

        # Define layers for the MLP
        layers = []
        current_dim = hidden_dim

        # Input layer
        layers.append(nn.Linear(input_dim, current_dim))
        layers.append(nn.LayerNorm(current_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(p=dropout_rate))

        # Hidden layers with residual connections
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, current_dim))
            layers.append(nn.LayerNorm(current_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout_rate))

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self.mlp.apply(self.init_weights)

        # Convert INPUT_MEAN and INPUT_STD to torch.Tensor
        input_mean = torch.tensor([0.2788, 0.2657], dtype=torch.float32)
        input_std = torch.tensor([0.2064, 0.1944], dtype=torch.float32)

        # Register mean and std as buffers
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.size(0)

        # Normalize the input
        track_left = (track_left - self.input_mean[None, None, :]) / self.input_std[None, None, :]
        track_right = (track_right - self.input_mean[None, None, :]) / self.input_std[None, None, :]

        # Flatten and concatenate left and right boundaries
        track_left = track_left.view(batch_size, -1)  # (b, n_track * 2)
        track_right = track_right.view(batch_size, -1)  # (b, n_track * 2)
        features = torch.cat([track_left, track_right], dim=1)  # (b, n_track * 4)

        # Pass through the MLP
        waypoints = self.mlp(features)  # (b, n_waypoints * 2)

        # Reshape to (b, n_waypoints, 2)
        waypoints = waypoints.view(batch_size, self.n_waypoints, 2)

        return waypoints

    @staticmethod
    def init_weights(m):
        """
        Initialize weights of the model.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)



class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 96,
        n_heads: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Learnable query embeddings for waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Linear projection to transform track inputs into d_model dimensionality
        self.input_proj = nn.Linear(2, d_model)

        # Learnable positional encodings for the concatenated track sequence
        self.positional_encoding = nn.Embedding(n_track * 2, d_model)

        # Multihead attention layer
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)

        # Output projection to (x, y) for each waypoint
        self.output_proj = nn.Linear(d_model, 2)

        # Convert INPUT_MEAN and INPUT_STD to torch.Tensor
        input_mean = torch.tensor([0.2788, 0.2657], dtype=torch.float32)
        input_std = torch.tensor([0.2064, 0.1944], dtype=torch.float32)

        # Register mean and std as buffers
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)

    def forward(
            self,
            track_left: torch.Tensor,
            track_right: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.size(0)

        # Normalize the input
        track_left = (track_left - self.input_mean[None, None, :]) / self.input_std[None, None, :]
        track_right = (track_right - self.input_mean[None, None, :]) / self.input_std[None, None, :]

        # Concatenate left and right track boundaries
        track = torch.cat([track_left, track_right], dim=1)  # Shape: (b, n_track * 2, 2)

        # Project input to d_model
        track_encoded = self.input_proj(track)  # Shape: (b, n_track * 2, d_model)

        # Add positional encoding
        positions = torch.arange(track_encoded.size(1), device=track_encoded.device)  # Shape: (n_track * 2,)
        positional_encodings = self.positional_encoding(positions)  # Shape: (n_track * 2, d_model)
        positional_encodings = positional_encodings.unsqueeze(0).expand(batch_size, -1,
                                                                        -1)  # Shape: (b, n_track * 2, d_model)
        track_encoded += positional_encodings

        # Prepare input for multihead attention (n_track * 2 as sequence length)
        track_encoded = track_encoded.permute(1, 0, 2)  # Shape: (n_track * 2, b, d_model)

        # Query embeddings for waypoints
        query = self.query_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)  # Shape: (n_waypoints, b, d_model)

        # Multihead attention
        attended_features, _ = self.attention(query, track_encoded, track_encoded)  # Shape: (n_waypoints, b, d_model)

        # Project attended outputs to (x, y)
        waypoints = self.output_proj(attended_features)  # Shape: (n_waypoints, b, 2)

        # Permute back to (b, n_waypoints, 2)
        waypoints = waypoints.permute(1, 0, 2)

        return waypoints


class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints: int = 3):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Input Convolutional layer to increase channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual block 1
        self.res_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
        )

        # Convolutional layer with increased channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Residual block 2
        self.res_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
        )

        # Convolutional layer with increased channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Residual block 3
        self.res_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Final linear layer to predict waypoints
        self.fc = nn.Linear(256, n_waypoints * 2)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Normalize the input
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # First convolution and residual block
        x = F.relu(self.bn1(self.conv1(x)))
        residual = x
        x = self.res_block1(x) + residual  # Residual connection 1

        # Second convolution and residual block
        x = F.relu(self.bn2(self.conv2(x)))
        residual = x
        x = self.res_block2(x) + residual  # Residual connection 2

        # Third convolution and residual block
        x = F.relu(self.bn3(self.conv3(x)))
        residual = x
        x = self.res_block3(x) + residual  # Residual connection 3

        # Global Average Pooling
        x = self.global_avg_pool(x).squeeze(-1).squeeze(-1)  # Shape: (b, 256)

        # Dropout for regularization
        x = self.dropout(x)

        # Linear layer to predict waypoints
        x = self.fc(x)  # Shape: (b, n_waypoints * 2)

        # Reshape to (b, n_waypoints, 2)
        waypoints = x.view(-1, self.n_waypoints, 2)

        return waypoints




MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
