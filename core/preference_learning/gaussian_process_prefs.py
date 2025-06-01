import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import math
import numpy as np

from reward_models.base_reward_model import BaseRewardModel, RewardOutput

class GaussianProcessPreferences(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        kernel_type: str = "rbf",
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise_variance: float = 0.1,
        inducing_points: Optional[torch.Tensor] = None,
        num_inducing: int = 100
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.kernel_type = kernel_type
        self.num_inducing = num_inducing
        
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(lengthscale)))
        self.log_variance = nn.Parameter(torch.log(torch.tensor(variance)))
        self.log_noise_variance = nn.Parameter(torch.log(torch.tensor(noise_variance)))
        
        if inducing_points is not None:
            self.register_buffer('inducing_points', inducing_points)
        else:
            self.inducing_points = nn.Parameter(torch.randn(num_inducing, feature_dim) * 0.1)
        
        self.variational_mean = nn.Parameter(torch.zeros(num_inducing))
        self.variational_logvar = nn.Parameter(torch.zeros(num_inducing))
        
        self.kernel = self._create_kernel()
    
    def _create_kernel(self) -> Callable:
        if self.kernel_type == "rbf":
            return self._rbf_kernel
        elif self.kernel_type == "matern32":
            return self._matern32_kernel
        elif self.kernel_type == "matern52":
            return self._matern52_kernel
        elif self.kernel_type == "linear":
            return self._linear_kernel
        elif self.kernel_type == "polynomial":
            return self._polynomial_kernel
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        lengthscale = torch.exp(self.log_lengthscale)
        variance = torch.exp(self.log_variance)
        
        dist_sq = torch.cdist(x1 / lengthscale, x2 / lengthscale)**2
        return variance * torch.exp(-0.5 * dist_sq)
    
    def _matern32_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        lengthscale = torch.exp(self.log_lengthscale)
        variance = torch.exp(self.log_variance)
        
        dist = torch.cdist(x1, x2) / lengthscale
        sqrt3_dist = math.sqrt(3) * dist
        
        return variance * (1 + sqrt3_dist) * torch.exp(-sqrt3_dist)
    
    def _matern52_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        lengthscale = torch.exp(self.log_lengthscale)
        variance = torch.exp(self.log_variance)
        
        dist = torch.cdist(x1, x2) / lengthscale
        sqrt5_dist = math.sqrt(5) * dist
        
        return variance * (1 + sqrt5_dist + (5/3) * dist**2) * torch.exp(-sqrt5_dist)
    
    def _linear_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        variance = torch.exp(self.log_variance)
        return variance * torch.matmul(x1, x2.transpose(-1, -2))
    
    def _polynomial_kernel(self, x1: torch.Tensor, x2: torch.Tensor, degree: int = 2) -> torch.Tensor:
        variance = torch.exp(self.log_variance)
        return variance * (torch.matmul(x1, x2.transpose(-1, -2)) + 1)**degree
    
    def forward(
        self,
        x_new: torch.Tensor,
        return_variance: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        K_uu = self.kernel(self.inducing_points, self.inducing_points)
        K_uu += torch.eye(self.num_inducing, device=K_uu.device) * 1e-6
        
        K_uf = self.kernel(self.inducing_points, x_new)
        K_ff_diag = torch.diagonal(self.kernel(x_new, x_new), dim1=-2, dim2=-1)
        
        L_uu = torch.linalg.cholesky(K_uu)
        A = torch.linalg.solve_triangular(L_uu, K_uf, upper=False)
        
        variational_cov = torch.diag(torch.exp(self.variational_logvar))
        
        mean = torch.matmul(A.transpose(-1, -2), self.variational_mean)
        
        if return_variance:
            var_contrib = torch.sum(A**2, dim=-2)
            quad_form = torch.matmul(A.transpose(-1, -2), torch.matmul(variational_cov, A))
            variance = K_ff_diag - var_contrib + torch.diagonal(quad_form, dim1=-2, dim2=-1)
            return mean, variance
        
        return mean, None
    
    def compute_preference_probability(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> torch.Tensor:
        f1_mean, f1_var = self.forward(x1, return_variance=True)
        f2_mean, f2_var = self.forward(x2, return_variance=True)
        
        diff_mean = f1_mean - f2_mean
        diff_var = f1_var + f2_var
        
        noise_var = torch.exp(self.log_noise_variance)
        total_var = diff_var + 2 * noise_var
        
        z = diff_mean / torch.sqrt(total_var + 1e-8)
        return torch.sigmoid(z)
    
    def elbo_loss(
        self,
        preference_pairs: List[Tuple[torch.Tensor, torch.Tensor, float]]
    ) -> torch.Tensor:
        kl_divergence = self._compute_kl_divergence()
        
        log_likelihood = 0.0
        for x1, x2, preference in preference_pairs:
            prob = self.compute_preference_probability(x1, x2)
            log_likelihood += preference * torch.log(prob + 1e-8) + (1 - preference) * torch.log(1 - prob + 1e-8)
        
        log_likelihood = log_likelihood / len(preference_pairs)
        
        return -log_likelihood + kl_divergence
    
    def _compute_kl_divergence(self) -> torch.Tensor:
        K_uu = self.kernel(self.inducing_points, self.inducing_points)
        K_uu += torch.eye(self.num_inducing, device=K_uu.device) * 1e-6
        
        L_uu = torch.linalg.cholesky(K_uu)
        
        variational_cov = torch.diag(torch.exp(self.variational_logvar))
        
        # KL[q(u) || p(u)]
        logdet_K = 2 * torch.sum(torch.log(torch.diag(L_uu)))
        logdet_S = torch.sum(self.variational_logvar)
        
        K_inv_m = torch.linalg.solve(K_uu, self.variational_mean)
        trace_term = torch.trace(torch.linalg.solve(K_uu, variational_cov))
        quad_term = torch.dot(self.variational_mean, K_inv_m)
        
        kl = 0.5 * (logdet_K - logdet_S - self.num_inducing + trace_term + quad_term)
        
        return kl

class NeuralGaussianProcess(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feature_dim: int = 64,
        num_inducing: int = 100
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, feature_dim))
        
        self.feature_extractor = nn.Sequential(*layers)
        self.gp = GaussianProcessPreferences(feature_dim, num_inducing=num_inducing)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        return self.gp(features)
    
    def compute_preference_probability(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        features1 = self.feature_extractor(x1)
        features2 = self.feature_extractor(x2)
        return self.gp.compute_preference_probability(features1, features2)

class MultiOutputGP(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_outputs: int,
        num_inducing: int = 100,
        rank: int = 5
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_outputs = num_outputs
        self.rank = rank
        
        self.base_gp = GaussianProcessPreferences(feature_dim, num_inducing=num_inducing)
        
        # Low-rank coregionalization matrix
        self.W = nn.Parameter(torch.randn(num_outputs, rank) * 0.1)
        self.kappa = nn.Parameter(torch.ones(num_outputs) * 0.1)
    
    def forward(
        self,
        x: torch.Tensor,
        output_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_mean, base_var = self.base_gp(x)
        
        # Apply coregionalization
        output_scale = torch.sum(self.W[output_idx]**2) + self.kappa[output_idx]
        
        scaled_mean = base_mean * math.sqrt(output_scale)
        scaled_var = base_var * output_scale
        
        return scaled_mean, scaled_var
    
    def compute_cross_output_covariance(self, output_i: int, output_j: int) -> torch.Tensor:
        if output_i == output_j:
            return torch.sum(self.W[output_i]**2) + self.kappa[output_i]
        else:
            return torch.dot(self.W[output_i], self.W[output_j])

class DeepGPPreferences(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        num_inducing_per_layer: List[int] = [50, 50, 50]
    ):
        super().__init__()
        self.num_layers = len(hidden_dims)
        
        self.gp_layers = nn.ModuleList()
        
        current_dim = input_dim
        for i, (hidden_dim, num_inducing) in enumerate(zip(hidden_dims, num_inducing_per_layer)):
            gp_layer = GaussianProcessPreferences(
                current_dim,
                num_inducing=num_inducing
            )
            self.gp_layers.append(gp_layer)
            current_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        current_input = x
        
        for gp_layer in self.gp_layers:
            mean, var = gp_layer(current_input)
            
            # Sample from GP for next layer input
            if self.training:
                eps = torch.randn_like(mean)
                current_input = mean + eps * torch.sqrt(var + 1e-8)
            else:
                current_input = mean
        
        return current_input, var
    
    def compute_preference_probability(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1, var1 = self.forward(x1)
        f2, var2 = self.forward(x2)
        
        diff_mean = f1 - f2
        diff_var = var1 + var2
        
        z = diff_mean / torch.sqrt(diff_var + 1e-8)
        return torch.sigmoid(z)

class PreferenceGPTrainer:
    def __init__(
        self,
        model: Union[GaussianProcessPreferences, NeuralGaussianProcess],
        learning_rate: float = 1e-3,
        num_epochs: int = 1000
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
    
    def train(
        self,
        preference_data: List[Tuple[torch.Tensor, torch.Tensor, float]],
        validation_data: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]] = None
    ) -> Dict[str, List[float]]:
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            self.model.train()
            
            epoch_loss = 0.0
            for x1, x2, preference in preference_data:
                self.optimizer.zero_grad()
                
                if isinstance(self.model, GaussianProcessPreferences):
                    loss = self.model.elbo_loss([(x1, x2, preference)])
                else:
                    prob = self.model.compute_preference_probability(x1, x2)
                    loss = -preference * torch.log(prob + 1e-8) - (1 - preference) * torch.log(1 - prob + 1e-8)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(preference_data))
            
            if validation_data:
                val_loss = self._evaluate(validation_data)
                val_losses.append(val_loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}")
                if validation_data:
                    print(f"Val Loss: {val_losses[-1]:.4f}")
        
        return {"train_losses": train_losses, "val_losses": val_losses}
    
    def _evaluate(self, data: List[Tuple[torch.Tensor, torch.Tensor, float]]) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x1, x2, preference in data:
                if isinstance(self.model, GaussianProcessPreferences):
                    loss = self.model.elbo_loss([(x1, x2, preference)])
                else:
                    prob = self.model.compute_preference_probability(x1, x2)
                    loss = -preference * torch.log(prob + 1e-8) - (1 - preference) * torch.log(1 - prob + 1e-8)
                
                total_loss += loss.item()
        
        return total_loss / len(data)

class ActiveLearningGP:
    def __init__(
        self,
        model: GaussianProcessPreferences,
        acquisition_function: str = "variance"
    ):
        self.model = model
        self.acquisition_function = acquisition_function
    
    def select_query_pair(
        self,
        candidate_points: torch.Tensor,
        num_pairs: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.acquisition_function == "variance":
            return self._variance_based_selection(candidate_points, num_pairs)
        elif self.acquisition_function == "entropy":
            return self._entropy_based_selection(candidate_points, num_pairs)
        elif self.acquisition_function == "disagreement":
            return self._disagreement_based_selection(candidate_points, num_pairs)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
    
    def _variance_based_selection(
        self,
        candidates: torch.Tensor,
        num_pairs: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_candidates = candidates.size(0)
        max_uncertainty = 0
        best_pair = None
        
        for _ in range(num_pairs):
            i, j = torch.randint(0, n_candidates, (2,))
            if i != j:
                prob = self.model.compute_preference_probability(
                    candidates[i:i+1], candidates[j:j+1]
                )
                uncertainty = prob * (1 - prob)
                
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty
                    best_pair = (candidates[i:i+1], candidates[j:j+1])
        
        return best_pair
    
    def _entropy_based_selection(
        self,
        candidates: torch.Tensor,
        num_pairs: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_candidates = candidates.size(0)
        max_entropy = 0
        best_pair = None
        
        for _ in range(num_pairs):
            i, j = torch.randint(0, n_candidates, (2,))
            if i != j:
                prob = self.model.compute_preference_probability(
                    candidates[i:i+1], candidates[j:j+1]
                )
                
                entropy = -(prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8))
                
                if entropy > max_entropy:
                    max_entropy = entropy
                    best_pair = (candidates[i:i+1], candidates[j:j+1])
        
        return best_pair
    
    def _disagreement_based_selection(
        self,
        candidates: torch.Tensor,
        num_pairs: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_candidates = candidates.size(0)
        max_disagreement = 0
        best_pair = None
        
        for _ in range(num_pairs):
            i, j = torch.randint(0, n_candidates, (2,))
            if i != j:
                # Compute model uncertainty by sampling
                probs = []
                for _ in range(10):
                    prob = self.model.compute_preference_probability(
                        candidates[i:i+1], candidates[j:j+1]
                    )
                    probs.append(prob)
                
                probs = torch.cat(probs)
                disagreement = probs.std()
                
                if disagreement > max_disagreement:
                    max_disagreement = disagreement
                    best_pair = (candidates[i:i+1], candidates[j:j+1])
        
        return best_pair

class PreferenceOptimization:
    def __init__(self, gp_model: GaussianProcessPreferences):
        self.gp_model = gp_model
    
    def bayesian_optimization(
        self,
        bounds: torch.Tensor,
        num_iterations: int = 50,
        acquisition_function: str = "ei"
    ) -> torch.Tensor:
        best_point = None
        best_value = float('-inf')
        
        for iteration in range(num_iterations):
            if acquisition_function == "ei":
                candidate = self._expected_improvement_acquisition(bounds, best_value)
            elif acquisition_function == "ucb":
                candidate = self._upper_confidence_bound_acquisition(bounds, iteration)
            else:
                candidate = self._random_acquisition(bounds)
            
            mean, var = self.gp_model(candidate.unsqueeze(0))
            value = mean.item()
            
            if value > best_value:
                best_value = value
                best_point = candidate.clone()
            
            # In practice, you would update the GP with the new observation here
        
        return best_point
    
    def _expected_improvement_acquisition(
        self,
        bounds: torch.Tensor,
        best_value: float,
        num_candidates: int = 1000
    ) -> torch.Tensor:
        candidates = self._sample_candidates(bounds, num_candidates)
        
        means, vars = self.gp_model(candidates)
        stds = torch.sqrt(vars + 1e-8)
        
        improvement = means - best_value
        z = improvement / stds
        
        ei = improvement * torch.distributions.Normal(0, 1).cdf(z) + stds * torch.distributions.Normal(0, 1).log_prob(z).exp()
        
        best_idx = torch.argmax(ei)
        return candidates[best_idx]
    
    def _upper_confidence_bound_acquisition(
        self,
        bounds: torch.Tensor,
        iteration: int,
        beta: float = 2.0,
        num_candidates: int = 1000
    ) -> torch.Tensor:
        candidates = self._sample_candidates(bounds, num_candidates)
        
        means, vars = self.gp_model(candidates)
        stds = torch.sqrt(vars + 1e-8)
        
        # Adaptive beta
        adaptive_beta = beta * math.sqrt(math.log(2 * iteration + 1))
        
        ucb = means + adaptive_beta * stds
        
        best_idx = torch.argmax(ucb)
        return candidates[best_idx]
    
    def _random_acquisition(self, bounds: torch.Tensor) -> torch.Tensor:
        return self._sample_candidates(bounds, 1)[0]
    
    def _sample_candidates(self, bounds: torch.Tensor, num_candidates: int) -> torch.Tensor:
        dim = bounds.size(0)
        candidates = torch.rand(num_candidates, dim)
        
        for i in range(dim):
            candidates[:, i] = candidates[:, i] * (bounds[i, 1] - bounds[i, 0]) + bounds[i, 0]
        
        return candidates