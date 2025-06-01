import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import math
import numpy as np
import random

from reward_models.base_reward_model import BaseRewardModel, RewardOutput, PreferenceData

class SelfSupervisedPreferenceGenerator:
    def __init__(
        self,
        base_model: BaseRewardModel,
        generation_strategy: str = "contrastive",
        augmentation_strength: float = 0.1,
        consistency_threshold: float = 0.8
    ):
        self.base_model = base_model
        self.generation_strategy = generation_strategy
        self.augmentation_strength = augmentation_strength
        self.consistency_threshold = consistency_threshold
        
        if generation_strategy == "contrastive":
            self.generator = ContrastivePreferenceGenerator(base_model)
        elif generation_strategy == "ranking":
            self.generator = RankingBasedGenerator(base_model)
        elif generation_strategy == "consistency":
            self.generator = ConsistencyBasedGenerator(base_model)
        elif generation_strategy == "adversarial":
            self.generator = AdversarialPreferenceGenerator(base_model)
        elif generation_strategy == "curriculum":
            self.generator = CurriculumPreferenceGenerator(base_model)
        else:
            raise ValueError(f"Unknown generation strategy: {generation_strategy}")
    
    def generate_preferences(
        self,
        input_data: List[Dict[str, torch.Tensor]],
        num_preferences: int = 1000
    ) -> List[PreferenceData]:
        return self.generator.generate(input_data, num_preferences)

class ContrastivePreferenceGenerator:
    def __init__(
        self,
        base_model: BaseRewardModel,
        temperature: float = 0.1,
        augmentation_types: List[str] = ["dropout", "noise", "permutation"]
    ):
        self.base_model = base_model
        self.temperature = temperature
        self.augmentation_types = augmentation_types
        
        self.augmentors = {
            "dropout": DropoutAugmentation(0.1),
            "noise": NoiseAugmentation(0.05),
            "permutation": PermutationAugmentation(0.1),
            "masking": MaskingAugmentation(0.15),
            "paraphrasing": ParaphrasingAugmentation()
        }
    
    def generate(
        self,
        input_data: List[Dict[str, torch.Tensor]],
        num_preferences: int
    ) -> List[PreferenceData]:
        preferences = []
        
        for _ in range(num_preferences):
            # Sample two different examples
            idx1, idx2 = random.sample(range(len(input_data)), 2)
            
            # Create augmented versions
            aug_type1 = random.choice(self.augmentation_types)
            aug_type2 = random.choice(self.augmentation_types)
            
            original1 = input_data[idx1]
            original2 = input_data[idx2]
            
            augmented1 = self.augmentors[aug_type1].augment(original1)
            augmented2 = self.augmentors[aug_type2].augment(original2)
            
            # Compute rewards for all versions
            with torch.no_grad():
                reward_orig1 = self.base_model(original1["input_ids"], original1["attention_mask"], return_dict=False)
                reward_orig2 = self.base_model(original2["input_ids"], original2["attention_mask"], return_dict=False)
                reward_aug1 = self.base_model(augmented1["input_ids"], augmented1["attention_mask"], return_dict=False)
                reward_aug2 = self.base_model(augmented2["input_ids"], augmented2["attention_mask"], return_dict=False)
            
            # Generate contrastive pairs
            pairs = [
                (original1, augmented1, 1.0),  # Original should be preferred over augmented
                (original2, augmented2, 1.0),
                (original1, augmented2, float(reward_orig1 > reward_aug2)),
                (original2, augmented1, float(reward_orig2 > reward_aug1))
            ]
            
            for chosen_data, rejected_data, preference in pairs:
                preferences.append(PreferenceData(
                    chosen=chosen_data["input_ids"],
                    rejected=rejected_data["input_ids"],
                    chosen_attention_mask=chosen_data["attention_mask"],
                    rejected_attention_mask=rejected_data["attention_mask"]
                ))
        
        return preferences

class RankingBasedGenerator:
    def __init__(
        self,
        base_model: BaseRewardModel,
        ranking_size: int = 5,
        margin_threshold: float = 0.1
    ):
        self.base_model = base_model
        self.ranking_size = ranking_size
        self.margin_threshold = margin_threshold
    
    def generate(
        self,
        input_data: List[Dict[str, torch.Tensor]],
        num_preferences: int
    ) -> List[PreferenceData]:
        preferences = []
        
        # Compute rewards for all examples
        all_rewards = []
        with torch.no_grad():
            for data in input_data:
                reward = self.base_model(data["input_ids"], data["attention_mask"], return_dict=False)
                all_rewards.append(reward.item())
        
        # Sort by rewards
        sorted_indices = sorted(range(len(all_rewards)), key=lambda i: all_rewards[i], reverse=True)
        
        # Generate preferences from rankings
        for _ in range(num_preferences):
            # Sample a ranking of examples
            ranking_indices = random.sample(sorted_indices, min(self.ranking_size, len(sorted_indices)))
            ranking_indices.sort(key=lambda i: all_rewards[i], reverse=True)
            
            # Create pairwise preferences from ranking
            for i in range(len(ranking_indices) - 1):
                for j in range(i + 1, len(ranking_indices)):
                    idx_better = ranking_indices[i]
                    idx_worse = ranking_indices[j]
                    
                    # Only include if margin is significant
                    if all_rewards[idx_better] - all_rewards[idx_worse] > self.margin_threshold:
                        preferences.append(PreferenceData(
                            chosen=input_data[idx_better]["input_ids"],
                            rejected=input_data[idx_worse]["input_ids"],
                            chosen_attention_mask=input_data[idx_better]["attention_mask"],
                            rejected_attention_mask=input_data[idx_worse]["attention_mask"]
                        ))
        
        return preferences

class ConsistencyBasedGenerator:
    def __init__(
        self,
        base_model: BaseRewardModel,
        num_ensemble: int = 5,
        consistency_threshold: float = 0.8
    ):
        self.base_model = base_model
        self.num_ensemble = num_ensemble
        self.consistency_threshold = consistency_threshold
    
    def generate(
        self,
        input_data: List[Dict[str, torch.Tensor]],
        num_preferences: int
    ) -> List[PreferenceData]:
        preferences = []
        
        for _ in range(num_preferences):
            # Sample two examples
            idx1, idx2 = random.sample(range(len(input_data)), 2)
            data1, data2 = input_data[idx1], input_data[idx2]
            
            # Get ensemble predictions
            predictions1 = []
            predictions2 = []
            
            for _ in range(self.num_ensemble):
                self.base_model.train()  # Enable dropout for diversity
                with torch.no_grad():
                    reward1 = self.base_model(data1["input_ids"], data1["attention_mask"], return_dict=False)
                    reward2 = self.base_model(data2["input_ids"], data2["attention_mask"], return_dict=False)
                    predictions1.append(reward1.item())
                    predictions2.append(reward2.item())
            
            self.base_model.eval()
            
            # Check consistency
            preferences1 = [r1 > r2 for r1, r2 in zip(predictions1, predictions2)]
            consistency = sum(preferences1) / len(preferences1)
            
            # Only include if consistent across ensemble
            if consistency > self.consistency_threshold or consistency < (1 - self.consistency_threshold):
                chosen_idx = idx1 if consistency > 0.5 else idx2
                rejected_idx = idx2 if consistency > 0.5 else idx1
                
                preferences.append(PreferenceData(
                    chosen=input_data[chosen_idx]["input_ids"],
                    rejected=input_data[rejected_idx]["input_ids"],
                    chosen_attention_mask=input_data[chosen_idx]["attention_mask"],
                    rejected_attention_mask=input_data[rejected_idx]["attention_mask"]
                ))
        
        return preferences

class AdversarialPreferenceGenerator:
    def __init__(
        self,
        base_model: BaseRewardModel,
        perturbation_magnitude: float = 0.01,
        num_perturbations: int = 10
    ):
        self.base_model = base_model
        self.perturbation_magnitude = perturbation_magnitude
        self.num_perturbations = num_perturbations
    
    def generate(
        self,
        input_data: List[Dict[str, torch.Tensor]],
        num_preferences: int
    ) -> List[PreferenceData]:
        preferences = []
        
        for _ in range(num_preferences):
            # Sample an example
            idx = random.randint(0, len(input_data) - 1)
            original_data = input_data[idx]
            
            # Generate adversarial perturbations
            best_perturbation = None
            best_reward_diff = 0
            
            for _ in range(self.num_perturbations):
                perturbed_data = self._create_adversarial_perturbation(original_data)
                
                with torch.no_grad():
                    original_reward = self.base_model(
                        original_data["input_ids"], 
                        original_data["attention_mask"], 
                        return_dict=False
                    )
                    perturbed_reward = self.base_model(
                        perturbed_data["input_ids"], 
                        perturbed_data["attention_mask"], 
                        return_dict=False
                    )
                
                reward_diff = abs(original_reward - perturbed_reward).item()
                
                if reward_diff > best_reward_diff:
                    best_reward_diff = reward_diff
                    best_perturbation = perturbed_data
            
            if best_perturbation is not None and best_reward_diff > 0.1:
                # Original should be preferred over adversarial perturbation
                preferences.append(PreferenceData(
                    chosen=original_data["input_ids"],
                    rejected=best_perturbation["input_ids"],
                    chosen_attention_mask=original_data["attention_mask"],
                    rejected_attention_mask=best_perturbation["attention_mask"]
                ))
        
        return preferences
    
    def _create_adversarial_perturbation(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Simple token substitution perturbation
        input_ids = data["input_ids"].clone()
        attention_mask = data["attention_mask"].clone()
        
        # Randomly substitute a few tokens
        seq_len = attention_mask.sum().item()
        num_substitutions = max(1, int(seq_len * 0.1))
        
        for _ in range(num_substitutions):
            pos = random.randint(1, seq_len - 2)  # Avoid special tokens
            # Replace with random token (simplified)
            input_ids[0, pos] = random.randint(1000, 5000)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

class CurriculumPreferenceGenerator:
    def __init__(
        self,
        base_model: BaseRewardModel,
        difficulty_levels: int = 5,
        examples_per_level: int = 100
    ):
        self.base_model = base_model
        self.difficulty_levels = difficulty_levels
        self.examples_per_level = examples_per_level
        self.current_level = 0
    
    def generate(
        self,
        input_data: List[Dict[str, torch.Tensor]],
        num_preferences: int
    ) -> List[PreferenceData]:
        # Organize data by difficulty
        difficulty_buckets = self._organize_by_difficulty(input_data)
        
        preferences = []
        examples_per_difficulty = num_preferences // self.difficulty_levels
        
        for level in range(min(self.current_level + 1, self.difficulty_levels)):
            level_data = difficulty_buckets[level]
            level_preferences = self._generate_level_preferences(level_data, examples_per_difficulty)
            preferences.extend(level_preferences)
        
        # Advance curriculum
        if len(preferences) >= num_preferences * 0.8:  # Success threshold
            self.current_level = min(self.current_level + 1, self.difficulty_levels - 1)
        
        return preferences[:num_preferences]
    
    def _organize_by_difficulty(self, input_data: List[Dict[str, torch.Tensor]]) -> Dict[int, List[Dict]]:
        # Compute difficulty scores (reward variance as proxy)
        difficulties = []
        
        for data in input_data:
            with torch.no_grad():
                # Use multiple forward passes to estimate variance
                rewards = []
                for _ in range(5):
                    self.base_model.train()
                    reward = self.base_model(data["input_ids"], data["attention_mask"], return_dict=False)
                    rewards.append(reward.item())
                
                difficulty = np.var(rewards)  # Higher variance = more difficult
                difficulties.append(difficulty)
        
        # Sort and bucket by difficulty
        sorted_indices = sorted(range(len(difficulties)), key=lambda i: difficulties[i])
        
        buckets = {}
        bucket_size = len(input_data) // self.difficulty_levels
        
        for level in range(self.difficulty_levels):
            start_idx = level * bucket_size
            end_idx = start_idx + bucket_size if level < self.difficulty_levels - 1 else len(input_data)
            bucket_indices = sorted_indices[start_idx:end_idx]
            buckets[level] = [input_data[i] for i in bucket_indices]
        
        return buckets
    
    def _generate_level_preferences(
        self,
        level_data: List[Dict[str, torch.Tensor]],
        num_preferences: int
    ) -> List[PreferenceData]:
        preferences = []
        
        for _ in range(num_preferences):
            if len(level_data) >= 2:
                # Sample two examples from this difficulty level
                idx1, idx2 = random.sample(range(len(level_data)), 2)
                data1, data2 = level_data[idx1], level_data[idx2]
                
                with torch.no_grad():
                    reward1 = self.base_model(data1["input_ids"], data1["attention_mask"], return_dict=False)
                    reward2 = self.base_model(data2["input_ids"], data2["attention_mask"], return_dict=False)
                
                if reward1 > reward2:
                    chosen_data, rejected_data = data1, data2
                else:
                    chosen_data, rejected_data = data2, data1
                
                preferences.append(PreferenceData(
                    chosen=chosen_data["input_ids"],
                    rejected=rejected_data["input_ids"],
                    chosen_attention_mask=chosen_data["attention_mask"],
                    rejected_attention_mask=rejected_data["attention_mask"]
                ))
        
        return preferences

class DropoutAugmentation:
    def __init__(self, dropout_rate: float = 0.1):
        self.dropout_rate = dropout_rate
    
    def augment(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = data["input_ids"].clone()
        attention_mask = data["attention_mask"].clone()
        
        # Apply dropout to embeddings (simulated by token masking)
        mask = torch.rand_like(input_ids.float()) > self.dropout_rate
        input_ids = input_ids * mask.long()
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}

class NoiseAugmentation:
    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std
    
    def augment(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # For token-level data, we simulate noise by slight token perturbations
        input_ids = data["input_ids"].clone()
        attention_mask = data["attention_mask"].clone()
        
        # Random token substitutions
        noise_mask = torch.rand_like(input_ids.float()) < self.noise_std
        random_tokens = torch.randint_like(input_ids, low=1000, high=5000)
        input_ids = torch.where(noise_mask, random_tokens, input_ids)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}

class PermutationAugmentation:
    def __init__(self, permutation_rate: float = 0.1):
        self.permutation_rate = permutation_rate
    
    def augment(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = data["input_ids"].clone()
        attention_mask = data["attention_mask"].clone()
        
        seq_len = attention_mask.sum().item()
        num_swaps = max(1, int(seq_len * self.permutation_rate))
        
        for _ in range(num_swaps):
            # Swap two random positions
            pos1 = random.randint(1, seq_len - 2)
            pos2 = random.randint(1, seq_len - 2)
            
            input_ids[0, pos1], input_ids[0, pos2] = input_ids[0, pos2], input_ids[0, pos1]
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}

class MaskingAugmentation:
    def __init__(self, mask_rate: float = 0.15, mask_token_id: int = 103):
        self.mask_rate = mask_rate
        self.mask_token_id = mask_token_id
    
    def augment(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = data["input_ids"].clone()
        attention_mask = data["attention_mask"].clone()
        
        seq_len = attention_mask.sum().item()
        num_masks = max(1, int(seq_len * self.mask_rate))
        
        for _ in range(num_masks):
            pos = random.randint(1, seq_len - 2)
            input_ids[0, pos] = self.mask_token_id
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}

class ParaphrasingAugmentation:
    def __init__(self):
        # This would typically use a paraphrasing model
        # For simplicity, we'll do basic synonym substitution
        self.synonym_map = {
            "good": ["great", "excellent", "fine"],
            "bad": ["poor", "terrible", "awful"],
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "miniature"]
        }
    
    def augment(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Simplified paraphrasing - in practice, use a proper paraphrasing model
        return data  # Placeholder implementation

class PreferenceQualityAssessment:
    def __init__(self, reward_model: BaseRewardModel):
        self.reward_model = reward_model
    
    def assess_preference_quality(
        self,
        preferences: List[PreferenceData],
        quality_threshold: float = 0.7
    ) -> List[PreferenceData]:
        high_quality_preferences = []
        
        for pref in preferences:
            quality_score = self._compute_quality_score(pref)
            
            if quality_score > quality_threshold:
                high_quality_preferences.append(pref)
        
        return high_quality_preferences
    
    def _compute_quality_score(self, preference: PreferenceData) -> float:
        with torch.no_grad():
            chosen_reward = self.reward_model(preference.chosen, preference.chosen_attention_mask, return_dict=False)
            rejected_reward = self.reward_model(preference.rejected, preference.rejected_attention_mask, return_dict=False)
        
        # Quality based on reward difference and confidence
        reward_diff = chosen_reward - rejected_reward
        confidence = torch.sigmoid(reward_diff).item()
        
        # Penalize preferences that are too easy or too hard
        difficulty_penalty = 1 - abs(confidence - 0.7) / 0.3
        
        return min(confidence, difficulty_penalty)

class PreferenceDataAugmenter:
    def __init__(self, augmentation_strategies: List[str] = ["synonym", "paraphrase", "reorder"]):
        self.strategies = augmentation_strategies
    
    def augment_preferences(
        self,
        preferences: List[PreferenceData],
        augmentation_factor: int = 2
    ) -> List[PreferenceData]:
        augmented_preferences = preferences.copy()
        
        for pref in preferences:
            for _ in range(augmentation_factor - 1):
                strategy = random.choice(self.strategies)
                
                if strategy == "synonym":
                    aug_pref = self._synonym_augmentation(pref)
                elif strategy == "paraphrase":
                    aug_pref = self._paraphrase_augmentation(pref)
                elif strategy == "reorder":
                    aug_pref = self._reorder_augmentation(pref)
                else:
                    aug_pref = pref
                
                augmented_preferences.append(aug_pref)
        
        return augmented_preferences
    
    def _synonym_augmentation(self, preference: PreferenceData) -> PreferenceData:
        # Placeholder for synonym replacement
        return preference
    
    def _paraphrase_augmentation(self, preference: PreferenceData) -> PreferenceData:
        # Placeholder for paraphrasing
        return preference
    
    def _reorder_augmentation(self, preference: PreferenceData) -> PreferenceData:
        # Placeholder for sentence reordering
        return preference