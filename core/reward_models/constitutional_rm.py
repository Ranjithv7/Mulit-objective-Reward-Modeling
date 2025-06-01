import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .base_reward_model import BaseRewardModel, RewardOutput, RewardType
from .transformer_reward_model import TransformerRewardModel

class ConstitutionPrinciple(Enum):
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    HONESTY = "honesty"
    TRANSPARENCY = "transparency"
    FAIRNESS = "fairness"
    RESPECT = "respect"
    PRIVACY = "privacy"

@dataclass
class ConstitutionalRule:
    principle: ConstitutionPrinciple
    description: str
    weight: float = 1.0
    violation_threshold: float = 0.5

class ConstitutionalRewardModel(TransformerRewardModel):
    def __init__(
        self,
        *args,
        constitution_rules: List[ConstitutionalRule] = None,
        self_critique_enabled: bool = True,
        chain_of_thought: bool = True,
        revision_steps: int = 3,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.constitution_rules = constitution_rules or self._default_constitution()
        self.self_critique_enabled = self_critique_enabled
        self.chain_of_thought = chain_of_thought
        self.revision_steps = revision_steps
        
        self.principle_heads = nn.ModuleDict({
            principle.value: PrincipleHead(self.hidden_size, self.dropout)
            for principle in ConstitutionPrinciple
        })
        
        if self_critique_enabled:
            self.critique_head = CritiqueHead(self.hidden_size, len(self.constitution_rules))
            self.revision_head = RevisionHead(self.hidden_size, self.dropout)
        
        self.constitution_aggregator = ConstitutionAggregator(
            len(self.constitution_rules),
            self.hidden_size
        )
    
    def _default_constitution(self) -> List[ConstitutionalRule]:
        return [
            ConstitutionalRule(
                ConstitutionPrinciple.HELPFULNESS,
                "Response should be helpful and address the user's needs",
                weight=1.0
            ),
            ConstitutionalRule(
                ConstitutionPrinciple.HARMLESSNESS,
                "Response should not cause harm or promote dangerous activities",
                weight=1.5
            ),
            ConstitutionalRule(
                ConstitutionPrinciple.HONESTY,
                "Response should be truthful and not contain misinformation",
                weight=1.2
            ),
            ConstitutionalRule(
                ConstitutionPrinciple.TRANSPARENCY,
                "Response should be clear about limitations and uncertainties",
                weight=0.8
            ),
            ConstitutionalRule(
                ConstitutionPrinciple.FAIRNESS,
                "Response should be fair and avoid bias",
                weight=1.0
            ),
            ConstitutionalRule(
                ConstitutionPrinciple.RESPECT,
                "Response should be respectful and considerate",
                weight=0.9
            ),
            ConstitutionalRule(
                ConstitutionPrinciple.PRIVACY,
                "Response should respect privacy and confidentiality",
                weight=1.1
            )
        ]
    
    def _create_reward_head(self) -> nn.Module:
        return ConstitutionalRewardHead(
            self.hidden_size,
            len(self.constitution_rules),
            self.dropout
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
        return_principle_scores: bool = True
    ) -> Union[torch.Tensor, RewardOutput]:
        pooled_output, backbone_outputs = self._encode_sequence(input_ids, attention_mask)
        pooled_output = self.dropout_layer(pooled_output)
        
        principle_scores = {}
        for principle, head in self.principle_heads.items():
            principle_scores[principle] = head(pooled_output)
        
        constitutional_reward = self.constitution_aggregator(
            principle_scores,
            [rule.weight for rule in self.constitution_rules]
        )
        
        if self.self_critique_enabled:
            critique_output = self.critique_head(pooled_output, principle_scores)
            constitutional_reward = self._apply_self_critique(
                constitutional_reward,
                critique_output,
                pooled_output
            )
        
        if self.normalize_rewards:
            constitutional_reward = self.normalize_reward(constitutional_reward)
        
        if not return_dict:
            return constitutional_reward
        
        return RewardOutput(
            rewards=constitutional_reward,
            hidden_states=pooled_output,
            objective_breakdown=principle_scores if return_principle_scores else None
        )
    
    def _apply_self_critique(
        self,
        initial_reward: torch.Tensor,
        critique_output: Dict[str, torch.Tensor],
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        violations = critique_output["violations"]
        confidence = critique_output["confidence"]
        
        penalty = torch.zeros_like(initial_reward)
        for i, rule in enumerate(self.constitution_rules):
            violation_score = violations[:, i]
            rule_penalty = violation_score * rule.weight * (violation_score > rule.violation_threshold).float()
            penalty += rule_penalty.unsqueeze(-1)
        
        uncertainty_penalty = (1 - confidence) * 0.1
        final_reward = initial_reward - penalty - uncertainty_penalty.unsqueeze(-1)
        
        return final_reward
    
    def constitutional_training_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_rewards: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        output = self.forward(input_ids, attention_mask, return_dict=True)
        
        reward_loss = F.mse_loss(output.rewards, target_rewards)
        
        principle_losses = {}
        if output.objective_breakdown:
            for principle, scores in output.objective_breakdown.items():
                if principle in [rule.principle.value for rule in self.constitution_rules]:
                    principle_losses[f"{principle}_loss"] = F.mse_loss(
                        scores,
                        target_rewards
                    )
        
        total_loss = reward_loss + sum(principle_losses.values()) * 0.1
        
        return {
            "total_loss": total_loss,
            "reward_loss": reward_loss,
            **principle_losses
        }
    
    def generate_constitutional_critique(
        self,
        response_text: str,
        tokenizer
    ) -> Dict[str, float]:
        inputs = tokenizer(
            response_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            output = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
                return_dict=True,
                return_principle_scores=True
            )
        
        critique = {}
        if output.objective_breakdown:
            for principle, scores in output.objective_breakdown.items():
                critique[principle] = scores.item()
        
        return critique

class PrincipleHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))

class CritiqueHead(nn.Module):
    def __init__(self, hidden_size: int, num_principles: int):
        super().__init__()
        self.num_principles = num_principles
        
        self.violation_detector = nn.Sequential(
            nn.Linear(hidden_size + num_principles, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_principles),
            nn.Sigmoid()
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size + num_principles, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        principle_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        principle_tensor = torch.cat(list(principle_scores.values()), dim=-1)
        combined_input = torch.cat([hidden_states, principle_tensor], dim=-1)
        
        violations = self.violation_detector(combined_input)
        confidence = self.confidence_estimator(combined_input)
        
        return {
            "violations": violations,
            "confidence": confidence
        }

class RevisionHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.revision_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
    
    def forward(
        self,
        original_hidden: torch.Tensor,
        critique_hidden: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([original_hidden, critique_hidden], dim=-1)
        revision = self.revision_network(combined)
        return original_hidden + revision

class ConstitutionAggregator(nn.Module):
    def __init__(self, num_principles: int, hidden_size: int):
        super().__init__()
        self.num_principles = num_principles
        
        self.attention_weights = nn.Sequential(
            nn.Linear(num_principles, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_principles),
            nn.Softmax(dim=-1)
        )
        
        self.final_projection = nn.Linear(num_principles, 1)
    
    def forward(
        self,
        principle_scores: Dict[str, torch.Tensor],
        rule_weights: List[float]
    ) -> torch.Tensor:
        scores_tensor = torch.cat(list(principle_scores.values()), dim=-1)
        
        dynamic_weights = self.attention_weights(scores_tensor)
        
        static_weights = torch.tensor(rule_weights, device=scores_tensor.device)
        static_weights = static_weights / static_weights.sum()
        
        combined_weights = 0.7 * dynamic_weights + 0.3 * static_weights.unsqueeze(0)
        
        weighted_scores = scores_tensor * combined_weights
        constitutional_reward = self.final_projection(weighted_scores)
        
        return constitutional_reward

class ConstitutionalRewardHead(nn.Module):
    def __init__(self, hidden_size: int, num_principles: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_principles),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class SelfSupervisedConstitutionalTrainer:
    def __init__(self, model: ConstitutionalRewardModel):
        self.model = model
    
    def generate_constitutional_pairs(
        self,
        responses: List[str],
        tokenizer,
        num_revisions: int = 3
    ) -> List[Tuple[str, str, float]]:
        pairs = []
        
        for response in responses:
            critique = self.model.generate_constitutional_critique(response, tokenizer)
            
            violation_score = sum(
                score for principle, score in critique.items()
                if score < 0.5
            ) / len(critique)
            
            revised_response = self._apply_constitutional_revision(
                response,
                critique,
                num_revisions
            )
            
            if revised_response != response:
                pairs.append((revised_response, response, violation_score))
        
        return pairs
    
    def _apply_constitutional_revision(
        self,
        original_response: str,
        critique: Dict[str, float],
        num_revisions: int
    ) -> str:
        violated_principles = [
            principle for principle, score in critique.items()
            if score < 0.5
        ]
        
        if not violated_principles:
            return original_response
        
        revision_prompt = self._generate_revision_prompt(
            original_response,
            violated_principles
        )
        
        return revision_prompt
    
    def _generate_revision_prompt(
        self,
        response: str,
        violated_principles: List[str]
    ) -> str:
        principle_descriptions = {
            "helpfulness": "more helpful and informative",
            "harmlessness": "safer and less harmful",
            "honesty": "more truthful and accurate",
            "transparency": "more transparent about limitations",
            "fairness": "more fair and unbiased",
            "respect": "more respectful and considerate",
            "privacy": "more respectful of privacy"
        }
        
        improvements = [
            principle_descriptions.get(principle, principle)
            for principle in violated_principles
        ]
        
        revision_instruction = f"Revise to be {', '.join(improvements)}: {response}"
        return revision_instruction

class ConstitutionalAlignment:
    @staticmethod
    def compute_principle_alignment(
        model_scores: Dict[str, torch.Tensor],
        human_preferences: Dict[str, torch.Tensor]
    ) -> float:
        alignments = []
        
        for principle in model_scores.keys():
            if principle in human_preferences:
                model_score = model_scores[principle]
                human_score = human_preferences[principle]
                
                correlation = torch.corrcoef(
                    torch.stack([model_score.flatten(), human_score.flatten()])
                )[0, 1]
                
                if not torch.isnan(correlation):
                    alignments.append(correlation.item())
        
        return sum(alignments) / len(alignments) if alignments else 0.0
    
    @staticmethod
    def detect_principle_conflicts(
        principle_scores: Dict[str, torch.Tensor],
        threshold: float = 0.3
    ) -> List[Tuple[str, str]]:
        conflicts = []
        principles = list(principle_scores.keys())
        
        for i, principle1 in enumerate(principles):
            for principle2 in principles[i+1:]:
                score1 = principle_scores[principle1]
                score2 = principle_scores[principle2]
                
                correlation = torch.corrcoef(
                    torch.stack([score1.flatten(), score2.flatten()])
                )[0, 1]
                
                if correlation < -threshold:
                    conflicts.append((principle1, principle2))
        
        return conflicts