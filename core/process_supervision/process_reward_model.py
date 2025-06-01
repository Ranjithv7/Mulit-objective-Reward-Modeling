import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import spacy
import re
from transformers import AutoTokenizer
import numpy as np

from reward_models.base_reward_model import BaseRewardModel, RewardOutput, RewardType
from reward_models.transformer_reward_model import TransformerRewardModel

class ProcessRewardModel(TransformerRewardModel):
    def __init__(
        self,
        *args,
        step_supervision_weight: float = 1.0,
        outcome_supervision_weight: float = 0.5,
        step_detection_method: str = "spacy",
        reasoning_format: str = "cot",  # chain-of-thought, step-by-step, etc.
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.step_supervision_weight = step_supervision_weight
        self.outcome_supervision_weight = outcome_supervision_weight
        self.step_detection_method = step_detection_method
        self.reasoning_format = reasoning_format
        
        # Initialize NLP tools for robust step detection
        if step_detection_method == "spacy":
            self.nlp = spacy.load("en_core_web_sm")
        
        # Process-level components
        self.step_reward_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        self.step_confidence_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.step_verifier = StepVerifier(self.hidden_size)
        self.reasoning_parser = ReasoningChainParser(self.nlp, reasoning_format)
        
        # Outcome reward head
        self.outcome_reward_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.num_objectives)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: Optional[List[List[Tuple[int, int]]]] = None,
        return_dict: bool = True,
        return_step_rewards: bool = True
    ) -> Union[torch.Tensor, RewardOutput]:
        # Get hidden states from backbone
        pooled_output, backbone_outputs = self._encode_sequence(input_ids, attention_mask)
        hidden_states = backbone_outputs.last_hidden_state
        
        # Detect step boundaries if not provided
        if step_boundaries is None:
            step_boundaries = self._detect_step_boundaries(input_ids, attention_mask)
        
        # Compute step-level rewards
        step_rewards = []
        step_confidences = []
        step_verifications = []
        
        batch_size = hidden_states.size(0)
        
        for batch_idx in range(batch_size):
            batch_steps = step_boundaries[batch_idx]
            batch_hidden = hidden_states[batch_idx]
            batch_step_rewards = []
            batch_step_confidences = []
            batch_step_verifications = []
            
            for start_pos, end_pos in batch_steps:
                # Extract step representation
                step_hidden = batch_hidden[start_pos:end_pos+1].mean(dim=0)
                
                # Compute step reward and confidence
                step_reward = self.step_reward_head(step_hidden)
                step_confidence = self.step_confidence_head(step_hidden)
                step_verification = self.step_verifier(step_hidden.unsqueeze(0))
                
                batch_step_rewards.append(step_reward)
                batch_step_confidences.append(step_confidence)
                batch_step_verifications.append(step_verification)
            
            if batch_step_rewards:
                step_rewards.append(torch.stack(batch_step_rewards))
                step_confidences.append(torch.stack(batch_step_confidences))
                step_verifications.append(torch.stack(batch_step_verifications))
            else:
                # Fallback for sequences without clear steps
                step_rewards.append(torch.zeros(1, 1, device=hidden_states.device))
                step_confidences.append(torch.zeros(1, 1, device=hidden_states.device))
                step_verifications.append(torch.zeros(1, 1, device=hidden_states.device))
        
        # Compute outcome reward
        outcome_reward = self.outcome_reward_head(pooled_output)
        
        # Combine step and outcome rewards
        step_reward_aggregated = torch.stack([sr.mean() for sr in step_rewards])
        combined_reward = (
            self.step_supervision_weight * step_reward_aggregated.unsqueeze(-1) +
            self.outcome_supervision_weight * outcome_reward
        )
        
        if self.normalize_rewards:
            combined_reward = self.normalize_reward(combined_reward)
        
        if not return_dict:
            return combined_reward
        
        process_rewards_dict = {
            "step_rewards": step_rewards,
            "step_confidences": step_confidences,
            "step_verifications": step_verifications,
            "step_boundaries": step_boundaries
        } if return_step_rewards else None
        
        return RewardOutput(
            rewards=combined_reward,
            process_rewards=step_rewards if return_step_rewards else None,
            hidden_states=pooled_output,
            objective_breakdown=process_rewards_dict
        )
    
    def _detect_step_boundaries(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[List[Tuple[int, int]]]:
        """Detect reasoning step boundaries using robust NLP methods"""
        batch_boundaries = []
        
        # Convert token IDs back to text for proper NLP processing
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        for batch_idx in range(input_ids.size(0)):
            seq_tokens = input_ids[batch_idx]
            seq_mask = attention_mask[batch_idx]
            
            # Get actual sequence length
            seq_len = seq_mask.sum().item()
            actual_tokens = seq_tokens[:seq_len]
            
            # Decode to text
            text = tokenizer.decode(actual_tokens, skip_special_tokens=True)
            
            # Parse reasoning steps
            step_spans = self.reasoning_parser.parse_steps(text)
            
            # Convert text spans back to token positions
            token_boundaries = self._text_spans_to_token_positions(
                text, step_spans, actual_tokens, tokenizer
            )
            
            batch_boundaries.append(token_boundaries)
        
        return batch_boundaries
    
    def _text_spans_to_token_positions(
        self,
        text: str,
        text_spans: List[Tuple[int, int]],
        tokens: torch.Tensor,
        tokenizer
    ) -> List[Tuple[int, int]]:
        """Convert character-level spans to token-level positions"""
        token_boundaries = []
        
        # Create character to token mapping
        char_to_token = {}
        current_char = 0
        
        for token_idx, token_id in enumerate(tokens):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            
            # Handle subword tokens
            if token_text.startswith("##") or token_text.startswith("▁"):
                token_text = token_text[2:] if token_text.startswith("##") else token_text[1:]
            
            for _ in range(len(token_text)):
                if current_char < len(text):
                    char_to_token[current_char] = token_idx
                    current_char += 1
        
        # Convert spans
        for char_start, char_end in text_spans:
            token_start = char_to_token.get(char_start, 0)
            token_end = char_to_token.get(char_end - 1, len(tokens) - 1)
            token_boundaries.append((token_start, token_end))
        
        return token_boundaries
    
    def compute_process_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_labels: List[List[float]],
        outcome_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute process-level supervised loss"""
        output = self.forward(input_ids, attention_mask, return_dict=True)
        
        # Step-level losses
        step_loss = 0.0
        confidence_loss = 0.0
        verification_loss = 0.0
        
        num_steps = 0
        
        for batch_idx, (step_rewards, step_confidences, step_verifications) in enumerate(
            zip(output.objective_breakdown["step_rewards"],
                output.objective_breakdown["step_confidences"],
                output.objective_breakdown["step_verifications"])
        ):
            batch_step_labels = step_labels[batch_idx]
            
            if len(batch_step_labels) == len(step_rewards):
                step_targets = torch.tensor(batch_step_labels, device=step_rewards.device)
                
                step_loss += F.mse_loss(step_rewards.squeeze(-1), step_targets)
                
                # Confidence should be high for correct steps
                confidence_targets = (step_targets > 0.5).float()
                confidence_loss += F.binary_cross_entropy(
                    step_confidences.squeeze(-1), 
                    confidence_targets
                )
                
                # Verification loss (steps should be verified as valid)
                verification_targets = torch.ones_like(step_verifications.squeeze(-1))
                verification_loss += F.binary_cross_entropy(
                    step_verifications.squeeze(-1),
                    verification_targets
                )
                
                num_steps += len(step_rewards)
        
        # Normalize step losses
        if num_steps > 0:
            step_loss /= num_steps
            confidence_loss /= num_steps
            verification_loss /= num_steps
        
        # Outcome loss
        outcome_loss = F.mse_loss(output.rewards, outcome_labels)
        
        # Combined loss
        total_loss = (
            self.step_supervision_weight * (step_loss + 0.1 * confidence_loss + 0.1 * verification_loss) +
            self.outcome_supervision_weight * outcome_loss
        )
        
        return {
            "total_loss": total_loss,
            "step_loss": step_loss,
            "confidence_loss": confidence_loss,
            "verification_loss": verification_loss,
            "outcome_loss": outcome_loss
        }

class StepVerifier(nn.Module):
    """Verifies if a reasoning step is valid and well-formed"""
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.verification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Multi-aspect verification
        self.coherence_head = nn.Linear(hidden_size, 1)
        self.relevance_head = nn.Linear(hidden_size, 1)
        self.completeness_head = nn.Linear(hidden_size, 1)
    
    def forward(self, step_hidden: torch.Tensor) -> torch.Tensor:
        # Overall verification score
        verification_score = self.verification_head(step_hidden)
        
        # Individual aspects
        coherence = torch.sigmoid(self.coherence_head(step_hidden))
        relevance = torch.sigmoid(self.relevance_head(step_hidden))
        completeness = torch.sigmoid(self.completeness_head(step_hidden))
        
        # Weighted combination
        combined_score = 0.4 * verification_score + 0.3 * coherence + 0.2 * relevance + 0.1 * completeness
        
        return combined_score

class ReasoningChainParser:
    """Robust reasoning chain parser using NLP tools"""
    def __init__(self, nlp_model, reasoning_format: str = "cot"):
        self.nlp = nlp_model
        self.reasoning_format = reasoning_format
        
        # Define step indicators for different reasoning formats
        self.step_patterns = {
            "cot": [
                r"Step \d+:",
                r"\d+\.",
                r"First,?",
                r"Second,?",
                r"Third,?",
                r"Next,?",
                r"Then,?",
                r"Finally,?",
                r"Therefore,?",
                r"So,?"
            ],
            "step_by_step": [
                r"Step \d+:",
                r"\d+\)",
                r"- ",
                r"• "
            ],
            "reasoning": [
                r"Because",
                r"Since",
                r"Given that",
                r"If",
                r"When",
                r"Therefore",
                r"Thus",
                r"Hence"
            ]
        }
    
    def parse_steps(self, text: str) -> List[Tuple[int, int]]:
        """Parse reasoning steps from text using NLP"""
        doc = self.nlp(text)
        
        # Method 1: Pattern-based detection
        pattern_spans = self._pattern_based_detection(text)
        
        # Method 2: Sentence-based segmentation
        sentence_spans = self._sentence_based_detection(doc)
        
        # Method 3: Discourse marker detection
        discourse_spans = self._discourse_marker_detection(doc)
        
        # Combine and refine spans
        all_spans = pattern_spans + sentence_spans + discourse_spans
        refined_spans = self._refine_spans(all_spans, len(text))
        
        return refined_spans
    
    def _pattern_based_detection(self, text: str) -> List[Tuple[int, int]]:
        """Detect steps using regex patterns"""
        spans = []
        patterns = self.step_patterns.get(self.reasoning_format, self.step_patterns["cot"])
        
        # Find pattern matches
        pattern_positions = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pattern_positions.append(match.start())
        
        pattern_positions = sorted(set(pattern_positions))
        
        # Create spans between patterns
        for i in range(len(pattern_positions)):
            start = pattern_positions[i]
            end = pattern_positions[i + 1] if i + 1 < len(pattern_positions) else len(text)
            
            # Find sentence end within this span
            span_text = text[start:end]
            sentences = span_text.split('. ')
            if len(sentences) > 1:
                end = start + len(sentences[0]) + 1
            
            spans.append((start, min(end, len(text))))
        
        return spans
    
    def _sentence_based_detection(self, doc) -> List[Tuple[int, int]]:
        """Detect steps based on sentence boundaries"""
        spans = []
        
        for sent in doc.sents:
            # Check if sentence contains reasoning indicators
            sent_text = sent.text.lower()
            
            reasoning_indicators = [
                "because", "since", "therefore", "thus", "hence", "so",
                "first", "second", "third", "next", "then", "finally",
                "step", "let's", "we need", "we can", "this means"
            ]
            
            if any(indicator in sent_text for indicator in reasoning_indicators):
                spans.append((sent.start_char, sent.end_char))
        
        return spans
    
    def _discourse_marker_detection(self, doc) -> List[Tuple[int, int]]:
        """Detect steps using discourse markers and dependency parsing"""
        spans = []
        
        # Look for discourse connectives and logical operators
        discourse_markers = {"because", "since", "therefore", "thus", "hence", "so", "then", "next"}
        
        current_span_start = 0
        
        for token in doc:
            if token.text.lower() in discourse_markers and token.dep_ in ["mark", "advmod", "cc"]:
                # End current span and start new one
                if current_span_start < token.idx:
                    spans.append((current_span_start, token.idx))
                current_span_start = token.idx
        
        # Add final span
        if current_span_start < len(doc.text):
            spans.append((current_span_start, len(doc.text)))
        
        return spans
    
    def _refine_spans(self, spans: List[Tuple[int, int]], text_length: int) -> List[Tuple[int, int]]:
        """Refine and merge overlapping spans"""
        if not spans:
            return [(0, text_length)]
        
        # Sort spans by start position
        spans = sorted(set(spans))
        
        # Merge overlapping spans
        merged = []
        current_start, current_end = spans[0]
        
        for start, end in spans[1:]:
            if start <= current_end + 10:  # Allow small gaps
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        
        # Filter out very short spans (< 10 characters)
        merged = [(s, e) for s, e in merged if e - s >= 10]
        
        # Ensure coverage of full text
        if not merged or merged[0][0] > 10:
            merged.insert(0, (0, merged[0][0] if merged else text_length))
        
        if merged[-1][1] < text_length - 10:
            merged.append((merged[-1][1], text_length))
        
        return merged

class ProcessRewardTrainer:
    """Trainer for process reward models with step-level supervision"""
    def __init__(
        self,
        model: ProcessRewardModel,
        learning_rate: float = 1e-4,
        step_loss_weight: float = 1.0,
        outcome_loss_weight: float = 0.5
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.step_loss_weight = step_loss_weight
        self.outcome_loss_weight = outcome_loss_weight
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step_labels: List[List[float]],
        outcome_labels: torch.Tensor
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        loss_dict = self.model.compute_process_loss(
            batch["input_ids"],
            batch["attention_mask"],
            step_labels,
            outcome_labels
        )
        
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Convert to metrics
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        
        return metrics
    
    def evaluate(
        self,
        eval_dataloader,
        step_labels_list: List[List[List[float]]],
        outcome_labels_list: List[torch.Tensor]
    ) -> Dict[str, float]:
        self.model.eval()
        
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader):
                step_labels = step_labels_list[batch_idx]
                outcome_labels = outcome_labels_list[batch_idx]
                
                loss_dict = self.model.compute_process_loss(
                    batch["input_ids"],
                    batch["attention_mask"],
                    step_labels,
                    outcome_labels
                )
                
                for key, value in loss_dict.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value.item() if torch.is_tensor(value) else value
                
                num_batches += 1
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return avg_metrics