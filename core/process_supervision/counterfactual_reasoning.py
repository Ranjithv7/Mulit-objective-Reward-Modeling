import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from scipy.special import expit
import itertools
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer
import spacy

class CounterfactualAnalyzer:
    """Analyzes counterfactual scenarios for reasoning step evaluation"""
    
    def __init__(
        self,
        counterfactual_method: str = "intervention",
        perturbation_strength: float = 0.1,
        num_counterfactuals: int = 10,
        similarity_threshold: float = 0.8,
        causal_strength_threshold: float = 0.3
    ):
        self.counterfactual_method = counterfactual_method
        self.perturbation_strength = perturbation_strength
        self.num_counterfactuals = num_counterfactuals
        self.similarity_threshold = similarity_threshold
        self.causal_strength_threshold = causal_strength_threshold
        
        # Initialize NLP tools for text manipulation
        self.nlp = spacy.load("en_core_web_sm")
        
        # Counterfactual generation methods
        self.counterfactual_methods = {
            "intervention": self._intervention_counterfactuals,
            "substitution": self._substitution_counterfactuals,
            "removal": self._removal_counterfactuals,
            "reordering": self._reordering_counterfactuals,
            "semantic_perturbation": self._semantic_perturbation_counterfactuals,
            "logical_negation": self._logical_negation_counterfactuals,
            "causal_intervention": self._causal_intervention_counterfactuals
        }
    
    def analyze_counterfactuals(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        original_rewards: torch.Tensor,
        tokenizer
    ) -> Dict[str, List[Dict]]:
        """Analyze counterfactual scenarios for reasoning evaluation"""
        
        counterfactual_fn = self.counterfactual_methods.get(
            self.counterfactual_method,
            self._intervention_counterfactuals
        )
        
        return counterfactual_fn(
            model, input_ids, attention_mask, step_boundaries, original_rewards, tokenizer
        )
    
    def _intervention_counterfactuals(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        original_rewards: torch.Tensor,
        tokenizer
    ) -> Dict[str, List[Dict]]:
        """Generate counterfactuals through direct intervention"""
        
        batch_size = input_ids.size(0)
        counterfactuals = {"interventions": []}
        
        for batch_idx in range(batch_size):
            steps = step_boundaries[batch_idx]
            original_reward = original_rewards[batch_idx].item()
            
            batch_counterfactuals = []
            
            for step_idx, (start_pos, end_pos) in enumerate(steps):
                # Intervention: mask out this step
                intervened_input = input_ids[batch_idx:batch_idx+1].clone()
                intervened_mask = attention_mask[batch_idx:batch_idx+1].clone()
                
                # Replace step with neutral tokens
                intervened_input[0, start_pos:end_pos+1] = tokenizer.pad_token_id
                intervened_mask[0, start_pos:end_pos+1] = 0
                
                # Compute counterfactual reward
                with torch.no_grad():
                    cf_output = model(intervened_input, intervened_mask, return_dict=True)
                    cf_reward = cf_output.rewards.item()
                
                # Compute causal effect
                causal_effect = original_reward - cf_reward
                
                # Generate text for analysis
                original_step_text = tokenizer.decode(
                    input_ids[batch_idx, start_pos:end_pos+1],
                    skip_special_tokens=True
                )
                
                counterfactual_data = {
                    "step_index": step_idx,
                    "step_text": original_step_text,
                    "original_reward": original_reward,
                    "counterfactual_reward": cf_reward,
                    "causal_effect": causal_effect,
                    "intervention_type": "removal",
                    "necessity": causal_effect > self.causal_strength_threshold
                }
                
                batch_counterfactuals.append(counterfactual_data)
            
            counterfactuals["interventions"].append(batch_counterfactuals)
        
        return counterfactuals
    
    def _substitution_counterfactuals(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        original_rewards: torch.Tensor,
        tokenizer
    ) -> Dict[str, List[Dict]]:
        """Generate counterfactuals through step substitution"""
        
        batch_size = input_ids.size(0)
        counterfactuals = {"substitutions": []}
        
        for batch_idx in range(batch_size):
            steps = step_boundaries[batch_idx]
            original_reward = original_rewards[batch_idx].item()
            
            batch_counterfactuals = []
            
            # Generate alternative steps for substitution
            alternative_steps = self._generate_alternative_steps(
                input_ids[batch_idx], steps, tokenizer
            )
            
            for step_idx, (start_pos, end_pos) in enumerate(steps):
                original_step_text = tokenizer.decode(
                    input_ids[batch_idx, start_pos:end_pos+1],
                    skip_special_tokens=True
                )
                
                # Try different substitutions for this step
                for alt_idx, alternative_tokens in enumerate(alternative_steps[step_idx]):
                    substituted_input = input_ids[batch_idx:batch_idx+1].clone()
                    
                    # Substitute the step
                    step_length = end_pos - start_pos + 1
                    if len(alternative_tokens) == step_length:
                        substituted_input[0, start_pos:end_pos+1] = torch.tensor(
                            alternative_tokens, device=input_ids.device
                        )
                    
                    # Compute counterfactual reward
                    with torch.no_grad():
                        cf_output = model(
                            substituted_input, 
                            attention_mask[batch_idx:batch_idx+1], 
                            return_dict=True
                        )
                        cf_reward = cf_output.rewards.item()
                    
                    # Decode alternative step
                    alt_step_text = tokenizer.decode(alternative_tokens, skip_special_tokens=True)
                    
                    counterfactual_data = {
                        "step_index": step_idx,
                        "original_step": original_step_text,
                        "alternative_step": alt_step_text,
                        "original_reward": original_reward,
                        "counterfactual_reward": cf_reward,
                        "reward_change": cf_reward - original_reward,
                        "substitution_index": alt_idx,
                        "sufficiency": cf_reward > original_reward
                    }
                    
                    batch_counterfactuals.append(counterfactual_data)
                    
                    # Limit number of alternatives per step
                    if alt_idx >= 3:
                        break
            
            counterfactuals["substitutions"].append(batch_counterfactuals)
        
        return counterfactuals
    
    def _generate_alternative_steps(
        self,
        input_ids: torch.Tensor,
        steps: List[Tuple[int, int]],
        tokenizer
    ) -> List[List[List[int]]]:
        """Generate alternative reasoning steps"""
        
        alternatives = []
        
        for start_pos, end_pos in steps:
            step_tokens = input_ids[start_pos:end_pos+1].tolist()
            step_text = tokenizer.decode(step_tokens, skip_special_tokens=True)
            
            step_alternatives = []
            
            # Method 1: Paraphrase generation (simplified)
            paraphrases = self._generate_paraphrases(step_text, tokenizer)
            step_alternatives.extend(paraphrases)
            
            # Method 2: Logical variations
            logical_variants = self._generate_logical_variants(step_text, tokenizer)
            step_alternatives.extend(logical_variants)
            
            # Method 3: Random perturbations
            random_variants = self._generate_random_variants(step_tokens, tokenizer)
            step_alternatives.extend(random_variants)
            
            alternatives.append(step_alternatives)
        
        return alternatives
    
    def _generate_paraphrases(self, text: str, tokenizer) -> List[List[int]]:
        """Generate paraphrases of reasoning step"""
        
        paraphrases = []
        
        # Simple paraphrasing patterns
        doc = self.nlp(text)
        
        # Synonym replacement
        for token in doc:
            if token.pos_ in ["VERB", "ADJ", "NOUN"] and not token.is_stop:
                # Simple synonym mapping (in practice, use WordNet or neural models)
                synonyms = self._get_simple_synonyms(token.text)
                
                for synonym in synonyms:
                    paraphrased_text = text.replace(token.text, synonym)
                    paraphrase_tokens = tokenizer.encode(
                        paraphrased_text, 
                        add_special_tokens=False
                    )
                    paraphrases.append(paraphrase_tokens)
                
                # Limit paraphrases
                if len(paraphrases) >= 2:
                    break
        
        return paraphrases
    
    def _get_simple_synonyms(self, word: str) -> List[str]:
        """Get simple synonyms (simplified implementation)"""
        
        synonym_map = {
            "calculate": ["compute", "determine"],
            "find": ["discover", "locate"],
            "solve": ["resolve", "figure out"],
            "therefore": ["thus", "hence"],
            "because": ["since", "as"],
            "good": ["excellent", "great"],
            "bad": ["poor", "terrible"],
            "big": ["large", "huge"],
            "small": ["tiny", "little"]
        }
        
        return synonym_map.get(word.lower(), [])
    
    def _generate_logical_variants(self, text: str, tokenizer) -> List[List[int]]:
        """Generate logical variants of reasoning step"""
        
        variants = []
        
        # Negation
        if "not" not in text.lower():
            negated_text = f"It is not the case that {text.lower()}"
            negated_tokens = tokenizer.encode(negated_text, add_special_tokens=False)
            variants.append(negated_tokens)
        
        # Conditional form
        if not text.lower().startswith("if"):
            conditional_text = f"If we assume that {text.lower()}"
            conditional_tokens = tokenizer.encode(conditional_text, add_special_tokens=False)
            variants.append(conditional_tokens)
        
        # Question form
        if not text.endswith("?"):
            question_text = f"Could it be that {text.lower()}?"
            question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
            variants.append(question_tokens)
        
        return variants
    
    def _generate_random_variants(self, tokens: List[int], tokenizer) -> List[List[int]]:
        """Generate random variants through token perturbation"""
        
        variants = []
        
        for _ in range(2):  # Generate 2 random variants
            variant_tokens = tokens.copy()
            
            # Random token substitution
            if len(variant_tokens) > 2:
                # Choose random position (avoid first/last special tokens)
                pos = np.random.randint(1, len(variant_tokens) - 1)
                # Replace with random token from vocabulary
                variant_tokens[pos] = np.random.randint(1000, 10000)
            
            variants.append(variant_tokens)
        
        return variants
    
    def _removal_counterfactuals(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        original_rewards: torch.Tensor,
        tokenizer
    ) -> Dict[str, List[Dict]]:
        """Generate counterfactuals through step removal"""
        
        batch_size = input_ids.size(0)
        counterfactuals = {"removals": []}
        
        for batch_idx in range(batch_size):
            steps = step_boundaries[batch_idx]
            original_reward = original_rewards[batch_idx].item()
            
            batch_counterfactuals = []
            
            # Try removing different combinations of steps
            for num_remove in range(1, min(len(steps), 4) + 1):
                for step_combo in itertools.combinations(range(len(steps)), num_remove):
                    # Create input with removed steps
                    removed_input = input_ids[batch_idx:batch_idx+1].clone()
                    removed_mask = attention_mask[batch_idx:batch_idx+1].clone()
                    
                    # Remove selected steps
                    for step_idx in step_combo:
                        start_pos, end_pos = steps[step_idx]
                        removed_input[0, start_pos:end_pos+1] = tokenizer.pad_token_id
                        removed_mask[0, start_pos:end_pos+1] = 0
                    
                    # Compute counterfactual reward
                    with torch.no_grad():
                        cf_output = model(removed_input, removed_mask, return_dict=True)
                        cf_reward = cf_output.rewards.item()
                    
                    # Analyze removed steps
                    removed_steps_text = []
                    for step_idx in step_combo:
                        start_pos, end_pos = steps[step_idx]
                        step_text = tokenizer.decode(
                            input_ids[batch_idx, start_pos:end_pos+1],
                            skip_special_tokens=True
                        )
                        removed_steps_text.append(step_text)
                    
                    counterfactual_data = {
                        "removed_steps": list(step_combo),
                        "removed_steps_text": removed_steps_text,
                        "original_reward": original_reward,
                        "counterfactual_reward": cf_reward,
                        "importance_score": original_reward - cf_reward,
                        "necessity_strength": (original_reward - cf_reward) / (abs(original_reward) + 1e-8)
                    }
                    
                    batch_counterfactuals.append(counterfactual_data)
                    
                    # Limit combinations to avoid explosion
                    if len(batch_counterfactuals) >= 10:
                        break
                
                if len(batch_counterfactuals) >= 10:
                    break
            
            counterfactuals["removals"].append(batch_counterfactuals)
        
        return counterfactuals
    
    def _reordering_counterfactuals(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        original_rewards: torch.Tensor,
        tokenizer
    ) -> Dict[str, List[Dict]]:
        """Generate counterfactuals through step reordering"""
        
        batch_size = input_ids.size(0)
        counterfactuals = {"reorderings": []}
        
        for batch_idx in range(batch_size):
            steps = step_boundaries[batch_idx]
            original_reward = original_rewards[batch_idx].item()
            
            if len(steps) < 2:
                counterfactuals["reorderings"].append([])
                continue
            
            batch_counterfactuals = []
            
            # Extract step tokens
            step_tokens = []
            for start_pos, end_pos in steps:
                step_token_list = input_ids[batch_idx, start_pos:end_pos+1].tolist()
                step_tokens.append(step_token_list)
            
            # Try different permutations
            for perm in itertools.permutations(range(len(steps))):
                if perm == tuple(range(len(steps))):
                    continue  # Skip original order
                
                # Reconstruct input with reordered steps
                reordered_input = self._reconstruct_with_reordered_steps(
                    input_ids[batch_idx], steps, step_tokens, perm
                )
                
                if reordered_input is not None:
                    reordered_input = reordered_input.unsqueeze(0)
                    
                    # Compute counterfactual reward
                    with torch.no_grad():
                        cf_output = model(
                            reordered_input, 
                            attention_mask[batch_idx:batch_idx+1], 
                            return_dict=True
                        )
                        cf_reward = cf_output.rewards.item()
                    
                    # Analyze reordering
                    reordering_description = " → ".join([str(i) for i in perm])
                    
                    counterfactual_data = {
                        "original_order": list(range(len(steps))),
                        "reordered_order": list(perm),
                        "reordering_description": reordering_description,
                        "original_reward": original_reward,
                        "counterfactual_reward": cf_reward,
                        "order_sensitivity": abs(cf_reward - original_reward),
                        "improvement": cf_reward > original_reward
                    }
                    
                    batch_counterfactuals.append(counterfactual_data)
                
                # Limit permutations
                if len(batch_counterfactuals) >= 6:
                    break
            
            counterfactuals["reorderings"].append(batch_counterfactuals)
        
        return counterfactuals
    
    def _reconstruct_with_reordered_steps(
        self,
        original_input: torch.Tensor,
        steps: List[Tuple[int, int]],
        step_tokens: List[List[int]],
        permutation: Tuple[int, ...]
    ) -> Optional[torch.Tensor]:
        """Reconstruct input with reordered steps"""
        
        # Simple reconstruction: concatenate reordered steps
        reordered_tokens = []
        
        # Add tokens before first step
        if steps:
            reordered_tokens.extend(original_input[:steps[0][0]].tolist())
        
        # Add reordered steps
        for step_idx in permutation:
            reordered_tokens.extend(step_tokens[step_idx])
            reordered_tokens.append(102)  # Separator token (simplified)
        
        # Add tokens after last step
        if steps:
            reordered_tokens.extend(original_input[steps[-1][1]+1:].tolist())
        
        # Convert back to tensor
        if len(reordered_tokens) <= original_input.size(0):
            # Pad to original length
            while len(reordered_tokens) < original_input.size(0):
                reordered_tokens.append(0)  # Padding token
            
            return torch.tensor(reordered_tokens[:original_input.size(0)], device=original_input.device)
        
        return None
    
    def _semantic_perturbation_counterfactuals(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        original_rewards: torch.Tensor,
        tokenizer
    ) -> Dict[str, List[Dict]]:
        """Generate semantically perturbed counterfactuals"""
        
        batch_size = input_ids.size(0)
        counterfactuals = {"semantic_perturbations": []}
        
        for batch_idx in range(batch_size):
            steps = step_boundaries[batch_idx]
            original_reward = original_rewards[batch_idx].item()
            
            batch_counterfactuals = []
            
            for step_idx, (start_pos, end_pos) in enumerate(steps):
                step_text = tokenizer.decode(
                    input_ids[batch_idx, start_pos:end_pos+1],
                    skip_special_tokens=True
                )
                
                # Generate semantic perturbations
                perturbations = self._generate_semantic_perturbations(step_text)
                
                for perturbation in perturbations:
                    # Encode perturbation
                    perturbed_tokens = tokenizer.encode(
                        perturbation, 
                        add_special_tokens=False
                    )
                    
                    # Create perturbed input
                    perturbed_input = input_ids[batch_idx:batch_idx+1].clone()
                    
                    # Replace step (if lengths match approximately)
                    step_length = end_pos - start_pos + 1
                    if abs(len(perturbed_tokens) - step_length) <= 2:
                        # Adjust to fit
                        if len(perturbed_tokens) < step_length:
                            perturbed_tokens.extend([tokenizer.pad_token_id] * (step_length - len(perturbed_tokens)))
                        elif len(perturbed_tokens) > step_length:
                            perturbed_tokens = perturbed_tokens[:step_length]
                        
                        perturbed_input[0, start_pos:end_pos+1] = torch.tensor(
                            perturbed_tokens, device=input_ids.device
                        )
                        
                        # Compute counterfactual reward
                        with torch.no_grad():
                            cf_output = model(
                                perturbed_input, 
                                attention_mask[batch_idx:batch_idx+1], 
                                return_dict=True
                            )
                            cf_reward = cf_output.rewards.item()
                        
                        # Compute semantic similarity
                        semantic_similarity = self._compute_semantic_similarity(
                            step_text, perturbation
                        )
                        
                        counterfactual_data = {
                            "step_index": step_idx,
                            "original_text": step_text,
                            "perturbed_text": perturbation,
                            "original_reward": original_reward,
                            "counterfactual_reward": cf_reward,
                            "semantic_similarity": semantic_similarity,
                            "reward_robustness": abs(cf_reward - original_reward),
                            "stable_semantics": abs(cf_reward - original_reward) < 0.1
                        }
                        
                        batch_counterfactuals.append(counterfactual_data)
            
            counterfactuals["semantic_perturbations"].append(batch_counterfactuals)
        
        return counterfactuals
    
    def _generate_semantic_perturbations(self, text: str) -> List[str]:
        """Generate semantic perturbations of text"""
        
        perturbations = []
        
        # Entity substitution
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                # Replace with generic entity
                generic_replacements = {
                    "PERSON": "someone",
                    "ORG": "an organization", 
                    "GPE": "a place"
                }
                perturbed = text.replace(ent.text, generic_replacements[ent.label_])
                perturbations.append(perturbed)
        
        # Number perturbation
        import re
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            # Slightly change number
            new_num = str(int(num) + 1)
            perturbed = text.replace(num, new_num, 1)
            perturbations.append(perturbed)
        
        # Verb tense changes
        for token in doc:
            if token.pos_ == "VERB" and token.text.endswith("ed"):
                # Change past to present
                present_form = token.text[:-2] if not token.text.endswith("ded") else token.text[:-1]
                perturbed = text.replace(token.text, present_form)
                perturbations.append(perturbed)
        
        return perturbations[:3]  # Limit to 3 perturbations
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between texts"""
        
        # Simple word overlap similarity (in practice, use sentence embeddings)
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        words1 = set([token.lemma_.lower() for token in doc1 if not token.is_stop])
        words2 = set([token.lemma_.lower() for token in doc2 if not token.is_stop])
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _logical_negation_counterfactuals(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        original_rewards: torch.Tensor,
        tokenizer
    ) -> Dict[str, List[Dict]]:
        """Generate counterfactuals through logical negation"""
        
        batch_size = input_ids.size(0)
        counterfactuals = {"logical_negations": []}
        
        for batch_idx in range(batch_size):
            steps = step_boundaries[batch_idx]
            original_reward = original_rewards[batch_idx].item()
            
            batch_counterfactuals = []
            
            for step_idx, (start_pos, end_pos) in enumerate(steps):
                step_text = tokenizer.decode(
                    input_ids[batch_idx, start_pos:end_pos+1],
                    skip_special_tokens=True
                )
                
                # Generate logical negation
                negated_text = self._negate_statement(step_text)
                
                if negated_text and negated_text != step_text:
                    # Encode negated statement
                    negated_tokens = tokenizer.encode(
                        negated_text, 
                        add_special_tokens=False
                    )
                    
                    # Create input with negated step
                    negated_input = input_ids[batch_idx:batch_idx+1].clone()
                    
                    # Replace step if lengths are compatible
                    step_length = end_pos - start_pos + 1
                    if abs(len(negated_tokens) - step_length) <= 3:
                        # Adjust length
                        if len(negated_tokens) < step_length:
                            negated_tokens.extend([tokenizer.pad_token_id] * (step_length - len(negated_tokens)))
                        elif len(negated_tokens) > step_length:
                            negated_tokens = negated_tokens[:step_length]
                        
                        negated_input[0, start_pos:end_pos+1] = torch.tensor(
                            negated_tokens, device=input_ids.device
                        )
                        
                        # Compute counterfactual reward
                        with torch.no_grad():
                            cf_output = model(
                                negated_input, 
                                attention_mask[batch_idx:batch_idx+1], 
                                return_dict=True
                            )
                            cf_reward = cf_output.rewards.item()
                        
                        counterfactual_data = {
                            "step_index": step_idx,
                            "original_statement": step_text,
                            "negated_statement": negated_text,
                            "original_reward": original_reward,
                            "counterfactual_reward": cf_reward,
                            "logical_consistency": cf_reward < original_reward,
                            "negation_impact": original_reward - cf_reward
                        }
                        
                        batch_counterfactuals.append(counterfactual_data)
            
            counterfactuals["logical_negations"].append(batch_counterfactuals)
        
        return counterfactuals
    
    def _negate_statement(self, statement: str) -> Optional[str]:
        """Generate logical negation of statement"""
        
        statement = statement.strip()
        
        # Simple negation patterns
        if statement.lower().startswith("this is"):
            return f"This is not {statement[8:]}"
        
        if statement.lower().startswith("we can"):
            return f"We cannot {statement[7:]}"
        
        if statement.lower().startswith("the"):
            return f"It is not true that {statement.lower()}"
        
        if "is" in statement and "not" not in statement.lower():
            return statement.replace("is", "is not", 1)
        
        # Default negation
        return f"It is not the case that {statement.lower()}"
    
    def _causal_intervention_counterfactuals(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        original_rewards: torch.Tensor,
        tokenizer
    ) -> Dict[str, List[Dict]]:
        """Generate counterfactuals through causal interventions"""
        
        # Build causal model of steps
        causal_graph = self._build_step_causal_graph(
            input_ids, step_boundaries, tokenizer
        )
        
        # Generate interventions based on causal structure
        return self._generate_causal_interventions(
            model, input_ids, attention_mask, step_boundaries, 
            original_rewards, tokenizer, causal_graph
        )
    
    def _build_step_causal_graph(
        self,
        input_ids: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        tokenizer
    ) -> Dict[str, np.ndarray]:
        """Build causal graph between reasoning steps"""
        
        batch_graphs = {}
        
        for batch_idx, steps in enumerate(step_boundaries):
            n_steps = len(steps)
            causal_matrix = np.zeros((n_steps, n_steps))
            
            # Extract step texts for causal analysis
            step_texts = []
            for start_pos, end_pos in steps:
                step_text = tokenizer.decode(
                    input_ids[batch_idx, start_pos:end_pos+1],
                    skip_special_tokens=True
                )
                step_texts.append(step_text)
            
            # Build causal relationships
            for i in range(n_steps):
                for j in range(i + 1, n_steps):
                    # Temporal causality + content analysis
                    causal_strength = self._estimate_causal_strength(
                        step_texts[i], step_texts[j], i, j
                    )
                    causal_matrix[i, j] = causal_strength
            
            batch_graphs[batch_idx] = causal_matrix
        
        return batch_graphs
    
    def _estimate_causal_strength(
        self,
        cause_text: str,
        effect_text: str,
        cause_position: int,
        effect_position: int
    ) -> float:
        """Estimate causal strength between two steps"""
        
        # Temporal decay
        temporal_factor = 1.0 / (1.0 + (effect_position - cause_position))
        
        # Linguistic indicators
        effect_doc = self.nlp(effect_text.lower())
        causal_indicators = ["therefore", "thus", "hence", "because", "since", "so"]
        
        linguistic_factor = 0.0
        for token in effect_doc:
            if token.text in causal_indicators:
                linguistic_factor = 0.8
                break
        
        # Content similarity
        cause_doc = self.nlp(cause_text.lower())
        cause_words = set([token.lemma_ for token in cause_doc if not token.is_stop])
        effect_words = set([token.lemma_ for token in effect_doc if not token.is_stop])
        
        if cause_words and effect_words:
            overlap = len(cause_words.intersection(effect_words))
            total = len(cause_words.union(effect_words))
            content_factor = overlap / total
        else:
            content_factor = 0.0
        
        # Combined causal strength
        causal_strength = 0.4 * temporal_factor + 0.4 * linguistic_factor + 0.2 * content_factor
        
        return causal_strength
    
    def _generate_causal_interventions(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_boundaries: List[List[Tuple[int, int]]],
        original_rewards: torch.Tensor,
        tokenizer,
        causal_graphs: Dict[str, np.ndarray]
    ) -> Dict[str, List[Dict]]:
        """Generate interventions based on causal graph"""
        
        counterfactuals = {"causal_interventions": []}
        
        for batch_idx in range(input_ids.size(0)):
            steps = step_boundaries[batch_idx]
            causal_graph = causal_graphs[batch_idx]
            original_reward = original_rewards[batch_idx].item()
            
            batch_counterfactuals = []
            
            # Identify high-causal-impact steps
            causal_impacts = np.sum(causal_graph, axis=1)
            high_impact_steps = np.argsort(causal_impacts)[-3:]  # Top 3
            
            for step_idx in high_impact_steps:
                if step_idx < len(steps):
                    start_pos, end_pos = steps[step_idx]
                    
                    # Intervention: strengthen/weaken causal step
                    original_text = tokenizer.decode(
                        input_ids[batch_idx, start_pos:end_pos+1],
                        skip_special_tokens=True
                    )
                    
                    # Strengthen intervention
                    strengthened_text = f"Importantly, {original_text.lower()}"
                    strengthened_tokens = tokenizer.encode(
                        strengthened_text, 
                        add_special_tokens=False
                    )
                    
                    # Apply strengthening intervention
                    strengthened_input = self._apply_text_intervention(
                        input_ids[batch_idx:batch_idx+1], 
                        strengthened_tokens, 
                        start_pos, end_pos
                    )
                    
                    if strengthened_input is not None:
                        with torch.no_grad():
                            cf_output = model(
                                strengthened_input, 
                                attention_mask[batch_idx:batch_idx+1], 
                                return_dict=True
                            )
                            strengthened_reward = cf_output.rewards.item()
                        
                        counterfactual_data = {
                            "step_index": step_idx,
                            "intervention_type": "strengthen",
                            "original_text": original_text,
                            "intervened_text": strengthened_text,
                            "original_reward": original_reward,
                            "counterfactual_reward": strengthened_reward,
                            "causal_impact": causal_impacts[step_idx],
                            "intervention_effect": strengthened_reward - original_reward
                        }
                        
                        batch_counterfactuals.append(counterfactual_data)
            
            counterfactuals["causal_interventions"].append(batch_counterfactuals)
        
        return counterfactuals
    
    def _apply_text_intervention(
        self,
        input_ids: torch.Tensor,
        intervention_tokens: List[int],
        start_pos: int,
        end_pos: int
    ) -> Optional[torch.Tensor]:
        """Apply text intervention to input"""
        
        step_length = end_pos - start_pos + 1
        
        if abs(len(intervention_tokens) - step_length) <= 5:
            intervened_input = input_ids.clone()
            
            # Adjust intervention to fit
            if len(intervention_tokens) < step_length:
                intervention_tokens.extend([0] * (step_length - len(intervention_tokens)))
            elif len(intervention_tokens) > step_length:
                intervention_tokens = intervention_tokens[:step_length]
            
            intervened_input[0, start_pos:end_pos+1] = torch.tensor(
                intervention_tokens, device=input_ids.device
            )
            
            return intervened_input
        
        return None

class CounterfactualEvaluator:
    """Evaluates quality and insights from counterfactual analysis"""
    
    def evaluate_counterfactuals(
        self,
        counterfactuals: Dict[str, List[Dict]],
        original_rewards: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate counterfactual analysis results"""
        
        evaluation = {}
        
        for cf_type, cf_data in counterfactuals.items():
            type_evaluation = self._evaluate_counterfactual_type(cf_data, original_rewards)
            evaluation[cf_type] = type_evaluation
        
        return evaluation
    
    def _evaluate_counterfactual_type(
        self,
        cf_data: List[Dict],
        original_rewards: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate specific type of counterfactual"""
        
        if not cf_data:
            return {"coverage": 0.0, "diversity": 0.0, "impact": 0.0}
        
        all_effects = []
        reward_changes = []
        
        for batch_cfs in cf_data:
            for cf in batch_cfs:
                if "causal_effect" in cf:
                    all_effects.append(abs(cf["causal_effect"]))
                elif "reward_change" in cf:
                    reward_changes.append(abs(cf["reward_change"]))
                elif "importance_score" in cf:
                    all_effects.append(abs(cf["importance_score"]))
        
        # Coverage: how many significant effects found
        significant_effects = sum(1 for effect in all_effects if effect > 0.1)
        coverage = significant_effects / len(all_effects) if all_effects else 0.0
        
        # Diversity: variance in effect sizes
        diversity = np.var(all_effects) if all_effects else 0.0
        
        # Impact: average effect size
        impact = np.mean(all_effects) if all_effects else 0.0
        
        return {
            "coverage": coverage,
            "diversity": diversity,
            "impact": impact,
            "num_counterfactuals": len(all_effects)
        }