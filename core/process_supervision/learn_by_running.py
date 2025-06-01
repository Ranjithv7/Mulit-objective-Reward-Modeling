"""
🔥 PROCESS SUPERVISION TEST SUITE 🔥
Educational test file to understand process supervision components by running and seeing outputs!

This file demonstrates:
1. Process Reward Model (OpenAI-style PRM)
2. Stepwise Reward Assignment
3. Temporal Credit Assignment  
4. Counterfactual Reasoning

Run each section to see how the components work!
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Import your process supervision modules
# (Assume these are in the same directory or properly installed)
# from process_reward_model import ProcessRewardModel, HierarchicalProcessModel
# from stepwise_rewards import StepwiseRewardAssigner, AdaptiveStepRewardModel
# from credit_assignment import TemporalCreditAssigner, LearnedCreditAssigner
# from counterfactual_reasoning import CounterfactualReasoningModel

def create_mock_data():
    """Create realistic mock data for testing."""
    print("🎯 Creating Mock Data...")
    
    # Mock reasoning sequence: "Solve 2x + 3 = 7"
    batch_size = 2
    num_steps = 5
    hidden_size = 256
    
    # Step descriptions for understanding
    step_descriptions = [
        "Problem: 2x + 3 = 7",
        "Subtract 3 from both sides: 2x = 4", 
        "Divide by 2: x = 2",
        "Verify: 2(2) + 3 = 7 ✓",
        "Conclusion: x = 2"
    ]
    
    # Mock step features (normally from transformer)
    step_sequence = torch.randn(batch_size, num_steps, hidden_size)
    
    # Mock attention mask (all steps are valid)
    attention_mask = torch.ones(batch_size, num_steps)
    
    # Mock step correctness scores
    step_correctness = torch.tensor([
        [0.9, 0.95, 0.98, 0.85, 0.92],  # First example
        [0.8, 0.90, 0.88, 0.95, 0.89]   # Second example
    ])
    
    # Mock step importance scores
    step_importance = torch.tensor([
        [0.7, 0.9, 0.95, 0.8, 0.85],   # Step 2&3 most important
        [0.75, 0.85, 0.92, 0.88, 0.9]
    ])
    
    # Final outcomes (reward for solving correctly)
    final_outcomes = torch.tensor([0.9, 0.85])  # Both solved correctly
    
    print(f"✅ Created data for {batch_size} examples with {num_steps} steps each")
    print(f"📝 Step descriptions: {step_descriptions}")
    
    return {
        'step_sequence': step_sequence,
        'attention_mask': attention_mask,
        'step_correctness': step_correctness,
        'step_importance': step_importance,
        'final_outcomes': final_outcomes,
        'step_descriptions': step_descriptions,
        'batch_size': batch_size,
        'num_steps': num_steps,
        'hidden_size': hidden_size
    }

def test_process_reward_model(data):
    """Test the Process Reward Model (OpenAI-style PRM)."""
    print("\n" + "="*60)
    print("🔥 TESTING PROCESS REWARD MODEL (OpenAI-style PRM)")
    print("="*60)
    
    # Create a simple mock process reward model
    class MockProcessRewardModel(nn.Module):
        def __init__(self, hidden_size, num_objectives=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_objectives = num_objectives
            
            # Process head for step-by-step rewards
            self.process_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_objectives)
            )
            
            # Outcome head for final reward
            self.outcome_head = nn.Linear(hidden_size, num_objectives)
            
        def forward(self, step_sequence, attention_mask):
            # Process-level rewards for each step
            step_rewards = self.process_head(step_sequence)  # [batch, steps, objectives]
            
            # Final outcome reward (use last step)
            final_hidden = step_sequence[:, -1, :]  # Last step
            outcome_reward = self.outcome_head(final_hidden)  # [batch, objectives]
            
            return {
                'step_rewards': step_rewards,
                'outcome_reward': outcome_reward,
                'combined_reward': step_rewards.mean(dim=1) + outcome_reward
            }
    
    # Initialize model
    model = MockProcessRewardModel(data['hidden_size'])
    
    print("🏗️ Created Process Reward Model")
    print(f"   - Input: step_sequence {data['step_sequence'].shape}")
    print(f"   - Input: attention_mask {data['attention_mask'].shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(data['step_sequence'], data['attention_mask'])
    
    print("\n📊 RESULTS:")
    print(f"   Step Rewards Shape: {output['step_rewards'].shape}")
    print(f"   Outcome Reward Shape: {output['outcome_reward'].shape}")
    
    # Show step-by-step rewards for first example
    print(f"\n🔍 Example 1 - Step-by-Step Rewards:")
    for i, desc in enumerate(data['step_descriptions']):
        reward = output['step_rewards'][0, i, 0].item()
        print(f"   Step {i+1}: {reward:.3f} | {desc}")
    
    print(f"\n🎯 Final Outcome Reward: {output['outcome_reward'][0, 0].item():.3f}")
    print(f"🏆 Combined Reward: {output['combined_reward'][0, 0].item():.3f}")
    
    return output

def test_stepwise_reward_assignment(data):
    """Test Stepwise Reward Assignment strategies."""
    print("\n" + "="*60)
    print("🔥 TESTING STEPWISE REWARD ASSIGNMENT")
    print("="*60)
    
    # Create stepwise reward assigner
    class MockStepwiseAssigner:
        def __init__(self):
            self.strategies = [
                "progressive", "uniform", "importance_weighted", 
                "exponential_decay", "critical_path"
            ]
        
        def assign_rewards(self, final_reward, step_correctness, step_importance, strategy):
            num_steps = step_correctness.shape[1]
            step_rewards = torch.zeros_like(step_correctness)
            
            if strategy == "progressive":
                # Later steps get higher weights
                for i in range(num_steps):
                    weight = (i + 1) / num_steps
                    step_rewards[:, i] = final_reward * weight * step_correctness[:, i]
                    
            elif strategy == "uniform":
                # Equal weights
                base_reward = final_reward.unsqueeze(1) / num_steps
                step_rewards = base_reward * step_correctness
                
            elif strategy == "importance_weighted":
                # Weight by importance
                total_importance = step_importance.sum(dim=1, keepdim=True)
                weights = step_importance / (total_importance + 1e-8)
                step_rewards = final_reward.unsqueeze(1) * weights * step_correctness
                
            elif strategy == "exponential_decay":
                # Recent steps get higher rewards
                decay_factor = 0.9
                for i in range(num_steps):
                    steps_from_end = num_steps - i - 1
                    weight = decay_factor ** steps_from_end
                    step_rewards[:, i] = final_reward * weight * step_correctness[:, i]
                    
            elif strategy == "critical_path":
                # High importance steps get more reward
                critical_threshold = step_importance.quantile(0.7, dim=1, keepdim=True)
                is_critical = (step_importance >= critical_threshold).float()
                weights = 0.8 * is_critical + 0.2 * (1 - is_critical)
                step_rewards = final_reward.unsqueeze(1) * weights * step_correctness
            
            return step_rewards
    
    assigner = MockStepwiseAssigner()
    
    print("🎯 Testing different reward assignment strategies...")
    
    results = {}
    for strategy in assigner.strategies:
        step_rewards = assigner.assign_rewards(
            data['final_outcomes'], 
            data['step_correctness'], 
            data['step_importance'], 
            strategy
        )
        results[strategy] = step_rewards
        
        print(f"\n📈 {strategy.upper()} Strategy:")
        print(f"   Example 1 step rewards: {step_rewards[0].tolist()}")
        print(f"   Total reward: {step_rewards[0].sum().item():.3f} (target: {data['final_outcomes'][0].item():.3f})")
    
    # Compare strategies visually
    print(f"\n🔍 STRATEGY COMPARISON (Example 1):")
    print("Step |", end="")
    for i, desc in enumerate(data['step_descriptions']):
        print(f" {i+1:>6} |", end="")
    print()
    print("-" * 50)
    
    for strategy, rewards in results.items():
        print(f"{strategy:12} |", end="")
        for reward in rewards[0]:
            print(f" {reward.item():6.3f} |", end="")
        print()
    
    return results

def test_temporal_credit_assignment(data):
    """Test Temporal Credit Assignment methods."""
    print("\n" + "="*60)
    print("🔥 TESTING TEMPORAL CREDIT ASSIGNMENT")
    print("="*60)
    
    class MockTemporalCreditAssigner:
        def __init__(self):
            self.methods = [
                "exponential_decay", "eligibility_traces", 
                "causal_influence", "attention_based"
            ]
        
        def assign_credit(self, step_sequence, final_outcomes, method):
            batch_size, num_steps, _ = step_sequence.shape
            credit_assignments = torch.zeros(batch_size, num_steps)
            
            if method == "exponential_decay":
                # Recent steps get more credit
                decay_factor = 0.9
                for b in range(batch_size):
                    for t in range(num_steps):
                        steps_from_end = num_steps - t - 1
                        weight = decay_factor ** steps_from_end
                        credit_assignments[b, t] = final_outcomes[b] * weight
                    # Normalize
                    total = credit_assignments[b].sum()
                    if total > 0:
                        credit_assignments[b] = credit_assignments[b] * (final_outcomes[b] / total)
                        
            elif method == "eligibility_traces":
                # Combination of recency and importance
                lambda_trace = 0.8
                for b in range(batch_size):
                    trace = torch.zeros(num_steps)
                    for t in range(num_steps):
                        trace *= decay_factor * lambda_trace
                        trace[t] = 1.0  # Current step gets full eligibility
                        if t == num_steps - 1:  # Final step
                            credit_assignments[b] = final_outcomes[b] * trace
                            
            elif method == "causal_influence":
                # Steps that influence later steps get more credit
                for b in range(batch_size):
                    for t in range(num_steps):
                        # Simple causal model: earlier steps influence later ones
                        influence = 0.0
                        for future_t in range(t, num_steps):
                            decay = 0.9 ** (future_t - t)
                            influence += decay
                        credit_assignments[b, t] = final_outcomes[b] * influence / num_steps
                        
            elif method == "attention_based":
                # Use attention weights (mock)
                # Simulate attention from final step to all previous steps
                attention_weights = torch.softmax(torch.randn(num_steps), dim=0)
                for b in range(batch_size):
                    credit_assignments[b] = final_outcomes[b] * attention_weights
            
            return credit_assignments
    
    assigner = MockTemporalCreditAssigner()
    
    print("⏰ Testing temporal credit assignment methods...")
    
    results = {}
    for method in assigner.methods:
        credits = assigner.assign_credit(
            data['step_sequence'], 
            data['final_outcomes'], 
            method
        )
        results[method] = credits
        
        print(f"\n📊 {method.upper()} Method:")
        print(f"   Example 1 credits: {credits[0].tolist()}")
        print(f"   Total credit: {credits[0].sum().item():.3f}")
    
    # Show credit distribution
    print(f"\n🔍 CREDIT ASSIGNMENT COMPARISON (Example 1):")
    print("Step |", end="")
    for i in range(data['num_steps']):
        print(f" {i+1:>8} |", end="")
    print()
    print("-" * 60)
    
    for method, credits in results.items():
        print(f"{method:15} |", end="")
        for credit in credits[0]:
            print(f" {credit.item():8.4f} |", end="")
        print()
    
    return results

def test_counterfactual_reasoning(data):
    """Test Counterfactual Reasoning analysis."""
    print("\n" + "="*60)
    print("🔥 TESTING COUNTERFACTUAL REASONING")
    print("="*60)
    
    class MockCounterfactualAnalyzer:
        def __init__(self):
            self.intervention_types = ["removal", "replacement", "noise"]
        
        def analyze_counterfactuals(self, step_sequence, final_outcomes, step_descriptions):
            batch_size, num_steps, hidden_size = step_sequence.shape
            
            results = {}
            
            for intervention in self.intervention_types:
                step_effects = torch.zeros(batch_size, num_steps)
                
                for step_idx in range(num_steps):
                    # Create counterfactual by modifying step
                    counterfactual_outcome = self._simulate_intervention(
                        step_idx, intervention, final_outcomes, step_descriptions
                    )
                    
                    # Effect = original - counterfactual
                    step_effects[:, step_idx] = final_outcomes - counterfactual_outcome
                
                results[intervention] = step_effects
            
            # Necessity and sufficiency analysis
            necessity_scores = self._analyze_necessity(results["removal"], final_outcomes)
            sufficiency_scores = self._analyze_sufficiency(results["replacement"], final_outcomes)
            
            results["necessity"] = necessity_scores
            results["sufficiency"] = sufficiency_scores
            
            return results
        
        def _simulate_intervention(self, step_idx, intervention, original_outcomes, descriptions):
            """Simulate what would happen if we intervene on a step."""
            
            # Mock intervention effects based on step importance
            step_importance_map = {
                0: 0.3,  # Problem statement - medium importance
                1: 0.8,  # Subtract 3 - high importance  
                2: 0.9,  # Divide by 2 - highest importance
                3: 0.6,  # Verification - medium-high importance
                4: 0.4   # Conclusion - medium importance
            }
            
            importance = step_importance_map.get(step_idx, 0.5)
            
            if intervention == "removal":
                # Removing important steps hurts performance more
                effect = importance * 0.5
                return original_outcomes - effect
            elif intervention == "replacement":
                # Replacing with random step
                effect = importance * 0.3
                return original_outcomes - effect
            elif intervention == "noise":
                # Adding noise has smaller effect
                effect = importance * 0.1
                return original_outcomes - effect
                
            return original_outcomes
        
        def _analyze_necessity(self, removal_effects, original_outcomes):
            """Analyze how necessary each step is."""
            # High removal effect = high necessity
            return torch.sigmoid(removal_effects * 2)  # Scale and normalize
        
        def _analyze_sufficiency(self, replacement_effects, original_outcomes):
            """Analyze how sufficient each step is."""
            # Lower replacement effect = higher sufficiency  
            return torch.sigmoid(-replacement_effects * 2 + 1)
    
    analyzer = MockCounterfactualAnalyzer()
    
    print("🔮 Analyzing counterfactual scenarios...")
    
    results = analyzer.analyze_counterfactuals(
        data['step_sequence'], 
        data['final_outcomes'], 
        data['step_descriptions']
    )
    
    print(f"\n📊 COUNTERFACTUAL ANALYSIS (Example 1):")
    print("-" * 80)
    
    for i, desc in enumerate(data['step_descriptions']):
        print(f"\nStep {i+1}: {desc}")
        print(f"  Removal Effect:    {results['removal'][0, i].item():6.3f}")
        print(f"  Replacement Effect: {results['replacement'][0, i].item():6.3f}")
        print(f"  Necessity Score:   {results['necessity'][0, i].item():6.3f}")
        print(f"  Sufficiency Score: {results['sufficiency'][0, i].item():6.3f}")
    
    # Generate explanations
    print(f"\n🔍 COUNTERFACTUAL EXPLANATIONS:")
    for i, desc in enumerate(data['step_descriptions']):
        necessity = results['necessity'][0, i].item()
        sufficiency = results['sufficiency'][0, i].item()
        
        if necessity > 0.7:
            print(f"  ⚠️  Step {i+1} is NECESSARY: Without '{desc}', performance drops significantly")
        if sufficiency > 0.7:
            print(f"  ✅ Step {i+1} is SUFFICIENT: '{desc}' alone contributes significantly to success")
    
    return results

def visualize_results(process_output,  data):
    """Create visualizations to understand the results."""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Process Supervision Analysis Results', fontsize=16)
        
        steps = list(range(1, data['num_steps'] + 1))
        
        # 1. Process Rewards
        ax1 = axes[0, 0]
        step_rewards = process_output['step_rewards'][0, :, 0].detach().numpy()
        ax1.bar(steps, step_rewards, alpha=0.7, color='blue')
        ax1.set_title('Process-Level Step Rewards')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        
        # # 2. Stepwise Assignment Comparison
        # ax2 = axes[0, 1]
        # strategies = ['progressive', 'uniform', 'importance_weighted']
        # for i, strategy in enumerate(strategies):
        #     rewards = stepwise_results[strategy][0].detach().numpy()
        #     ax2.plot(steps, rewards, marker='o', label=strategy)
        # ax2.set_title('Stepwise Assignment Strategies')
        # ax2.set_xlabel('Step')
        # ax2.set_ylabel('Assigned Reward')
        # ax2.legend()
        
        # # 3. Credit Assignment
        # ax3 = axes[1, 0]
        # methods = ['exponential_decay', 'eligibility_traces']
        # for method in methods:
        #     credits = credit_results[method][0].detach().numpy()
        #     ax3.plot(steps, credits, marker='s', label=method)
        # ax3.set_title('Temporal Credit Assignment')
        # ax3.set_xlabel('Step')
        # ax3.set_ylabel('Credit')
        # ax3.legend()
        
        # # 4. Counterfactual Analysis
        # ax4 = axes[1, 1]
        # necessity = counterfactual_results['necessity'][0].detach().numpy()
        # sufficiency = counterfactual_results['sufficiency'][0].detach().numpy()
        
        # x = np.arange(len(steps))
        # width = 0.35
        # ax4.bar(x - width/2, necessity, width, label='Necessity', alpha=0.7)
        # ax4.bar(x + width/2, sufficiency, width, label='Sufficiency', alpha=0.7)
        # ax4.set_title('Necessity vs Sufficiency')
        # ax4.set_xlabel('Step')
        # ax4.set_ylabel('Score')
        # ax4.set_xticks(x)
        # ax4.set_xticklabels(steps)
        # ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualizations created! Check the plots to see:")
        print("   - How process rewards vary across steps")
        print("   - How different assignment strategies compare")
        print("   - How credit is distributed temporally")
        print("   - Which steps are necessary vs sufficient")
        
    except ImportError:
        print("⚠️ Matplotlib not available, skipping visualizations")
        print("   Install with: pip install matplotlib")

def run_comprehensive_demo():
    """Run the complete process supervision demo."""
    print("🚀 PROCESS SUPERVISION COMPREHENSIVE DEMO")
    print("=" * 80)
    print("This demo shows how each component works with the same reasoning example!")
    print("Example: Solving the equation '2x + 3 = 7' step by step")
    print("=" * 80)
    
    # Create mock data
    data = create_mock_data()
    
    # Test each component
    process_output = test_process_reward_model(data)
    # stepwise_results = test_stepwise_reward_assignment(data)
    # credit_results = test_temporal_credit_assignment(data)
    # counterfactual_results = test_counterfactual_reasoning(data)
    
    # Create visualizations
    # visualize_results(process_output, stepwise_results, credit_results, counterfactual_results, data)
    visualize_results(process_output, data)

    
    # Summary insights
    print("\n" + "="*60)
    print("🎯 KEY INSIGHTS FROM THE DEMO")
    print("="*60)
    print("1. PROCESS REWARDS: Each step gets evaluated individually")
    print("2. STEPWISE ASSIGNMENT: Different strategies for distributing final reward")
    print("3. CREDIT ASSIGNMENT: Temporal methods assign credit based on timing/causality")
    print("4. COUNTERFACTUAL: 'What if' analysis shows step importance")
    print("\n🏆 This is how process supervision enables fine-grained reward modeling!")

if __name__ == "__main__":
    # Run the comprehensive demo
    run_comprehensive_demo()
    
    print("\n" + "🔥" * 30)
    print("DEMO COMPLETE! Now you understand:")
    print("✅ How Process Reward Models work")
    print("✅ Different stepwise reward assignment strategies") 
    print("✅ Temporal credit assignment methods")
    print("✅ Counterfactual reasoning for step analysis")
    print("🔥" * 30)