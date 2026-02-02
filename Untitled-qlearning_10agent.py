import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from collections import defaultdict

# For reproducibility
random.seed(42)
np.random.seed(42)

# ============================================================================
# COST FUNCTIONS (same as baseline)
# ============================================================================

def cost_A(n_A):
    """Cost on Route A if n_A agents choose it"""
    return 10 + 15 * n_A if n_A > 0 else 0

def cost_B(n_B):
    """Cost on Route B if n_B agents choose it"""
    return 30 + 5 * n_B if n_B > 0 else 0

# ============================================================================
# Q-LEARNING AGENT (10-agent version)
# ============================================================================

class QLearningAgent10:
    """Q-learning agent for 10-agent congestion game with aggregated state"""
    
    def __init__(self, agent_id, alpha=0.1, epsilon=0.1, gamma=0.9):
        self.agent_id = agent_id
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Q-table: Q[n_A_last_round][my_action]
        # State = number of agents on A in previous round (0-10)
        # Action = my route choice (A or B)
        self.Q = defaultdict(lambda: {'A': 0.0, 'B': 0.0})
        
        # Initialize for all 11 possible states
        for n_A in range(11):
            self.Q[n_A] = {'A': 0.0, 'B': 0.0}
        
        self.last_action = None
    
    def choose_action(self, state):
        """
        ε-greedy action selection
        state: n_A from previous round (0-10)
        """
        if random.random() < self.epsilon:
            return random.choice(['A', 'B'])
        else:
            q_vals = self.Q[state]
            if q_vals['A'] > q_vals['B']:
                return 'A'
            elif q_vals['B'] > q_vals['A']:
                return 'B'
            else:
                return random.choice(['A', 'B'])
    
    def update(self, state, action, reward, next_state):
        """
        Q-learning update
        state: n_A from previous round
        action: 'A' or 'B'
        reward: -cost
        next_state: n_A from current round
        """
        current_q = self.Q[state][action]
        max_next_q = max(self.Q[next_state].values())
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[state][action] = new_q
        
        self.last_action = action

# ============================================================================
# SIMULATION
# ============================================================================
def generate_latex_table(results, file_name='summary_table.tex'):
    """Generate LaTeX table from results for report inclusion"""
    df = pd.DataFrame({
        'Experiment': [name for name, _ in results],
        'Avg n_A': [analysis['avg_nA'] for _, analysis in results],
        'Std': [analysis['std_nA'] for _, analysis in results],
        'Near-Nash %': [analysis['near_nash_pct'] for _, analysis in results],
        'Avg Cost': [analysis['avg_cost'] for _, analysis in results],
        'Gap': [analysis['welfare_gap'] for _, analysis in results]
    })
    latex = df.to_latex(index=False, float_format="%.2f", caption="Summary of Ten-Agent Q-Learning Results (Last 100 Episodes)", label="tab:10agent_summary")
    with open(file_name, 'w') as f:
        f.write(latex)
    print(f"\n✓ Saved LaTeX table: {file_name}")

def generate_scaling_latex(n10_results, n2_data, file_name='scaling_table.tex'):
    # n2_data = list of dicts like [{'experiment': 'Baseline', 'nash_pct': 87, 'gap': 6.30, 'std': None}, ...]
    df_n2 = pd.DataFrame(n2_data)
    df_n10 = pd.DataFrame({
        'Experiment': [name for name, _ in n10_results],
        'Nash%': [analysis['near_nash_pct'] for _, analysis in n10_results],
        'Gap': [analysis['welfare_gap'] for _, analysis in n10_results],
        'Std': [analysis['std_nA'] for _, analysis in n10_results]
    })
    df = pd.concat([df_n2, df_n10], axis=1, keys=['N=2', 'N=10'])
    latex = df.to_latex(multirow=True, float_format="%.2f", caption="Scaling Comparison: 2-Agent vs 10-Agent Results", label="tab:scaling_comparison")
    with open(file_name, 'w') as f:
        f.write(latex)
    print(f"\n✓ Saved LaTeX scaling table: {file_name}")

def plot_welfare_gaps(results, file_name='welfare_gaps.png'):
    """Bar plot of welfare gaps for report"""
    experiments = [name for name, _ in results]
    gaps = [analysis['welfare_gap'] for _, analysis in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(experiments, gaps, color='skyblue')
    ax.set_ylabel('Welfare Gap (+ above Nash)')
    ax.set_title('Welfare Gaps by Experiment (N=10)')
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(file_name, dpi=200)
    print(f"\n✓ Saved: {file_name}")

def run_10agent_experiment(episodes=3000, alpha=0.1, epsilon=0.1, gamma=0.9, verbose=True):
    """Run Q-learning with 10 agents"""
    
    N = 10
    agents = [QLearningAgent10(i, alpha, epsilon, gamma) for i in range(N)]
    
    # History tracking
    history = {
        'n_A': [],           # agents on A each episode
        'n_B': [],           # agents on B each episode
        'social_cost': [],   # total cost each episode
        'costs': []          # individual costs each episode
    }
    
    # Initial state: everyone on A (arbitrary choice)
    state = 10  # n_A from "previous" round
    
    for episode in range(episodes):
        # Each agent chooses action based on previous n_A
        actions = [agent.choose_action(state) for agent in agents]
        
        # Count agents on each route
        n_A = actions.count('A')
        n_B = actions.count('B')
        
        # Calculate costs
        c_A = cost_A(n_A)
        c_B = cost_B(n_B)
        
        costs = []
        for action in actions:
            cost = c_A if action == 'A' else c_B
            costs.append(cost)
        
        # Each agent updates Q-table
        for i, agent in enumerate(agents):
            reward = -costs[i]
            agent.update(state=state, action=actions[i], reward=reward, next_state=n_A)
        
        # Record history
        history['n_A'].append(n_A)
        history['n_B'].append(n_B)
        history['social_cost'].append(sum(costs))
        history['costs'].append(costs)
        
        # Update state for next round
        state = n_A
        
        # Progress reporting
        if verbose and (episode + 1) % 500 == 0:
            recent_nA = history['n_A'][-100:]
            avg_nA = np.mean(recent_nA)
            std_nA = np.std(recent_nA)
            avg_social = np.mean(history['social_cost'][-100:])
            
            print(f"Episode {episode+1:4d} | Avg n_A: {avg_nA:4.1f}±{std_nA:3.1f} | "
                  f"Social cost: {avg_social:6.1f}")
    return agents, history

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_10agent_results(history, nash_nA=3, nash_cost=62.0, experiment_name=""):
    """Analyze convergence and welfare for 10-agent experiment"""
    
    print("\n" + "="*80)
    print(f"ANALYSIS: {experiment_name}")
    print("="*80)
    
    # Last 100 episodes statistics
    recent_nA = history['n_A'][-100:]
    recent_costs = history['social_cost'][-100:]
    
    avg_nA = np.mean(recent_nA)
    std_nA = np.std(recent_nA)
    avg_social = np.mean(recent_costs)
    avg_per_agent = avg_social / 10
    
    # Convergence assessment
    nash_tolerance = 1.0  # Within ±1 agent of Nash
    near_nash_count = sum(1 for n in recent_nA if abs(n - nash_nA) <= nash_tolerance)
    near_nash_pct = near_nash_count / 100 * 100
    
    # Convergence quality
    if std_nA < 1.0:
        convergence = "✓ Strong convergence"
    elif std_nA < 2.0:
        convergence = "~ Moderate convergence"
    else:
        convergence = "✗ Weak/no convergence"
    
    print(f"\n{convergence} (std={std_nA:.2f})")
    print(f"\nLast 100 episodes:")
    print(f"  Average n_A: {avg_nA:.2f} (Nash={nash_nA})")
    print(f"  Std dev n_A: {std_nA:.2f}")
    print(f"  Near-Nash %: {near_nash_pct:.1f}% (within ±{nash_tolerance} of Nash)")
    
    print(f"\nWelfare analysis:")
    print(f"  Nash avg cost: {nash_cost:.2f}")
    print(f"  Q-learning avg cost: {avg_per_agent:.2f}")
    print(f"  Welfare gap: +{avg_per_agent - nash_cost:.2f} ({(avg_per_agent/nash_cost - 1)*100:.1f}%)")
    
    return {
        'avg_nA': avg_nA,
        'std_nA': std_nA,
        'near_nash_pct': near_nash_pct,
        'avg_cost': avg_per_agent,
        'welfare_gap': avg_per_agent - nash_cost,
        'convergence': convergence
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_10agent_convergence(histories, experiment_names):
    """Plot convergence for multiple 10-agent experiments"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    nash_nA = 3
    
    for idx, (history, name) in enumerate(zip(histories, experiment_names)):
        ax = axes[idx // 2, idx % 2]
        
        episodes = range(len(history['n_A']))
        
        # Plot n_A over time
        ax.plot(history['n_A'], alpha=0.3, linewidth=0.5, color='steelblue', label='n_A (raw)')
        
        # Rolling average
        window = 100
        rolling_nA = [np.mean(history['n_A'][max(0, i-window):i+1]) 
                      for i in range(len(history['n_A']))]
        ax.plot(rolling_nA, color='darkblue', linewidth=2, label='n_A (rolling avg)')
        
        # Nash equilibrium line
        ax.axhline(y=nash_nA, color='green', linestyle='--', linewidth=2, 
                   label=f'Nash (n_A={nash_nA})')
        
        # Tolerance band
        ax.axhspan(nash_nA - 1, nash_nA + 1, alpha=0.1, color='green')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Number on Route A')
        ax.set_title(f'{name}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.5, 10.5)
    
    plt.tight_layout()
    plt.savefig('10agent_convergence.png', dpi=200)
    print("\n✓ Saved: 10agent_convergence.png")

# ============================================================================
# RUN ALL EXPERIMENTS
# ============================================================================

def run_all_10agent_experiments():
    """Run all 4 parameter configurations for 10 agents"""
    
    print("="*80)
    print("10-AGENT Q-LEARNING EXPERIMENTS")
    print("="*80)
    print("Nash Equilibrium: 3 on A, 7 on B | Avg cost = 62.00")
    print("="*80)
    
    results = []
    histories = []
    names = []
    
    # Experiment 1: Baseline
    print("\n### EXPERIMENT 1: Baseline (α=0.1, ε=0.1, γ=0.9) ###")
    agents1, hist1 = run_10agent_experiment(episodes=3000, alpha=0.1, epsilon=0.1, gamma=0.9)
    analysis1 = analyze_10agent_results(hist1, experiment_name="Baseline")
    results.append(('Baseline', analysis1))
    histories.append(hist1)
    names.append('Baseline (α=0.1, ε=0.1, γ=0.9)')
    
    # Experiment 2: Fast Learning
    print("\n### EXPERIMENT 2: Fast Learning (α=0.3, ε=0.1, γ=0.9) ###")
    agents2, hist2 = run_10agent_experiment(episodes=3000, alpha=0.3, epsilon=0.1, gamma=0.9)
    analysis2 = analyze_10agent_results(hist2, experiment_name="Fast Learning")
    results.append(('Fast Learning', analysis2))
    histories.append(hist2)
    names.append('Fast Learning (α=0.3, ε=0.1, γ=0.9)')
    
    # Experiment 3: High Exploration
    print("\n### EXPERIMENT 3: High Exploration (α=0.1, ε=0.3, γ=0.9) ###")
    agents3, hist3 = run_10agent_experiment(episodes=3000, alpha=0.1, epsilon=0.3, gamma=0.9)
    analysis3 = analyze_10agent_results(hist3, experiment_name="High Exploration")
    results.append(('High Exploration', analysis3))
    histories.append(hist3)
    names.append('High Exploration (α=0.1, ε=0.3, γ=0.9)')
    
    # Experiment 4: Myopic (CRITICAL TEST)
    print("\n### EXPERIMENT 4: Myopic (α=0.1, ε=0.1, γ=0.0) ###")
    agents4, hist4 = run_10agent_experiment(episodes=3000, alpha=0.1, epsilon=0.1, gamma=0.0)
    analysis4 = analyze_10agent_results(hist4, experiment_name="Myopic (γ=0)")
    results.append(('Myopic', analysis4))
    histories.append(hist4)
    names.append('Myopic (α=0.1, ε=0.1, γ=0.0)')
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE (N=10)")
    print("="*80)
    print(f"{'Experiment':<20} {'Avg n_A':<10} {'Std':<8} {'Near-Nash %':<12} {'Avg Cost':<10} {'Gap':<8}")
    print("-"*80)
    for name, analysis in results:
        print(f"{name:<20} {analysis['avg_nA']:>8.2f}   {analysis['std_nA']:>6.2f}   "
              f"{analysis['near_nash_pct']:>10.1f}%   {analysis['avg_cost']:>8.2f}   "
              f"+{analysis['welfare_gap']:>5.2f}")
    
    # Generate plots and tables
    plot_10agent_convergence(histories, names)
    generate_latex_table(results)
    plot_welfare_gaps(results)
    
    # Scaling table (with mock N=2 data)
    n2_data = [
        {'experiment': 'Baseline', 'nash_pct': 87, 'gap': 6.30, 'std': None},
        {'experiment': 'Fast', 'nash_pct': 92, 'gap': 5.80, 'std': None},
        {'experiment': 'High-ε', 'nash_pct': 49, 'gap': 10.10, 'std': None},
        {'experiment': 'Myopic', 'nash_pct': 88, 'gap': 6.20, 'std': None}
    ]
    generate_scaling_latex(results, n2_data)
    
    print("\n" + "="*80)
    print("HYPOTHESIS TESTING")
    print("="*80)
    
    # Test key hypotheses
    baseline_gap = results[0][1]['welfare_gap']
    myopic_gap = results[3][1]['welfare_gap']
    
    print(f"\nHypothesis: Myopic learning yields >10% welfare gap")
    print(f"  Result: {myopic_gap:.2f} ({(myopic_gap/62)*100:.1f}%)")
    print(f"  Status: {'✓ CONFIRMED' if myopic_gap > 6.2 else '✗ REJECTED'}")
    
    print(f"\nHypothesis: γ=0 worse than γ=0.9")
    print(f"  γ=0.9 gap: {baseline_gap:.2f}")
    print(f"  γ=0.0 gap: {myopic_gap:.2f}")
    print(f"  Difference: +{myopic_gap - baseline_gap:.2f}")
    print(f"  Status: {'✓ CONFIRMED' if myopic_gap > baseline_gap else '✗ REJECTED'}")
    
    return results, histories

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results, histories = run_all_10agent_experiments()
    print("\n✓ 10-agent experiments complete!")