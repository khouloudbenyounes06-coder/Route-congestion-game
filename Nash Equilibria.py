import numpy as np
from math import comb  # For binomial coefficients (number of ways)

# ============================================================================
# COST FUNCTIONS (shared)
# ============================================================================

def cost_A(n_A):
    """Cost on Route A if n_A agents choose it"""
    return 10 + 15 * n_A if n_A > 0 else 0

def cost_B(n_B):
    """Cost on Route B if n_B agents choose it"""
    return 30 + 5 * n_B if n_B > 0 else 0

# ============================================================================
# GENERAL NASH EQUILIBRIA FINDER (any N)
# ============================================================================

def find_pure_nash_equilibria(N, verbose=True):
    """
    Find all pure-strategy Nash equilibria for N agents.
    Returns list of dicts with equilibrium splits and costs.
    """
    if verbose:
        print("=" * 70)
        print(f"PURE NASH EQUILIBRIA ANALYSIS FOR N={N} AGENTS")
        print("=" * 70)
        print("Route A: base=10, congestion=15 per agent")
        print("Route B: base=30, congestion=5 per agent")
        print("-" * 70)

    equilibria = []

    for n_A in range(N + 1):
        n_B = N - n_A

        c_A = cost_A(n_A)
        c_B = cost_B(n_B)

        # Cost if one agent on A switches to B
        c_switch_A_to_B = cost_B(n_B + 1) if n_A > 0 else np.inf

        # Cost if one agent on B switches to A
        c_switch_B_to_A = cost_A(n_A + 1) if n_B > 0 else np.inf

        # No deviation from A?
        no_dev_from_A = (n_A == 0) or (c_switch_A_to_B >= c_A)

        # No deviation from B?
        no_dev_from_B = (n_B == 0) or (c_switch_B_to_A >= c_B)

        is_nash = no_dev_from_A and no_dev_from_B

        total_cost = n_A * c_A + n_B * c_B

        if verbose:
            status = "✓ NASH" if is_nash else ""
            print(f"n_A = {n_A:2d}  |  n_B = {n_B:2d}  |  "
                  f"c_A = {c_A:5.1f}  |  c_B = {c_B:5.1f}  |  "
                  f"total = {total_cost:6.1f}  |  {status}")

        if is_nash:
            num_labeled = comb(N, n_A) if n_A > 0 and n_B > 0 else 1
            equilibria.append({
                'n_A': n_A,
                'n_B': n_B,
                'cost_A': c_A,
                'cost_B': c_B,
                'total_cost': total_cost,
                'avg_cost': total_cost / N if N > 0 else 0,
                'num_labeled_profiles': num_labeled
            })

    if verbose:
        print("=" * 70)
        print(f"Found {len(equilibria)} stable split(s)")
        print("Detailed Nash Equilibria:")
        print("-" * 70)
        for i, eq in enumerate(equilibria, 1):
            print(f"Nash Split {i}: {eq['n_A']} on A, {eq['n_B']} on B")
            print(f"  Costs: A = {eq['cost_A']:.1f}, B = {eq['cost_B']:.1f}")
            print(f"  Total cost: {eq['total_cost']:.1f}")
            print(f"  Avg cost per agent: {eq['avg_cost']:.2f}")
            print(f"  Number of labeled profiles: {eq['num_labeled_profiles']}")
            print("-" * 70)

    return equilibria

# ============================================================================
# SOCIAL OPTIMUM FINDER (any N)
# ============================================================================

def find_social_optimum(N, verbose=True):
    """
    Find split that minimizes total social cost for N agents.
    Returns dict with optimal split info.
    """
    if verbose:
        print("\n" + "=" * 70)
        print(f"SOCIAL OPTIMUM FOR N={N}")
        print("=" * 70)

    min_cost = float('inf')
    optimal = None

    for n_A in range(N + 1):
        n_B = N - n_A
        total_cost = n_A * cost_A(n_A) + n_B * cost_B(n_B)

        if verbose:
            print(f"n_A={n_A:2d}, n_B={n_B:2d} | total cost = {total_cost:6.1f}")

        if total_cost < min_cost:
            min_cost = total_cost
            optimal = {
                'n_A': n_A,
                'n_B': n_B,
                'total_cost': total_cost,
                'avg_cost': total_cost / N if N > 0 else 0
            }

    if verbose:
        print("=" * 70)
        print(f"✓ Social optimum: {optimal['n_A']} on A, {optimal['n_B']} on B")
        print(f"  Total cost: {optimal['total_cost']:.1f}")
        print(f"  Avg cost per agent: {optimal['avg_cost']:.2f}")

    return optimal

# ============================================================================
# MAIN EXECUTION - Run for any N
# ============================================================================


# ============================================================================
# COMPREHENSIVE COMPARISON FUNCTION
# ============================================================================

def compare_nash_vs_optimum(N):
    """
    Comprehensive comparison of Nash equilibria vs social optimum.
    Provides game-theoretic insights.
    """
    print("\n" + "="*80)
    print(f"COMPREHENSIVE GAME-THEORETIC ANALYSIS FOR N={N} AGENTS")
    print("="*80)
    
    # Get Nash and Optimum
    nash_eq = find_pure_nash_equilibria(N, verbose=False)
    optimum = find_social_optimum(N, verbose=False)
    
    # Summary Table Header
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    EQUILIBRIUM SUMMARY                          │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    # Nash Equilibria
    print(f"│ Nash Equilibria Found: {len(nash_eq):<44} │")
    for i, eq in enumerate(nash_eq, 1):
        print("├─────────────────────────────────────────────────────────────────┤")
        print(f"│ Nash {i}: Split ({eq['n_A']} on A, {eq['n_B']} on B)                              │")
        print(f"│   • Cost on Route A:        {eq['cost_A']:>6.1f}                         │")
        print(f"│   • Cost on Route B:        {eq['cost_B']:>6.1f}                         │")
        print(f"│   • Cost difference (B-A):  {eq['cost_B'] - eq['cost_A']:>6.1f}                         │")
        print(f"│   • Average cost per agent: {eq['avg_cost']:>6.2f}                         │")
        print(f"│   • Total social cost:      {eq['total_cost']:>6.1f}                         │")
        print(f"│   • Num. labeled profiles:  {eq['num_labeled_profiles']:>6}                           │")
    
    # Social Optimum
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Social Optimum: ({optimum['n_A']} on A, {optimum['n_B']} on B)                        │")
    print(f"│   • Total social cost:      {optimum['total_cost']:>6.1f}                         │")
    print(f"│   • Average cost per agent: {optimum['avg_cost']:>6.2f}                         │")
    
    # Price of Anarchy
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│ PRICE OF ANARCHY ANALYSIS                                      │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    for i, eq in enumerate(nash_eq, 1):
        poa = eq['total_cost'] / optimum['total_cost']
        inefficiency_pct = (poa - 1) * 100
        gap = eq['avg_cost'] - optimum['avg_cost']
        
        print(f"│ Nash {i} vs Optimum:                                            │")
        print(f"│   • Price of Anarchy (PoA): {poa:>6.4f}                           │")
        print(f"│   • Inefficiency:           {inefficiency_pct:>6.2f}%                          │")
        print(f"│   • Cost gap per agent:     {gap:>+6.2f}                           │")
        
        # Interpretation
        if poa == 1.0:
            print("│   • Interpretation: ✓ Nash IS social optimum (efficient!)      │")
        elif poa < 1.05:
            print("│   • Interpretation: Nash very close to optimum (<5% loss)      │")
        elif poa < 1.33:
            print("│   • Interpretation: Moderate inefficiency (within PoA=4/3)     │")
        else:
            print("│   • Interpretation: Significant inefficiency (PoA > 4/3)       │")
    
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # Game-Theoretic Insights
    print("\n" + "="*80)
    print("GAME-THEORETIC INSIGHTS")
    print("="*80)
    
    eq = nash_eq[0]  # Focus on first (usually only) Nash
    
    print(f"\n1. EQUILIBRIUM STRUCTURE:")
    print(f"   • Route A is SHORT but CONGESTIBLE (base=10, penalty=15/agent)")
    print(f"   • Route B is LONG but STABLE (base=30, penalty=5/agent)")
    print(f"   • At Nash, {eq['n_A']} agents choose A (paying {eq['cost_A']:.0f})")
    print(f"   • At Nash, {eq['n_B']} agents choose B (paying {eq['cost_B']:.0f})")
    print(f"   • Cost difference: Route B costs {eq['cost_B'] - eq['cost_A']:.0f} more than A")
    
    print(f"\n2. WHY IS THIS A NASH EQUILIBRIUM?")
    # Check deviation incentives
    if eq['n_A'] > 0:
        cost_if_A_switches = cost_B(eq['n_B'] + 1)
        print(f"   • If an agent on A (paying {eq['cost_A']:.0f}) switches to B:")
        print(f"     → Would pay {cost_if_A_switches:.0f} (B with {eq['n_B']+1} agents)")
        print(f"     → No incentive to switch ({cost_if_A_switches:.0f} ≥ {eq['cost_A']:.0f})")
    
    if eq['n_B'] > 0:
        cost_if_B_switches = cost_A(eq['n_A'] + 1)
        print(f"   • If an agent on B (paying {eq['cost_B']:.0f}) switches to A:")
        print(f"     → Would pay {cost_if_B_switches:.0f} (A with {eq['n_A']+1} agents)")
        print(f"     → No incentive to switch ({cost_if_B_switches:.0f} ≥ {eq['cost_B']:.0f})")
    
    print(f"\n3. SOCIAL EFFICIENCY:")
    if eq['n_A'] == optimum['n_A'] and eq['n_B'] == optimum['n_B']:
        print(f"   ✓ Nash equilibrium IS the social optimum!")
        print(f"   • Selfish behavior achieves socially optimal outcome")
        print(f"   • No centralized coordination needed")
        print(f"   • This is RARE in congestion games (usually PoA > 1)")
    else:
        gap_pct = ((eq['total_cost'] / optimum['total_cost']) - 1) * 100
        print(f"   ✗ Nash differs from social optimum")
        print(f"   • Nash uses ({eq['n_A']}A, {eq['n_B']}B), Optimum uses ({optimum['n_A']}A, {optimum['n_B']}B)")
        print(f"   • Inefficiency: {gap_pct:.1f}% welfare loss")
    
    print(f"\n4. LABELED PROFILES (SYMMETRY):")
    print(f"   • There are C({N},{eq['n_A']}) = {eq['num_labeled_profiles']} ways to assign specific agents")
    print(f"   • Example: Agents {list(range(eq['n_A']))} on A, rest on B")
    print(f"   • All {eq['num_labeled_profiles']} assignments are equivalent Nash equilibria")
    print(f"   • This creates an equilibrium selection problem for learning agents")
    
    print("\n" + "="*80)
    
    return nash_eq, optimum


# ============================================================================
# SCALING COMPARISON
# ============================================================================

def compare_scaling():
    """Compare how equilibria change from N=2 to N=10"""
    print("\n" + "="*80)
    print("SCALING ANALYSIS: N=2 vs N=10")
    print("="*80)
    
    nash_2 = find_pure_nash_equilibria(2, verbose=False)[0]
    nash_10 = find_pure_nash_equilibria(10, verbose=False)[0]
    opt_2 = find_social_optimum(2, verbose=False)
    opt_10 = find_social_optimum(10, verbose=False)
    
    print("\n┌────────────────────────────────────────────────────────────┐")
    print("│                    COMPARISON TABLE                        │")
    print("├────────────────┬──────────────────┬──────────────────────┤")
    print("│   Metric       │      N=2         │        N=10          │")
    print("├────────────────┼──────────────────┼──────────────────────┤")
    print(f"│ Nash Split     │  ({nash_2['n_A']}A, {nash_2['n_B']}B)          │  ({nash_10['n_A']}A, {nash_10['n_B']}B)              │")
    print(f"│ Cost on A      │  {nash_2['cost_A']:>6.1f}          │  {nash_10['cost_A']:>6.1f}              │")
    print(f"│ Cost on B      │  {nash_2['cost_B']:>6.1f}          │  {nash_10['cost_B']:>6.1f}              │")
    print(f"│ Avg Cost/Agent │  {nash_2['avg_cost']:>6.2f}          │  {nash_10['avg_cost']:>6.2f}              │")
    print(f"│ Social Optimum │  ({opt_2['n_A']}A, {opt_2['n_B']}B)          │  ({opt_10['n_A']}A, {opt_10['n_B']}B)              │")
    print(f"│ PoA            │  {nash_2['total_cost']/opt_2['total_cost']:>6.4f}          │  {nash_10['total_cost']/opt_10['total_cost']:>6.4f}              │")
    print("└────────────────┴──────────────────┴──────────────────────┘")
    
    print("\nKEY OBSERVATIONS:")
    print(f"  • As N increases from 2→10:")
    print(f"    - Proportion on A: {nash_2['n_A']/2:.0%} → {nash_10['n_A']/10:.0%}")
    print(f"    - Average cost: {nash_2['avg_cost']:.0f} → {nash_10['avg_cost']:.0f} ({(nash_10['avg_cost']/nash_2['avg_cost']-1)*100:+.0f}%)")
    print(f"    - Cost on A: {nash_2['cost_A']:.0f} → {nash_10['cost_A']:.0f} ({(nash_10['cost_A']/nash_2['cost_A']-1)*100:+.0f}%)")
    print(f"    - Cost on B: {nash_2['cost_B']:.0f} → {nash_10['cost_B']:.0f} ({(nash_10['cost_B']/nash_2['cost_B']-1)*100:+.0f}%)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("CONGESTION GAME BASELINE ANALYSIS\n")
    
    # Comprehensive analysis for N=10
    nash_10, opt_10 = compare_nash_vs_optimum(N=10)
    
    # Scaling comparison
    compare_scaling()
    
    print("\n✓ Baseline analysis complete. Ready for Q-learning experiments.")