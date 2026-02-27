#!/usr/bin/env python3
"""
RAX-TCP Congestion Control Simulation

Simulates window evolution for two congestion control variants:
1. Fixed-parameter AIMD (baseline)
2. Regime-aware RAX-TCP for space links

This script models a single flow with ECN-based congestion signals
and generates metrics and plots for analysis.
"""

import sys
import numpy as np
import matplotlib
# Use Agg backend for non-interactive environments, but allow override
if 'DISPLAY' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def congestion_probability(w, C):
    """
    Compute the fraction of ECN-marked packets based on window size.
    
    Linear model: No marking below capacity; linear increase above capacity.
    This is a simple fluid model suitable for congestion control analysis.
    
    Args:
        w: Current congestion window (packets)
        C: Bottleneck link capacity (packets per RTT)
    
    Returns:
        p: Congestion probability in [0, 1]
    """
    return max(0, min(1, (w - C) / C))


def simulate_fixed_aimd(C, T, w0, a_fixed, b_fixed, w_min=1.0):
    """
    Simulate fixed-parameter AIMD congestion control (terrestrial).
    
    This variant represents a basic controller with fixed parameters,
    tuned for stable terrestrial networks with negligible non-congestive loss.
    No regime awareness. Uses simple linear ECN marking above capacity.
    
    Args:
        C: Bottleneck capacity (packets per RTT)
        T: Number of RTT steps to simulate
        w0: Initial window size
        a_fixed: Additive increase parameter
        b_fixed: Multiplicative decrease parameter
        w_min: Minimum window size (packets)
    
    Returns:
        w: Array of window sizes over time (length T+1)
    """
    w = np.zeros(T + 1)
    w[0] = w0
    
    for t in range(T):
        p = congestion_probability(w[t], C)
        w[t + 1] = w[t] + a_fixed - b_fixed * p * w[t]
        w[t + 1] = max(w_min, w[t + 1])
    
    return w


def simulate_space_with_fixed_params(C, T, w0, a_fixed, b_fixed, loss_prob, 
                                     w_min=1.0, random_seed=None):
    """
    Simulate space link using FIXED terrestrial parameters + random loss.
    
    This variant demonstrates why naive reuse of terrestrial parameters on space links
    is problematic. It applies the same aggressive (a_fixed, b_fixed) as terrestrial
    AIMD, but adds random non-congestive loss typical of space channels.
    This highlights why regime-aware tuning (RAX-TCP) is beneficial.
    
    Args:
        C: Bottleneck capacity (packets per RTT)
        T: Number of RTT steps to simulate
        w0: Initial window size
        a_fixed: Additive increase parameter (terrestrial tuning)
        b_fixed: Multiplicative decrease parameter (terrestrial tuning)
        loss_prob: Probability of random non-congestive loss per RTT
        w_min: Minimum window size (packets)
        random_seed: Seed for RNG reproducibility
    
    Returns:
        w: Array of window sizes over time (length T+1)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    w = np.zeros(T + 1)
    w[0] = w0
    
    for t in range(T):
        # Check for random loss event (space channel variability)
        if np.random.rand() < loss_prob:
            # Random loss: apply multiplicative decrease
            w[t + 1] = 0.9 * w[t]
        else:
            # Normal ECN/AIMD update with terrestrial parameters
            p = congestion_probability(w[t], C)
            w[t + 1] = w[t] + a_fixed - b_fixed * p * w[t]
        
        w[t + 1] = max(w_min, w[t + 1])
    
    return w


def simulate_dctcp(C, T, w0, a_dctcp, alpha_dctcp, w_min=1.0, random_seed=None):
    """
    Simulate DCTCP (Data Center TCP) congestion control.
    
    DCTCP uses a fractional multiplicative decrease: w = w * (1 - alpha*p)
    instead of standard AIMD's w = w + a - b*p*w.
    This allows DCTCP to achieve high utilization with low queueing.
    
    Can run on terrestrial or space links; here configured for space.
    
    Args:
        C: Bottleneck capacity (packets per RTT)
        T: Number of RTT steps to simulate
        w0: Initial window size
        a_dctcp: Additive increase parameter
        alpha_dctcp: DCTCP multiplicative decrease parameter (typically 0.5)
        w_min: Minimum window size (packets)
        random_seed: Seed for RNG reproducibility
    
    Returns:
        w: Array of window sizes over time (length T+1)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    w = np.zeros(T + 1)
    w[0] = w0
    
    for t in range(T):
        p = congestion_probability(w[t], C)
        # DCTCP formula: w = w * (1 - alpha * p) + a
        # This achieves high utilization with small queues
        w[t + 1] = w[t] * (1.0 - alpha_dctcp * p) + a_dctcp
        w[t + 1] = max(w_min, w[t + 1])
    
    return w


def simulate_rax_tcp_space(C, T, w0, a_space, b_space, gamma_space, loss_prob, 
                           w_min=1.0, random_seed=None):
    """
    Simulate regime-aware RAX-TCP for space links.
    
    This variant employs careful parameter tuning (smaller a, larger b) and
    explicit random loss handling (gamma_space) designed for space channels with
    higher latency and variable non-congestive loss rates.
    Demonstrates the benefit of regime-aware congestion control.
    
    Includes random non-congestive loss events to model the variable reliability
    of space communication channels (e.g., atmospheric absorption, fading).
    
    Args:
        C: Bottleneck capacity (packets per RTT)
        T: Number of RTT steps to simulate
        w0: Initial window size
        a_space: Additive increase parameter (space regime, conservative)
        b_space: Multiplicative decrease parameter (space regime, aggressive)
        gamma_space: Multiplicative decrease factor for random losses
        loss_prob: Probability of random loss in each RTT
        w_min: Minimum window size (packets)
        random_seed: Seed for RNG reproducibility
    
    Returns:
        w: Array of window sizes over time (length T+1)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    w = np.zeros(T + 1)
    w[0] = w0
    
    for t in range(T):
        # Check for random loss event
        if np.random.rand() < loss_prob:
            # Random loss: apply multiplicative decrease
            w[t + 1] = gamma_space * w[t]
        else:
            # Normal ECN/AIMD update
            p = congestion_probability(w[t], C)
            w[t + 1] = w[t] + a_space - b_space * p * w[t]
        
        w[t + 1] = max(w_min, w[t + 1])
    
    return w


def compute_queue_and_delay(w, C, base_rtt, k_delay):
    """
    Compute queue length and delay proxy over time.
    
    Args:
        w: Array of window sizes over time
        C: Bottleneck capacity (packets per RTT)
        base_rtt: Base RTT for this link (ms)
        k_delay: Delay per queued packet (ms)
    
    Returns:
        queue: Array of queue lengths (packets)
        delay: Array of delay values (ms)
    """
    queue = np.maximum(0, w - C)
    delay = base_rtt + k_delay * queue
    return queue, delay


def main():
    """Main simulation and plotting routine."""
    
    # ========== Simulation Parameters ==========
    
    # Link and flow parameters
    C = 100  # Bottleneck capacity (packets per RTT)
    T = 500  # Time horizon (RTT steps)
    w0 = 10  # Initial congestion window (packets)
    w_min = 1.0  # Minimum window size (packets)
    
    # Fixed-parameter AIMD baseline parameters (terrestrial)
    a_fixed = 1.0
    b_fixed = 0.3
    
    # RAX-TCP space regime parameters (regime-aware)
    a_space = 0.5
    b_space = 0.6
    gamma_space = 0.9  # Multiplicative decrease for random loss
    loss_prob = 0.02   # Probability of random loss per RTT
    
    # DCTCP parameters (for comparison on space link)
    a_dctcp = 0.5      # Additive increase
    alpha_dctcp = 0.5  # DCTCP's fractional multiplicative decrease parameter
    
    # Delay model parameters
    base_rtt_fixed = 50    # Baseline RTT for fixed variant (ms)
    base_rtt_space = 200   # Baseline RTT for space variant (ms)
    k_delay = 1.0          # Delay per queued packet (ms per packet)
    
    # ========== Run Simulations ==========
    
    print("=" * 80)
    print("RAX-TCP Congestion Control Simulation: Regime-Aware Tuning Benefits")
    print("=" * 80)
    print(f"\nSimulation Parameters:")
    print(f"  Bottleneck capacity (C):        {C} packets/RTT")
    print(f"  Simulation duration:             {T} RTT steps")
    print(f"  Initial window (w0):             {w0} packets")
    print(f"  Delay constant (k):              {k_delay} ms/packet")
    print(f"  ECN model:                       Linear marking p(w) = max(0,min(1,(w-C)/C))")
    print(f"\n{'─' * 80}")
    print(f"\nVariant 1: Fixed-parameter AIMD (Terrestrial Link)")
    print(f"  Parameters: a={a_fixed}, b={b_fixed}")
    print(f"  Base RTT: {base_rtt_fixed} ms (terrestrial, stable)")
    print(f"  Assumptions: Negligible non-congestive loss")
    
    print(f"\nVariant 2: Space Link with TERRESTRIAL Parameters (Mistuned)")
    print(f"  Parameters: a={a_fixed}, b={b_fixed} (same as terrestrial)")
    print(f"  Base RTT: {base_rtt_space} ms (space/satellite)")
    print(f"  Random loss: loss_prob = {loss_prob} (2% per RTT)")
    print(f"  Purpose: Show why naive parameter reuse fails on space channels")
    
    print(f"\nVariant 3: RAX-TCP Space Regime (Regime-Aware, Tuned)")
    print(f"  Parameters: a={a_space}, b={b_space} (conservative & aggressive)")
    print(f"  Base RTT: {base_rtt_space} ms (space/satellite)")
    print(f"  Random loss model: gamma={gamma_space}, loss_prob = {loss_prob}")
    print(f"  Purpose: Show benefit of regime-aware design")
    
    print(f"\nVariant 4: DCTCP (Data Center TCP) on Space Link")
    print(f"  Parameters: a={a_dctcp}, alpha={alpha_dctcp}")
    print(f"  Update rule: w = w*(1 - {alpha_dctcp}*p) + {a_dctcp}")
    print(f"  Base RTT: {base_rtt_space} ms (space/satellite, same as Variants 2 & 3)")
    print(f"  Purpose: Compare against industry standard algorithm\n")
    w_fixed = simulate_fixed_aimd(C, T, w0, a_fixed, b_fixed, w_min)
    queue_fixed, delay_fixed = compute_queue_and_delay(
        w_fixed, C, base_rtt_fixed, k_delay
    )
    
    # Simulate space link with terrestrial parameters (without regime awareness)
    # Use seed+1 for independent random stream
    w_space_mistuned = simulate_space_with_fixed_params(
        C, T, w0, a_fixed, b_fixed, loss_prob, w_min, random_seed=41
    )
    queue_space_mistuned, delay_space_mistuned = compute_queue_and_delay(
        w_space_mistuned, C, base_rtt_space, k_delay
    )
    
    # Simulate RAX-TCP space with regime-aware tuning
    w_space = simulate_rax_tcp_space(
        C, T, w0, a_space, b_space, gamma_space, loss_prob, w_min, random_seed=42
    )
    queue_space, delay_space = compute_queue_and_delay(
        w_space, C, base_rtt_space, k_delay
    )
    
    # Simulate DCTCP on space link
    w_dctcp = simulate_dctcp(C, T, w0, a_dctcp, alpha_dctcp, w_min, random_seed=43)
    queue_dctcp, delay_dctcp = compute_queue_and_delay(
        w_dctcp, C, base_rtt_space, k_delay
    )
    
    # ========== Compute Metrics ==========
    
    # Terrestrial baseline metrics
    avg_window_fixed = np.mean(w_fixed)
    avg_delay_fixed = np.mean(delay_fixed)
    max_window_fixed = np.max(w_fixed)
    min_window_fixed = np.min(w_fixed)
    avg_queue_fixed = np.mean(queue_fixed)
    
    # Space with terrestrial parameters (mistuned)
    avg_window_mistuned = np.mean(w_space_mistuned)
    avg_delay_mistuned = np.mean(delay_space_mistuned)
    max_window_mistuned = np.max(w_space_mistuned)
    min_window_mistuned = np.min(w_space_mistuned)
    avg_queue_mistuned = np.mean(queue_space_mistuned)
    
    # RAX-TCP space (tuned)
    avg_window_space = np.mean(w_space)
    avg_delay_space = np.mean(delay_space)
    max_window_space = np.max(w_space)
    min_window_space = np.min(w_space)
    avg_queue_space = np.mean(queue_space)
    
    # DCTCP on space
    avg_window_dctcp = np.mean(w_dctcp)
    avg_delay_dctcp = np.mean(delay_dctcp)
    max_window_dctcp = np.max(w_dctcp)
    min_window_dctcp = np.min(w_dctcp)
    avg_queue_dctcp = np.mean(queue_dctcp)
    
    # ========== Print Summary Statistics ==========
    
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    print(f"\n{'─' * 80}")
    print(f"Variant 1: Fixed-parameter AIMD (Terrestrial Link)")
    print(f"{'─' * 80}")
    print(f"  Average window size:    {avg_window_fixed:8.2f} packets")
    print(f"  Average queue depth:    {avg_queue_fixed:8.2f} packets")
    print(f"  Average delay:          {avg_delay_fixed:8.2f} ms")
    print(f"  Max window:             {max_window_fixed:8.2f} packets")
    print(f"  Min window:             {min_window_fixed:8.2f} packets")
    print(f"  Link utilization:       {(avg_window_fixed/C)*100:6.1f}%")
    
    print(f"\n{'─' * 80}")
    print(f"Variant 2: Space Link + Terrestrial Parameters (MISTUNED)")
    print(f"{'─' * 80}")
    print(f"  Average window size:    {avg_window_mistuned:8.2f} packets")
    print(f"  Average queue depth:    {avg_queue_mistuned:8.2f} packets")
    print(f"  Average delay:          {avg_delay_mistuned:8.2f} ms")
    print(f"  Max window:             {max_window_mistuned:8.2f} packets")
    print(f"  Min window:             {min_window_mistuned:8.2f} packets")
    print(f"  Link utilization:       {(avg_window_mistuned/C)*100:6.1f}%")
    
    print(f"\n{'─' * 80}")
    print(f"Variant 3: RAX-TCP Space Regime (REGIME-AWARE, TUNED)")
    print(f"{'─' * 80}")
    print(f"  Average window size:    {avg_window_space:8.2f} packets")
    print(f"  Average queue depth:    {avg_queue_space:8.2f} packets")
    print(f"  Average delay:          {avg_delay_space:8.2f} ms")
    print(f"  Max window:             {max_window_space:8.2f} packets")
    print(f"  Min window:             {min_window_space:8.2f} packets")
    print(f"  Link utilization:       {(avg_window_space/C)*100:6.1f}%")
    
    print(f"\n{'─' * 80}")
    print(f"Variant 4: DCTCP (Data Center TCP) on Space Link")
    print(f"{'─' * 80}")
    print(f"  Average window size:    {avg_window_dctcp:8.2f} packets")
    print(f"  Average queue depth:    {avg_queue_dctcp:8.2f} packets")
    print(f"  Average delay:          {avg_delay_dctcp:8.2f} ms")
    print(f"  Max window:             {max_window_dctcp:8.2f} packets")
    print(f"  Min window:             {min_window_dctcp:8.2f} packets")
    print(f"  Link utilization:       {(avg_window_dctcp/C)*100:6.1f}%")
    
    print(f"\n{'─' * 80}")
    print(f"Key Comparison: Space Link Mistuned vs Regime-Aware vs DCTCP")
    print(f"{'─' * 80}")
    if avg_queue_space > 0:
        queue_diff = ((avg_queue_mistuned - avg_queue_space) / avg_queue_space) * 100
        print(f"  Queue depth diff:       {queue_diff:+.1f}% (Mistuned vs RAX-TCP)")
    else:
        print(f"  Queue depth diff:        Both near zero (queues well-controlled)")
    
    if avg_window_space > 0:
        window_diff = ((avg_window_mistuned - avg_window_space) / avg_window_space) * 100
        print(f"  Window size diff:       {window_diff:+.1f}% (Mistuned vs RAX-TCP)")
    
    print(f"\n  DCTCP Performance on Space Link:")
    print(f"    Avg queue:            {avg_queue_dctcp:6.2f} packets (vs RAX-TCP: {avg_queue_space:.2f})")
    print(f"    Avg window:           {avg_window_dctcp:6.2f} packets (vs RAX-TCP: {avg_window_space:.2f})")
    print(f"    Link utilization:     {(avg_window_dctcp/C)*100:5.1f}% (vs RAX-TCP: {(avg_window_space/C)*100:.1f}%)")
    print(f"\n  → RAX-TCP: Lower gains (a=0.5) reduces aggressive growth")
    print(f"  → RAX-TCP: Higher decreases (b=0.6) dampen oscillations under loss")
    print(f"  → DCTCP: Achieves high window with fractional multiplicative decrease\n")
    
    # ========== Generate Plots ==========
    
    time_steps = np.arange(T + 1)
    
    # Plot 1: Window Evolution (all four variants)
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(time_steps, w_fixed, label='Terrestrial (Fixed AIMD)', 
             linewidth=2, alpha=0.8, color='#1f77b4')
    ax1.plot(time_steps, w_space_mistuned, label='Space + Terrestrial Params (Mistuned)', 
             linewidth=2, alpha=0.8, color='#ff7f0e', linestyle='--')
    ax1.plot(time_steps, w_space, label='Space + RAX-TCP Params (Tuned)', 
             linewidth=2, alpha=0.8, color='#2ca02c')
    ax1.plot(time_steps, w_dctcp, label='Space + DCTCP', 
             linewidth=2, alpha=0.8, color='#d62728', linestyle=':')
    ax1.axhline(y=C, color='red', linestyle=':', linewidth=1.5, 
                label=f'Link capacity (C={C})', alpha=0.6)
    ax1.set_xlabel('Time (RTT steps)', fontsize=12)
    ax1.set_ylabel('Congestion Window (packets)', fontsize=12)
    ax1.set_title('Congestion Window Evolution: Terrestrial vs Space, Mistuned vs Regime-Aware vs DCTCP', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, T)
    ax1.set_ylim(0, max(np.max(w_fixed), np.max(w_space_mistuned), np.max(w_space), np.max(w_dctcp)) * 1.1)
    fig1.tight_layout()
    
    # Plot 2: Delay Evolution (all three variants)
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(time_steps, delay_fixed, label='Terrestrial (Fixed AIMD)', 
             linewidth=2, alpha=0.8, color='#1f77b4')
    ax2.plot(time_steps, delay_space_mistuned, label='Space + Terrestrial Params (Mistuned)', 
             linewidth=2, alpha=0.8, color='#ff7f0e', linestyle='--')
    ax2.plot(time_steps, delay_space, label='Space + RAX-TCP Params (Tuned)', 
             linewidth=2, alpha=0.8, color='#2ca02c')
    ax2.plot(time_steps, delay_dctcp, label='Space + DCTCP', 
             linewidth=2, alpha=0.8, color='#d62728', linestyle=':')
    ax2.set_xlabel('Time (RTT steps)', fontsize=12)
    ax2.set_ylabel('End-to-End Delay (ms)', fontsize=12)
    ax2.set_title('Delay Proxy: Impact of Regime-Aware Tuning vs DCTCP on Space Links', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, T)
    fig2.tight_layout()
    
    # Plot 3: Queue Depth (all three variants)
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(time_steps, queue_fixed, label='Terrestrial (Fixed AIMD)', 
             linewidth=2, alpha=0.8, color='#1f77b4')
    ax3.plot(time_steps, queue_space_mistuned, label='Space + Terrestrial Params (Mistuned)', 
             linewidth=2, alpha=0.8, color='#ff7f0e', linestyle='--')
    ax3.plot(time_steps, queue_space, label='Space + RAX-TCP Params (Tuned)', 
             linewidth=2, alpha=0.8, color='#2ca02c')
    ax3.plot(time_steps, queue_dctcp, label='Space + DCTCP', 
             linewidth=2, alpha=0.8, color='#d62728', linestyle=':')
    ax3.set_xlabel('Time (RTT steps)', fontsize=12)
    ax3.set_ylabel('Queue Length (packets)', fontsize=12)
    ax3.set_title('Queue Buildup: Regime-Aware Tuning vs DCTCP Under Loss', 
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, T)
    ax3.set_ylim(bottom=0)
    fig3.tight_layout()
    
    # Save plots to files
    script_dir = "/Users/kannaiyanishwaryaa/SC4052 Cloud Computing/Assignment 1"
    fig1.savefig(f"{script_dir}/window_evolution.png", dpi=150, bbox_inches='tight')
    fig2.savefig(f"{script_dir}/delay_evolution.png", dpi=150, bbox_inches='tight')
    fig3.savefig(f"{script_dir}/queue_evolution.png", dpi=150, bbox_inches='tight')
    print("=" * 80)
    print("Plots saved to:")
    print(f"  - {script_dir}/window_evolution.png")
    print(f"  - {script_dir}/delay_evolution.png")
    print(f"  - {script_dir}/queue_evolution.png")
    print("=" * 80 + "\n")
    
    # Try to show interactively if available, otherwise just display file paths
    try:
        if matplotlib.get_backend() != 'Agg':
            plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
