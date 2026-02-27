"""
RAX-TCP Congestion Control Simulation
======================================

A toy congestion control model comparing:
1. Fixed-parameter ECN/AIMD (baseline)
2. Regime-aware RAX-TCP (space scenario with random loss)

Uses discrete RTT-step simulation with a simple fluid model.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


class RAXTCPSimulator:
    """Simulator for RAX-TCP and baseline congestion control variants."""
    
    def __init__(self, capacity: float = 100, time_steps: int = 500, 
                 initial_window: float = 10):
        """
        Initialize the simulator.
        
        Args:
            capacity: Bottleneck link capacity in packets per RTT (C)
            time_steps: Number of RTT steps to simulate (T)
            initial_window: Initial congestion window (w0)
        """
        self.C = capacity
        self.T = time_steps
        self.w0 = initial_window
        
        # Baseline (fixed-parameter ECN/AIMD) parameters
        self.a_fixed = 1.0
        self.b_fixed = 0.3
        
        # Space (RAX-TCP) parameters
        self.a_space = 0.7
        self.b_space = 0.45
        self.gamma_space = 0.95
        self.loss_prob = 0.02
        
        # Base RTT (now same for all variants to enable fair comparison)
        self.base_rtt = 200.0  # ms
        
        # Delay model parameters
        self.k_delay = 1.0  # ms per queued packet
        self.w_min = 1.0  # minimum window
        
        # Storage for results (three variants)
        self.window_baseline = []
        self.window_space = []
        self.window_space_bad = []
        
        self.queue_baseline = []
        self.queue_space = []
        self.queue_space_bad = []
        
        self.delay_baseline = []
        self.delay_space = []
        self.delay_space_bad = []
        
    def congestion_signal(self, w: float) -> float:
        """
        Calculate ECN marking fraction based on congestion window.
        
        p(w_t) = max(0, min(1, (w_t - C) / C))
        
        Args:
            w: Current congestion window
            
        Returns:
            Fraction of ECN-marked packets [0, 1]
        """
        return max(0.0, min(1.0, (w - self.C) / self.C))
    
    def calculate_queue_and_delay(self, w: float) -> Tuple[float, float]:
        """
        Calculate queue backlog and resulting delay.
        
        Args:
            w: Current congestion window
            
        Returns:
            Tuple of (queue_size, delay_ms)
        """
        queue = max(0.0, w - self.C)
        delay = self.base_rtt + self.k_delay * queue
        return queue, delay
    
    def step_baseline(self, w: float) -> float:
        """
        One RTT step for baseline fixed-parameter ECN/AIMD.
        
        w_{t+1} = w_t + a_fixed - b_fixed * p(w_t) * w_t
        
        Args:
            w: Current window
            
        Returns:
            Next window value
        """
        p = self.congestion_signal(w)
        w_next = w + self.a_fixed - self.b_fixed * p * w
        return max(self.w_min, w_next)
    
    def step_space(self, w: float) -> float:
        """
        One RTT step for regime-aware RAX-TCP (space scenario).
        
        With probability loss_prob: random loss event triggers multiplicative decrease.
        Otherwise: standard ECN/AIMD update with different parameters.
        
        Args:
            w: Current window
            
        Returns:
            Next window value
        """
        # Check for random non-congestive loss event
        if np.random.rand() < self.loss_prob:
            # Mild multiplicative decrease due to random loss
            w_next = self.gamma_space * w
        else:
            # Standard ECN/AIMD update with space parameters
            p = self.congestion_signal(w)
            w_next = w + self.a_space - self.b_space * p * w
        
        return max(self.w_min, w_next)
    
    def step_space_bad(self, w: float) -> float:
        """
        One RTT step for space link with NAIVE terrestrial parameters.
        
        Shows what happens if using fixed (terrestrial) parameters on a space link.
        Same gamma as RAX-TCP to isolate the effect of (a, b) parameter tuning only.
        
        Args:
            w: Current window
            
        Returns:
            Next window value
        """
        # Check for random non-congestive loss event
        if np.random.rand() < self.loss_prob:
            # Same gentle multiplicative decrease as RAX-TCP
            w_next = self.gamma_space * w
        else:
            # Standard ECN/AIMD update with terrestrial parameters
            p = self.congestion_signal(w)
            w_next = w + self.a_fixed - self.b_fixed * p * w
        
        return max(self.w_min, w_next)
    
    def run(self) -> None:
        """Run the complete simulation for all three variants."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Initialize windows
        w_baseline = self.w0
        w_space = self.w0
        w_space_bad = self.w0
        
        print("=" * 80)
        print("RAX-TCP Congestion Control Simulation (3 Variants)")
        print("=" * 80)
        print(f"Simulation parameters:")
        print(f"  Bottleneck capacity (C): {self.C} packets/RTT")
        print(f"  Time horizon (T): {self.T} RTT steps")
        print(f"  Initial window (w0): {self.w0} packets")
        print(f"  Base RTT (all variants): {self.base_rtt} ms")
        print()
        print("Variant 1: Baseline (Terrestrial Fixed ECN/AIMD):")
        print(f"  a_fixed = {self.a_fixed}, b_fixed = {self.b_fixed}")
        print()
        print("Variant 2: RAX-TCP (Space with Regime-Aware Parameters):")
        print(f"  a_space = {self.a_space}, b_space = {self.b_space}")
        print(f"  gamma_space = {self.gamma_space}, loss_prob = {self.loss_prob}")
        print()
        print("Variant 3: Space + Naive Terrestrial Parameters (Control Experiment):")
        print(f"  a = {self.a_fixed}, b = {self.b_fixed} (terrestrial, not space-tuned)")
        print(f"  gamma = {self.gamma_space} (same as RAX-TCP), loss_prob = {self.loss_prob}")
        print("-" * 80)
        
        # Simulation loop
        for t in range(self.T):
            # Store window values
            self.window_baseline.append(w_baseline)
            self.window_space.append(w_space)
            self.window_space_bad.append(w_space_bad)
            
            # Calculate queues and delays
            queue_b, delay_b = self.calculate_queue_and_delay(w_baseline)
            queue_s, delay_s = self.calculate_queue_and_delay(w_space)
            queue_sb, delay_sb = self.calculate_queue_and_delay(w_space_bad)
            
            self.queue_baseline.append(queue_b)
            self.queue_space.append(queue_s)
            self.queue_space_bad.append(queue_sb)
            
            self.delay_baseline.append(delay_b)
            self.delay_space.append(delay_s)
            self.delay_space_bad.append(delay_sb)
            
            # Update windows for next RTT
            w_baseline = self.step_baseline(w_baseline)
            w_space = self.step_space(w_space)
            w_space_bad = self.step_space_bad(w_space_bad)
        
        # Convert to numpy arrays for easy computation
        self.window_baseline = np.array(self.window_baseline)
        self.window_space = np.array(self.window_space)
        self.window_space_bad = np.array(self.window_space_bad)
        
        self.queue_baseline = np.array(self.queue_baseline)
        self.queue_space = np.array(self.queue_space)
        self.queue_space_bad = np.array(self.queue_space_bad)
        
        self.delay_baseline = np.array(self.delay_baseline)
        self.delay_space = np.array(self.delay_space)
        self.delay_space_bad = np.array(self.delay_space_bad)
        
        # Compute summary statistics
        self._compute_statistics()
    
    def _compute_statistics(self) -> None:
        """Compute and print summary statistics for all three variants."""
        # Baseline statistics
        avg_window_baseline = np.mean(self.window_baseline)
        peak_queue_baseline = np.max(self.queue_baseline)
        avg_queue_baseline = np.mean(self.queue_baseline) * self.k_delay
        avg_delay_baseline = np.mean(self.delay_baseline)
        
        # Space (RAX-TCP) statistics
        avg_window_space = np.mean(self.window_space)
        peak_queue_space = np.max(self.queue_space)
        avg_queue_space = np.mean(self.queue_space) * self.k_delay
        avg_delay_space = np.mean(self.delay_space)
        
        # Space + bad params statistics
        avg_window_space_bad = np.mean(self.window_space_bad)
        peak_queue_space_bad = np.max(self.queue_space_bad)
        avg_queue_space_bad = np.mean(self.queue_space_bad) * self.k_delay
        avg_delay_space_bad = np.mean(self.delay_space_bad)
        
        print("\nSummary Statistics:")
        print("-" * 80)
        
        print("\n[1] BASELINE (Terrestrial Fixed ECN/AIMD):")
        print(f"  Average window size: {avg_window_baseline:.2f} packets")
        print(f"  Peak queue length:   {peak_queue_baseline:.2f} packets")
        print(f"  Average queueing delay: {avg_queue_baseline:.2f} ms")
        print(f"  Average total delay:    {avg_delay_baseline:.2f} ms")
        
        print("\n[2] RAX-TCP (Space with Regime-Aware Parameters):")
        print(f"  Average window size: {avg_window_space:.2f} packets")
        print(f"  Peak queue length:   {peak_queue_space:.2f} packets")
        print(f"  Average queueing delay: {avg_queue_space:.2f} ms")
        print(f"  Average total delay:    {avg_delay_space:.2f} ms")
        
        print("\n[3] NAIVE (Space with Terrestrial Parameters):")
        print(f"  Average window size: {avg_window_space_bad:.2f} packets")
        print(f"  Peak queue length:   {peak_queue_space_bad:.2f} packets")
        print(f"  Average queueing delay: {avg_queue_space_bad:.2f} ms")
        print(f"  Average total delay:    {avg_delay_space_bad:.2f} ms")
        
        print("\n" + "-" * 80)
        print("Key Insights (RAX-TCP vs Baseline):")
        print(f"  Window size delta:              {avg_window_space - avg_window_baseline:+.2f} packets")
        print(f"  Queueing delay delta:           {avg_queue_space - avg_queue_baseline:+.2f} ms")
        print(f"  Peak queue length delta:        {peak_queue_space - peak_queue_baseline:+.2f} packets")
        
        print("\nKey Insights (Naive vs RAX-TCP on Space Link):")
        print(f"  Window size delta:              {avg_window_space_bad - avg_window_space:+.2f} packets")
        print(f"  Queueing delay delta:           {avg_queue_space_bad - avg_queue_space:+.2f} ms")
        print(f"  Peak queue length delta:        {peak_queue_space_bad - peak_queue_space:+.2f} packets")
        print("=" * 80)
    
    def plot_results(self) -> None:
        """Generate and display plots for all three variants."""
        time_steps = np.arange(len(self.window_baseline))
        
        # Figure 1: Window evolution over time (all three variants)
        fig1, ax1 = plt.subplots(figsize=(13, 6))
        ax1.plot(time_steps, self.window_baseline, label='Baseline (Terrestrial)',
                 linewidth=1.5, alpha=0.85, color='C0')
        ax1.plot(time_steps, self.window_space, label='RAX-TCP (Space-Aware)',
                 linewidth=1.5, alpha=0.85, color='C1')
        ax1.plot(time_steps, self.window_space_bad, label='Naive (Space + Terrestrial Params)',
                 linewidth=1.5, alpha=0.85, color='C2', linestyle='--')
        ax1.axhline(y=self.C, color='red', linestyle='--', alpha=0.4,
                    label=f'Capacity (C={self.C})')
        ax1.set_xlabel('Time (RTT steps)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Congestion Window (packets)', fontsize=12, fontweight='bold')
        ax1.set_title('Congestion Window Evolution — Regime-Aware vs Naive Tuning', 
                      fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig('rax_tcp_window_evolution.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved: rax_tcp_window_evolution.png")
        
        # Figure 2: Total delay over time (all three variants)
        fig2, ax2 = plt.subplots(figsize=(13, 6))
        ax2.plot(time_steps, self.delay_baseline, label='Baseline (Terrestrial)',
                 linewidth=1.5, alpha=0.85, color='C0')
        ax2.plot(time_steps, self.delay_space, label='RAX-TCP (Space-Aware)',
                 linewidth=1.5, alpha=0.85, color='C1')
        ax2.plot(time_steps, self.delay_space_bad, label='Naive (Space + Terrestrial Params)',
                 linewidth=1.5, alpha=0.85, color='C2', linestyle='--')
        ax2.axhline(y=self.base_rtt, color='gray', linestyle=':', alpha=0.5,
                    label=f'Base RTT ({self.base_rtt} ms)')
        ax2.set_xlabel('Time (RTT steps)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Delay (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Total Delay Over Time (Base RTT + Queueing)', 
                      fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig('rax_tcp_total_delay.png', dpi=150, bbox_inches='tight')
        print("Plot saved: rax_tcp_total_delay.png")
        
        # Figure 3: QUEUEING DELAY ONLY (most important for fair comparison)
        fig3, ax3 = plt.subplots(figsize=(13, 6))
        ax3.plot(time_steps, self.queue_baseline * self.k_delay, 
                 label='Baseline (Terrestrial)', linewidth=1.5, alpha=0.85, color='C0')
        ax3.plot(time_steps, self.queue_space * self.k_delay, 
                 label='RAX-TCP (Space-Aware)', linewidth=1.5, alpha=0.85, color='C1')
        ax3.plot(time_steps, self.queue_space_bad * self.k_delay, 
                 label='Naive (Space + Terrestrial Params)', linewidth=1.5, alpha=0.85, 
                 color='C2', linestyle='--')
        ax3.set_xlabel('Time (RTT steps)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Queueing Delay (ms)', fontsize=12, fontweight='bold')
        ax3.set_title('Queueing Delay Only — Fair Comparison (No Base RTT Confounding)', 
                      fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=11)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig('rax_tcp_queueing_delay.png', dpi=150, bbox_inches='tight')
        print("Plot saved: rax_tcp_queueing_delay.png")
        
        plt.show()


def main():
    """Main entry point for the simulation."""
    # Create and run simulator
    simulator = RAXTCPSimulator(capacity=100, time_steps=500, initial_window=10)
    simulator.run()
    simulator.plot_results()


if __name__ == '__main__':
    main()
