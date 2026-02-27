# RAX-TCP Congestion Control Simulation
## Enhanced Version with Regime-Aware Comparison

## Overview

This simulation demonstrates why regime-aware congestion control is beneficial for space communication links. It compares three variants:

1. **Terrestrial Fixed-Parameter AIMD:** A baseline controller tuned for stable, low-loss terrestrial links
2. **Space Link with Terrestrial Parameters (Mistuned):** The same aggressive parameters applied naively to a space link with non-congestive loss
3. **RAX-TCP Space Regime:** A regime-aware controller with parameters tuned for space environments

## Parameter Values Used

### Bottleneck Link Parameters
- **Capacity (C):** 100 packets/RTT
- **Simulation Duration (T):** 500 RTT steps  
- **Initial Window (w₀):** 10 packets
- **Minimum Window (w_min):** 1.0 packet
- **Delay per Queued Packet (k):** 1.0 ms/packet

### ECN Congestion Signal Model
$$p(w) = \max\left(0, \min\left(1, \frac{w - C}{C}\right)\right)$$

**Interpretation:**
- If $w \leq C$: No congestion signal ($p = 0$)
- If $w > C$: Linear marking increase, saturating at 1.0 as window grows well beyond capacity

### Variant 1: Fixed-Parameter AIMD (Terrestrial Baseline)
- **Additive Increase (a_fixed):** 1.0 packet/RTT (aggressive growth)
- **Multiplicative Decrease (b_fixed):** 0.3 (mild decrease on congestion)
- **Base RTT:** 50 ms (typical terrestrial link)
- **Random Loss:** None (assumption: negligible on terrestrial paths)

**Update Rule per RTT:**
$$w_{t+1} = w_t + 1.0 - 0.3 \cdot p(w_t) \cdot w_t$$

### Variant 2: Space Link + Terrestrial Parameters (Mistuned)
- **Parameters:** Same as terrestrial (a=1.0, b=0.3)
- **Base RTT:** 200 ms (representative of orbital/satellite ground link)
- **Random Non-Congestive Loss:** loss_prob = 0.02 (2% per RTT)

**When Random Loss Occurs:**
$$w_{t+1} = 0.9 \cdot w_t$$

### Variant 3: RAX-TCP Space Regime (Regime-Aware)
- **Additive Increase (a_space):** 0.5 packet/RTT (conservative growth)
- **Multiplicative Decrease (b_space):** 0.6 (aggressive decrease on signal)
- **Loss Response Factor (gamma_space):** 0.9
- **Random Loss Probability:** 0.02 (same 2% per RTT as Variant 2)
- **Base RTT:** 200 ms (same space link as Variant 2)

**Update Rules per RTT:**
- **On random loss** (2% chance): $w_{t+1} = 0.9 \cdot w_t$
- **On ECN congestion signal:** $w_{t+1} = w_t + 0.5 - 0.6 \cdot p(w_t) \cdot w_t$

### Variant 4: DCTCP (Data Center TCP) on Space Link
- **Additive Increase (a_dctcp):** 0.5 packet/RTT
- **DCTCP Parameter (alpha_dctcp):** 0.5
- **Base RTT:** 200 ms (same space link as Variants 2 & 3)
- **No explicit random loss modeling** (DCTCP focuses on congestion control, not channel variability)

**Update Rule per RTT:**
$$w_{t+1} = w_t \cdot (1.0 - 0.5 \cdot p(w_t)) + 0.5$$

This uses multiplicative decrease proportional to ECN marking rate, allowing high utilization with feedback-based control.

## Simulation Results

### Variant 1: Fixed-Parameter AIMD (Terrestrial)
- **Average Window Size:** 94.45 packets
- **Average Queue Depth:** 2.63 packets
- **Average Delay:** 52.63 ms
- **Maximum Window:** 103.23 packets
- **Minimum Window:** 10.00 packets
- **Link Utilization:** 94.5%

### Variant 2: Space Link + Terrestrial Parameters (Mistuned)
- **Average Window Size:** 93.18 packets
- **Average Queue Depth:** 2.14 packets
- **Average Delay:** 202.14 ms
- **Maximum Window:** 103.23 packets
- **Minimum Window:** 10.00 packets
- **Link Utilization:** 93.2%

### Variant 3: RAX-TCP Space Regime (Regime-Aware)
- **Average Window Size:** 78.27 packets
- **Average Queue Depth:** 0.20 packets  
- **Average Delay:** 200.20 ms
- **Maximum Window:** 100.83 packets
- **Minimum Window:** 10.00 packets
- **Link Utilization:** 78.3%

### Variant 4: DCTCP on Space Link
- **Average Window Size:** 84.37 packets
- **Average Queue Depth:** 0.63 packets
- **Average Delay:** 200.63 ms
- **Maximum Window:** 100.99 packets
- **Minimum Window:** 10.00 packets
- **Link Utilization:** 84.4%

## Key Comparison Table

| Metric | Terrestrial | Space Mistuned | RAX-TCP | DCTCP |
|--------|------------|-----------------|--------|-------|
| **Avg Window (packets)** | 94.45 | 93.18 | **78.27** | 84.37 |
| **Avg Queue (packets)** | 2.63 | 2.14 | **0.20** | 0.63 |
| **Link Utilization** | 94.5% | 93.2% | 78.3% | **84.4%** |
| **Avg Delay (ms)** | 52.63 | 202.14 | 200.20 | 200.63 |

**Key Insights:**
- **RAX-TCP:** Achieve lowest queue (0.20 packets) with conservative approach but lowest utilization (78.3%)
- **DCTCP:** Achieves middle ground—higher window (84.37) and utilization (84.4%) than RAX-TCP while keeping queue small (0.63)
- **Mistuned Space:** Shows why naive terrestrial parameters fail—same aggressiveness but ~10× queue buildup vs RAX-TCP

## Assumptions and Design Choices

1. **Regarding Terrestrial vs Space Loss:** Terrestrial links modeled as having no random non-congestive loss (accurate: fiber/copper have BER < 10⁻⁹). Space links assume 2% random loss (representative of atmospheric fading, hardware jitter).

2. **ECN Signaling:** Assumes perfect, instantaneous ECN feedback. Real TCP experiences feedback delay and retransmission delays (not modeled here for clarity).

3. **Fluid Model:** Continuous window evolution per RTT; ignores packet-level timing discreteness.

4. **Single Flow, Non-Shared:** No competing flows, no fairness considerations.

5. **Queue Model:** Simple linear model $q = \max(0, w - C)$; assumes link drains exactly C packets/RTT without burstiness.

6. **Window Bounds:** $w \in [w_{\min}, \infty)$ to prevent collapse below 1 packet.

7. **Parameter Choices:** The specific values are representative examples. Real tuning requires empirical data or theoretical optimization.

## Plots Generated

### 1. **window_evolution.png** — Congestion Window Over Time
- **Terrestrial:** Grows to ~94 packets, stable orbit
- **Space + Mistuned:** Similar ~93 packets (naive approach)
- **Space + Tuned:** Conservative ~78 packets (regime-aware)
- **Key Visual:** Shows how conservative design keeps peak windows lower

### 2. **delay_evolution.png** — End-to-End Delay Over Time
- **Terrestrial:** ~50 ms (base RTT + small queue)
- **Space variants:** ~200+ ms (dominated by base RTT)
- **Key Visual:** Base RTT difference dominates; queueing effect minimal but visible

### 3. **queue_evolution.png** — Queue Length at Bottleneck
- **Terrestrial:** Bounces 0–3 packets
- **Space + Mistuned:** Also 0–3 packets (surprisingly similar!)
- **Space + Tuned:** Stays near 0 (<1 packet most of time)
- **Key Visual:** **Clearest demonstration of regime-aware benefit**—RAX-TCP keeps queue virtually empty despite same random loss rate

## How to Run

```bash
python3 rax_tcp_simulation.py
```

The script will:
1. Print detailed parameters for all three variants
2. Compute 500 RTT steps for each
3. Display summary statistics with key comparisons
4. Save three PNG plots to the same directory
5. Display plots interactively if in a GUI environment

## Dependencies

- Python 3.6+
- `numpy` — numerical arrays
- `matplotlib` — plotting

Install with:
```bash
pip install numpy matplotlib
```

## Code-to-Theory Mapping

For examiners interested in the implementation:

- **`congestion_probability(w, C)`** → Implements $p(w) = \max(0, \min(1, (w-C)/C))$
- **`simulate_fixed_aimd(...)`** → Baseline: $w_{t+1} = w_t + a - b \cdot p(w_t) \cdot w_t$ (no regime awareness)
- **`simulate_space_with_fixed_params(...)`** → Demonstrates misconfiguration: correct loss model, wrong gains
- **`simulate_rax_tcp_space(...)`** → Regime-aware: conservative a, aggressive b, explicit loss handling

## Discussion Points for Your Report

**Why might real space protocols accept even lower window sizes?**
- Safety margins for unpredictable channels
- Fairness with other flows
- Avoiding reordering and out-of-order delivery

**What if random loss were 5% instead of 2%?**
- RAX-TCP advantage would be even more pronounced
- Mistuned variant would show more severe oscillations

**How would feedback delay (RTT feedback loop) change results?**
- Would slow convergence for all variants
- Mistuned variant would suffer more due to delayed awareness of loss events
- RAX-TCP's lower additive increase would help stabilize despite delay
