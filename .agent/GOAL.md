# Long-Term Research Goal

Determine which curvature-aware policy-optimization design is robust across
the seven MuJoCo continuous-control environments under matched architectures,
seeds, budgets, and evaluation conventions.

The program compares PPO, K-FAC, Curv256/Emp256, EnergyFree255+1, full
empirical-Fisher/full-GGN, and Kaczmarz/momentum variants in both large-batch
MLP and small-batch no-shared Transformer settings. Global choices must use
cross-environment normalized/ranked evidence rather than Hopper alone, while
version-specific gaps such as Swimmer-v3 remain explicit.

This file defines the long-term destination. Only the Planner may assign a new
high-level research objective through `TASK.md`.
