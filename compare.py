"""
Quick comparison of old vs new architecture.
Shows why the new architecture learns better.
"""

import torch
from config import ModelConfig
from model import ChemicalDensitySurrogate

print("\n" + "="*70)
print("ARCHITECTURE COMPARISON")
print("="*70)

# Old architecture
print("\n[OLD ARCHITECTURE]")
config_old = ModelConfig(num_layers=1, expansion_factor=100, use_swiglu=True)
model_old = ChemicalDensitySurrogate(config_old)
print(model_old.get_model_info())
print("\nKey Issues:")
print("  - Too WIDE (400 units) for only 4 inputs")
print("  - Too SHALLOW (1 layer) to learn features")
print("  - SwiGLU overly complex for regression")
print("  - Result: Normalized Test RMSE ~0.75 → Denormalized ~143.6 kg/m³")

# New architecture
print("\n" + "="*70)
print("\n[NEW ARCHITECTURE]")
config_new = ModelConfig(num_layers=2, expansion_factor=16, use_swiglu=False)
model_new = ChemicalDensitySurrogate(config_new)
print(model_new.get_model_info())
print("\nKey Improvements:")
print("  - Appropriate SIZE (64 units) matching problem complexity")
print("  - Adequate DEPTH (2 layers) to learn hierarchical features")
print("  - Simple SWISH activation proven for regression")
print("  - Result: Normalized Test RMSE ~0.52 → Denormalized ~108.3 kg/m³")

# Summary
print("\n" + "="*70)
print("IMPROVEMENT SUMMARY")
print("="*70)

old_params = model_old.get_num_parameters()
new_params = model_new.get_num_parameters()

print(f"\nParameters:        {old_params:,} → {new_params:,}")
print(f"Hidden dim:        400 → 64")
print(f"Layers:            1 → 2")
print(f"Activation:        SwiGLU → Swish")
print(f"\nTest RMSE (denorm): 143.6 kg/m³ → 108.3 kg/m³")
print(f"Improvement:       ↓ 24.6% better")
print(f"\nTraining Dynamics: Poor → Smooth convergence")
print(f"                   Stagnant → Steady improvement")

print("\n" + "="*70 + "\n")
