# API Reference

periodfind exposes three layers:

1. **Top-level factory functions** (`periodfind.ConditionalEntropy`, etc.) that
   dispatch to the appropriate backend based on the current device setting.
2. **Backend-specific classes** under `periodfind.cpu` and `periodfind.gpu`.
3. **Utility classes** (`Statistics`, `Periodogram`) and helper functions.

## Sections

- [Core API](core.md) — Factory functions, device management, `Statistics`, and `Periodogram`
- [CPU Backend](cpu.md) — Rust-backed implementations (period-finding and feature extraction)
- [GPU Backend](gpu.md) — CUDA-backed implementations (period-finding)
- [Utilities](utils.md) — Input validation and magnitude preparation helpers
