# Core API

The top-level `periodfind` module provides device-agnostic factory functions
and shared data classes.

## Device Management

::: periodfind.set_device

::: periodfind.get_device

## Period-Finding Factory Functions

These create algorithm instances on the current device (or an explicitly
specified device via the `device` keyword).

::: periodfind.ConditionalEntropy

::: periodfind.AOV

::: periodfind.LombScargle

::: periodfind.FPW

## Feature Extraction Factory Functions

::: periodfind.FourierDecomposition

::: periodfind.DmDt

::: periodfind.BasicStats

## Utility Functions

::: periodfind.remove_high_cadence

## Data Classes

::: periodfind.Statistics

::: periodfind.Periodogram
