# OpenPI Policy Adapters

This directory contains policy-bound adapters for OpenPI-compatible policy
payloads.

## Components

- `base_openpi_adapter.py`: shared conversion and validation for OpenPI-style
  policy outputs.
- `dummy_openpi_adapter.py`: local dummy policy adapter used by protocol tests.
- `pi05_policy_adapter.py`: pi0.5 policy adapter. It converts canonical
  PhyAgentOS runtime observations into the OpenPI-compatible pi0.5 input format
  and converts OpenPI `actions` payloads into runtime policy action chunks.

## Runtime IDs

`SKILLRUNTIME.md` should use:

```text
policy_adapter://openpi_pi05_adapter
```
