# LIBERO Target Adapter

This directory contains the target-bound adapter for LIBERO benchmark observations
and actions.

## Components

- `target_adapter.py`: converts LIBERO raw target observations into canonical
  PhyAgentOS runtime observations and validates executable `[T, 7]` action
  chunks.

## Runtime IDs

`TARGETS.md` and runtime contracts should use:

```text
target_adapter://libero_adapter
```

The old `target_adapter://libero_mock_adapter` id is retained only as a
compatibility alias for older local tests.
