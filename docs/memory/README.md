# Memory Checkpoints

`docs/memory/` is reserved for short-lived handoff checkpoints.

Retention policy:
- Keep only active handoff notes for in-flight work.
- Remove stale checkpoint files after the run is completed or replaced.
- Do not treat this folder as long-term documentation.

For stable guidance, use:
- `docs/skill/`
- `docs/guides/`
- `docs/benchmarks/`
