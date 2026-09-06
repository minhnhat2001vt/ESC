# Public release checklist

This checklist tracks the conversion of the internal experiment workspace into a focused, reproducible research release.

## Completed in the first cleanup pass

- [x] Create an isolated release branch and draft pull request
- [x] Add a public-facing README
- [x] Add machine-readable citation metadata
- [x] Consolidate duplicate Conda environment exports
- [x] Remove machine-specific Conda prefixes
- [x] Strengthen ignore rules for local credentials, caches, checkpoints, and transient logs
- [x] Remove the unrelated root-level quantization script
- [x] Remove byte-identical backup scripts
- [x] Preserve all existing raw results, as requested

## Required before public release

- [x] Confirm the camera-ready ESC scripts and expose them as `run_esc_safety.py` and `run_esc_vqa.py`
- [ ] Replace hard-coded paths across maintained scripts (completed for the canonical ESC, baseline, VLSafe, and POPE paths)
- [x] Create release-facing entry points from the confirmed camera-ready scripts
- [ ] Remove or archive broken and obsolete utilities
- [ ] Add a model-free pipeline smoke test (portable path configuration already has a unit test)
- [x] Add automated Python and shell syntax checks in GitHub Actions
- [ ] Verify installation from a fresh Linux environment
- [ ] Document dataset acquisition and expected directory layouts
- [ ] Document exact commands for the paper's main tables
- [x] Add the MIT license
- [ ] Run a secret scan over the full Git history
- [ ] Add a tagged release after the pull request is approved

## Audit observations

- The repository contains 779 tracked files and approximately 979 MB of content.
- Approximately 976 MB is under `results/`; these files are retained intentionally.
- The original tree had no root README or license.
- Dozens of scripts contain machine-specific absolute paths.
- Several files are clearly named as backups, rebuttal variants, diagnostics, or tests; they need author validation before removal.
- The current environments were full machine exports rather than minimal dependency specifications.
- At least one legacy evaluator contains incomplete placeholder code and should not be presented as a supported entry point.

## Release principle

The public interface should expose a small number of documented, stable commands. Raw experiments and historical scripts may remain available for traceability, but they should be clearly separated from the supported reproduction workflow.
