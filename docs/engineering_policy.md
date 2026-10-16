# Project Rule: Safe Drive Persistence

## The Failure Mode
Colab's Drive mount is FUSE-based. Writes go to a local cache first, then sync to Google's servers in the background. If the session ends before sync completes, writes are silently lost.

## The Rule
For all artifacts that must persist:
1. Build in /content/ (local Colab storage).
2. Verify counts and hashes locally.
3. Copy to Drive.
4. Run `!sync` and wait 15-30 seconds.
5. READ BACK from Drive and verify counts/hashes.
6. Only then declare the artifact persisted.

## Incident Log
- 2026-10-15: Baseline training run, predictions.json, and corrected CSV were lost due to FUSE cache not syncing before session end. Code was preserved via GitHub (network operation, not Drive write).
