# Task Plan: Remote CTM Conda Deployment and Forward Smoke Test

## Goal
Deploy this CTM repository under the remote `/home/featurize/work` area with a Conda environment, then obtain real evidence that a minimal CTM forward pass succeeds.

## Phases
- [x] Phase 1: Inspect project requirements and define the smallest valid CTM forward path
- [x] Phase 2: Validate the SSH run site and inspect remote GPU, Conda, storage, and repository state
- [x] Phase 3: Deploy or update the repository, upload the local maze dataset, and create/reuse the Conda environment
- [x] Phase 4: Run preflight checks and the minimal CTM forward test
- [x] Phase 5: Verify the uploaded dataset, record evidence, review, and report

## Key Questions
1. Which CTM constructor and tensor shapes form the smallest dependency-light forward pass?
2. What remote Conda and CUDA stack is available?
3. Can the forward pass run successfully on the remote target, preferably with CUDA if available?

## Decisions Made
- Use `/home/featurize/work` as the remote storage root, as requested.
- Compress and upload the existing local `data/mazes` directory; do not fetch a replacement dataset.
- Treat the user's deployment request as explicit authorization to create the Conda environment and install repository dependencies.
- Require a real model forward result; import-only checks do not count as completion.

## Errors Encountered
- Initial key authentication failed because the local key was not authorized; installed the public key using the supplied password, then verified non-interactive key login.
- A remote `du -sh /home/featurize/work` scan was too slow on cloud storage; replaced it with `df` and exact target probes.
- One remote probe had a zsh parse error from nested PowerShell/remote-shell variable quoting; replaced the loop with explicit path checks.
- Two preflight one-liners were affected by cross-shell quoting/version formatting; replaced them with a PowerShell-safe quoted command and obtained a complete passing preflight.
- Direct script execution could not import the root `models` package; diagnosed the Python path behavior and reran successfully as `python -m scripts.ctm_forward_smoke`.
- A final full per-file byte walk exceeded the command window on the slow work disk; archive SHA-256 identity plus matching extracted file count were used as the integrity evidence.

## Status
**Complete** - Remote deployment, dataset upload/extraction, preflight, GPU CTM forward, log synchronization, and final verification all passed.
