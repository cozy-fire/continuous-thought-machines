# Notes: Remote CTM Deployment

## Local project facts
- README recommends `conda create --name=ctm python=3.12` and `pip install -r requirements.txt`.
- Local code revision is `4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466` on `main`.
- Minimal forward will use the tested parity-style configuration with `parity_backbone`, small dimensions, and a binary `(B, L)` input, then validate prediction/certainty shapes and finite values.
- Local `data/mazes` contains 170,011 files totaling 295,310,429 bytes (about 0.275 GiB before compression).

## Remote facts
- Target: `featurize@workspace.featurize.cn:49030`
- Requested storage root: `/home/featurize/work`
- Hardware: x86_64, NVIDIA GeForce RTX 2080 Ti, 22,528 MiB VRAM; about 22,009 MiB free at inspection time.
- `/home/featurize/work` has about 28 GiB free; `/home/featurize/data` does not exist on this instance.
- Conda 24.3.0 is available at `/environment/miniconda3/bin/conda`; only `base` and `system` environments existed before deployment.
- Repository target `/home/featurize/work/continuous-thought-machines` and environment target `/home/featurize/work/envs/ctm` did not exist.
- Password login was used once to authorize the existing local Ed25519 public key; subsequent key-based SSH succeeded.

## Execution evidence
- Remote code path: `/home/featurize/work/continuous-thought-machines`
- Remote Conda prefix: `/home/featurize/work/envs/ctm`
- Installed Python 3.12.14, Torch 2.14.0+cu130, Torchvision 0.29.0+cu130; `pip check` reports no broken requirements.
- Uploaded archive: `/home/featurize/work/continuous-thought-machines/data/mazes.tar.gz`
- Archive SHA-256 matched locally and remotely: `ddd9119807d5a568191a95a9b829b78e9dfae73aa8e26d411fbf3fe71568dc5d`.
- Extracted dataset path: `/home/featurize/work/continuous-thought-machines/data/mazes`; remote file count is 170,011, matching local.
- Reproducible execution: `cd /home/featurize/work/continuous-thought-machines && bash run_job.sh`
- Forward result: passed on CUDA / NVIDIA GeForce RTX 2080 Ti; input `[2,16]`, predictions `[2,32,3]`, certainties `[2,2,3]`, synchronisation `[2,4]`, all outputs finite.
- Remote stdout log: `/home/featurize/work/continuous-thought-machines/logs/ctm_forward_smoke.stdout.log`
- Local synced stdout log: `logs/ctm_forward_smoke.stdout.log`; stderr log is empty.
