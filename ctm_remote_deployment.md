# CTM Remote Deployment Result

## Status

Deployment and the minimal CTM GPU forward test passed on 2026-09-08.

## Remote layout

- Repository: `/home/featurize/work/continuous-thought-machines`
- Repository revision: `4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466`
- Conda prefix: `/home/featurize/work/envs/ctm`
- Maze dataset: `/home/featurize/work/continuous-thought-machines/data/mazes`
- Uploaded maze archive: `/home/featurize/work/continuous-thought-machines/data/mazes.tar.gz`
- Runtime logs: `/home/featurize/work/continuous-thought-machines/logs`

## Environment evidence

- Python: `3.12.14`
- Torch: `2.14.0+cu130`
- Torchvision: `0.29.0+cu130`
- GPU: `NVIDIA GeForce RTX 2080 Ti`
- CUDA available: `true`
- `pip check`: no broken requirements

Activate the environment with:

```bash
source /environment/miniconda3/etc/profile.d/conda.sh
conda activate /home/featurize/work/envs/ctm
```

## Dataset evidence

- Local source file count: `170011`
- Remote extracted file count: `170011`
- Local and remote archive SHA-256:
  `ddd9119807d5a568191a95a9b829b78e9dfae73aa8e26d411fbf3fe71568dc5d`

## Forward evidence

Run:

```bash
cd /home/featurize/work/continuous-thought-machines
bash run_job.sh
```

Verified result:

```json
{"certainty_shape":[2,2,3],"cuda_available":true,"device":"cuda","finite_outputs":true,"gpu_name":"NVIDIA GeForce RTX 2080 Ti","input_shape":[2,16],"prediction_shape":[2,32,3],"status":"passed","synchronisation_shape":[2,4],"torch_version":"2.14.0+cu130"}
```

The authoritative stdout log is `logs/ctm_forward_smoke.stdout.log`; stderr is empty.

## Workflow state

- `run_site`: `remote`
- `workflow`: `python_packages_conda -> preflight -> ssh_direct`
- `install_intent`: `python_packages`
- `install_state`: `success`
- `verify_state`: `success`
- `conda_writeback`: `success`
- `execution_channel`: `ssh_direct`
- `execution_mode`: `null`
- `access_mode`: `ssh`
- `preflight_passed`: `true`
- `submission_state`: `executed`
- `execution_state`: `passed`
- `log_state`: `synced`
- `blocking_reason`: `none`
- `next_action`: `complete`
- `resume_target`: `onescience-runtime`
- `resume_phase`: `execute`
