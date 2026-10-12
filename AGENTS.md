# Project Execution Rules

## Virtual environments

- Always run Python, dependency installation, tests, training, inference, and evaluation inside an explicit virtual environment. Never use the system Python or the base Conda environment for project work.
- Use `/mnt/ssd1/szhao/DexTool/openpi_env` for DexCondVLA/OpenPI data processing, training, checkpoint loading, and policy inference.
- Use `/mnt/ssd1/szhao/DexTool/bench2dex_isaaclab_env` for Bench2Dex/Isaac Sim evaluation. This environment uses Isaac Sim 5.1, PyTorch 2.7 with CUDA 12.8, and Isaac Lab v2.3.2.
- Before starting a long-running job, verify the interpreter path and record it in the job log.
- Keep checkpoints, training/evaluation logs, downloaded assets, and evaluation results under `/mnt/ssd1/szhao/DexTool`.

## Training checkpoint registry

- Whenever a new training run starts, add an entry to `TRAINING_CHECKPOINT_REGISTRY.md` with the unique checkpoint/experiment name, purpose, complete setting, and every special change relative to the closest prior run.
- Record the launch before or at the same time as starting training; later update the same entry with resume history, final status, and the checkpoint selected for evaluation.
- Never reuse or overwrite a checkpoint name or remove an earlier registry entry.
- Do not bypass the automatic code snapshot in the training entrypoints. Verify that `<checkpoint_dir>/code_snapshots/LATEST` exists immediately after every new training or resume launch.
