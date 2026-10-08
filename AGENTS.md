# Project Execution Rules

## Virtual environments

- Always run Python, dependency installation, tests, training, inference, and evaluation inside an explicit virtual environment. Never use the system Python or the base Conda environment for project work.
- Use `/mnt/ssd1/szhao/DexTool/openpi_env` for DexCondVLA/OpenPI data processing, training, checkpoint loading, and policy inference.
- Use `/mnt/ssd1/szhao/DexTool/bench2dex_isaaclab_env` for Bench2Dex/Isaac Sim evaluation. This environment uses Isaac Sim 5.1, PyTorch 2.7 with CUDA 12.8, and Isaac Lab v2.3.2.
- Before starting a long-running job, verify the interpreter path and record it in the job log.
- Keep checkpoints, training/evaluation logs, downloaded assets, and evaluation results under `/mnt/ssd1/szhao/DexTool`.
