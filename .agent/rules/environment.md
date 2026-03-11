---
trigger: always_on
---

# Python Environment Rules

When executing python scripts, installing pip packages, or running commands, you MUST use the specific Conda environment located at:
`D:\miniconda3\envs\avatar\avatar\python.exe`

You can use `conda activate avatar`

Do NOT use the system python or the base environment.
Do NOT try to create a new venv.
ALWAYS activate this environment first if running a shell command requires it.