# About

The goal is to ratatouille/control remote models with smaller models

# Setup

1. Install Pytorch & other libraries, make sure to match your GPU driver version
```
uv sync
```

2. Install flash-attn
uv pip install flash-attn --no-build-isolation


 
3. Start the training
```
uv run accelerate launch --num_processes 3 --config_file configs/deepspeed_zero3.yaml run_r1_grpo.py --config grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml
```




