# Reproduce the results

## Install Pytorch & other libraries, make sure to match your GPU driver version
%pip install "torch==2.5.1" tensorboard "setuptools<71.0.0"  --index-url https://download.pytorch.org/whl/cu121

## Install flash-attn
%pip install flash-attn 
 
## Install Hugging Face libraries
%pip install  --upgrade \
  "transformers==4.48.1" \
  "datasets==3.1.0" \
  "accelerate==1.3.0" \
  "hf-transfer==0.1.9" \
  "deepspeed==0.15.4" \
  "trl==0.14.0"
 
## Install vLLM ls
%pip install "vllm==0.7.0"
 
#### IMPORTANT: If you want to run the notebook and the interactive cells you also need to install the following libraries:
# But first read it the blog post and then decide as they might conflict with the libraries for distributed training. 
# %pip install "peft==0.14.0" "bitsandbytes==0.45.0"
 
```
uv run accelerate launch --num_processes 3 --config_file configs/deepspeed_zero3.yaml run_r1_grpo.py --config grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml
```




