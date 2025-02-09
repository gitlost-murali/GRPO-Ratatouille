import logging
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random
import re 
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
# Import the remote inference requester
from utils.inference_requester import InferenceRequester
from utils.generic_utils import (
    extract_prompt_from_completion,
    extract_equation_from_completion,
    extract_lhs_from_equation,
    validate_equation_numbers,
    is_valid_equation_format,
    evaluate_equation
)

from dotenv import load_dotenv
load_dotenv()

assert "OPENAI_API_TOKEN" in os.environ, "OPENAI_API_TOKEN is not set"

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Helper functions
########################

requester = InferenceRequester()

def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><prompt>...</prompt>
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected prompts
        
        Returns:
            list[float]: Reward scores
    Validates the output format which must consist of matching <think> and <prompt> tags.
    Reward is 1.0 for correctly formatted outputs, and 0.0 otherwise.
    """
    rewards = []

    for completion, gt in zip(completions, target):

      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        if random.random() < 0.1:  # 1% chance to write samples into a file
          os.makedirs("completion_samples", exist_ok=True)
          log_file = os.path.join("completion_samples", "completion_samples.txt")
          with open(log_file, "a") as f:
            f.write(f"\n\n==============\n")
            f.write(completion)
        
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<prompt>([\s\S]*?)<\/prompt>$"

        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
    return rewards

def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions by extracting the <prompt> portion (the refined prompt),
    sending it to a remote inference model, and then comparing the remote model's reply.
    
    The remote model is accessed via InferenceRequester. For now, the reward is 1.0
    if the remote response is non-empty; otherwise, 0.0.
    
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected prompts (or reference prompts).
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # Add synthetic <think> tag
            completion = "<think>" + completion
            
            # Extract and validate prompt
            generated_prompt = extract_prompt_from_completion(completion)
            if not generated_prompt:
                rewards.append(0.0)
                continue
                
            # Get equation from remote inference
            raw_response = requester.generate_response(generated_prompt, stream=False)
            raw_equation = extract_equation_from_completion(str(raw_response).strip())
            equation = extract_lhs_from_equation(raw_equation)

            if equation is None:
                equation = ""

            # Validate equation format and numbers
            if not all([
                validate_equation_numbers(equation, numbers),
                is_valid_equation_format(equation),
                evaluate_equation(equation, float(gt))
            ]):
                os.makedirs("failed_completion_samples", exist_ok=True)
                failed_log_file = os.path.join("failed_completion_samples", "failed_completion_samples.txt")
                with open(failed_log_file, "a") as f:
                    f.write("\n\n==============\n")
                    f.write("Completion:\n")
                    f.write(str(completion))
                    f.write("\n\nRemote Equation:\n")
                    f.write(str(equation))
                    f.write("\n\nExpected Equation:\n")
                    f.write(str(gt))
                    f.write("\n\nRaw response:\n")
                    f.write(str(raw_response))
                    f.write("\n\n************\n")

                rewards.append(0.0)
                continue

                
            # All validations passed
            rewards.append(1.0)
            
            # Example: Write the remote equation along with the completion to a file for inspection.
            if random.random() < 0.10:  # 10% chance to write successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write("\n\n==============\n")
                    f.write("Completion:\n")
                    f.write(str(completion))
                    f.write("\n\nRemote Equation:\n")
                    f.write(str(equation))
                    
        except Exception as e:
            rewards.append(0.0)
            print(f"Error while processing completion: {e}")
            
    return rewards

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    # select a random subset of 50k samples
    dataset = dataset.shuffle(seed=42).select(range(50000))

    #####################
    # Prepare and format dataset
    #####################

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(numbers, target):
        """
        Generate a prompt to train the model in crafting effective prompts for a remote LLM.
        The model should learn to:
        1. Be clear and explicit in instructions
        2. Structure information in an easy-to-parse way
        3. Include constraints without being overly restrictive
        4. Handle potential LLM limitations or biases
        """
        r1_prefix = [{
            "role": "system",
            "content": """You are an expert prompt engineer who crafts clear and effective prompts for LLMs.
Your goal is to create prompts that maximize the chance of getting a valid mathematical equation as a response.
Focus on clarity, structure, and explicit constraints while keeping the prompt general enough for any LLM."""
          },
          { 
            "role": "user",
            "content": f"""I need help crafting an effective prompt for a remote LLM to generate a mathematical equation.

Given Information:
- Numbers to use: {numbers}
- Desired result: {target}

Required Constraints:
- Each number must be used exactly once
- Only basic arithmetic operations (+, -, *, /) are allowed
- Response must include deep and detailed reasoning with an accurate and final valid equation. The response format must include: some reasoning tokens ... <equation>equation here</equation>. For instance, <equation>(30 * 14) - 2</equation>

Think step by step about:
1. How to present the information clearly without assuming LLM capabilities
2. Ways to encourage a direct, equation-only response
3. How to prevent common LLM behaviors like:
   - Providing partial solutions
   - Ignoring some constraints
4. Balance between being specific and being too restrictive

Show your prompt engineering reasoning in <think> </think> tags.
Put your final crafted prompt in <prompt> </prompt> tags."""
          },
          {
            "role": "assistant",
            "content": "Let me craft an effective prompt that will guide the LLM in solving the problem. \n<think>"
          }]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
            "target": target, 
            "nums": numbers
        }

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[format_reward_func, equation_reward_func],
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()