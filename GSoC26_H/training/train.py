"""
QLoRA fine-tuning entry point for the DBpedia Hindi Chapter — Experiment 1.

Usage:
    python3 train.py                                  # uses default config
    python3 train.py training.learning_rate=1e-5       # override learning rate
    python3 train.py data.max_samples=200               # smoke test on a subset
"""

import os
import re
import json
import math
import random

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import wandb


# ─────────────────────────────────────────────────────────────
# Data formatting
# ─────────────────────────────────────────────────────────────

def merge_system_into_user(messages):
    """
    Gemma's chat template does not reliably support a separate
    'system' role, so we fold the system instruction into the
    first user turn instead of dropping it.
    """
    merged = []
    pending_system = None
    for m in messages:
        if m["role"] == "system":
            pending_system = m["content"]
        elif m["role"] == "user":
            content = m["content"]
            if pending_system:
                content = f"{pending_system}\n\n{content}"
                pending_system = None
            merged.append({"role": "user", "content": content})
        else:
            merged.append(m)
    return merged


def build_formatter(tokenizer):
    def formatting_func(example):
        merged = merge_system_into_user(example["messages"])
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=False
        )
    return formatting_func


# ─────────────────────────────────────────────────────────────
# Slug-format validation + exact-match helpers (for evaluation)
# ─────────────────────────────────────────────────────────────

def extract_final_answer(text):
    """Pull the slug answer out of either an Optimal or CoT trace."""
    if "[ANSWER]" in text:
        text = text.split("[ANSWER]")[-1]
    return text.strip()


def is_valid_slug_format(text):
    text = extract_final_answer(text)
    if text == "NONE":
        return True
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return False
    for line in lines:
        parts = line.split("|")
        if len(parts) != 3 or not all(p.strip() for p in parts):
            return False
    return True


def get_prompt_and_reference(example, tokenizer):
    """
    Given a raw dataset example, build the generation prompt (system+user,
    with an assistant generation cue) and the ground-truth final answer.
    """
    messages = example["messages"]
    user_turns = merge_system_into_user(messages)
    reference = extract_final_answer(messages[-1]["content"])
    prompt = tokenizer.apply_chat_template(
        user_turns[:-0] if user_turns[-1]["role"] != "assistant" else user_turns[:-1],
        tokenize=False,
        add_generation_prompt=True,
    ) if False else tokenizer.apply_chat_template(
        [m for m in user_turns if m["role"] == "user"],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, reference


# ─────────────────────────────────────────────────────────────
# Custom evaluation callback — pass@1 and valid_format_rate
# ─────────────────────────────────────────────────────────────

class GenerationEvalCallback(TrainerCallback):
    """
    Runs actual text generation on a sample of the validation set at every
    evaluation point, and logs pass@1 (exact match against ground truth)
    and valid_format_rate (fraction of outputs in correct slug format).

    This is separate from the Trainer's built-in eval_loss, which only
    measures teacher-forced loss and does not reflect real generation quality.
    """

    def __init__(self, tokenizer, eval_dataset, n_samples=50, max_new_tokens=256):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        n_samples = min(n_samples, len(eval_dataset))
        indices = random.sample(range(len(eval_dataset)), n_samples)
        self.samples = [eval_dataset[i] for i in indices]

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        model.eval()
        correct = 0
        valid_format = 0

        for example in self.samples:
            prompt, reference = get_prompt_and_reference(example, self.tokenizer)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            predicted = extract_final_answer(generated)

            if is_valid_slug_format(generated):
                valid_format += 1
            if predicted.strip() == reference.strip():
                correct += 1

        n = len(self.samples)
        pass_at_1 = correct / n if n else 0.0
        valid_format_rate = valid_format / n if n else 0.0

        print(f"[eval] step={state.global_step} pass@1={pass_at_1:.3f} "
              f"valid_format_rate={valid_format_rate:.3f}")

        if wandb.run is not None:
            wandb.log({
                "eval/pass_at_1": pass_at_1,
                "eval/valid_format_rate": valid_format_rate,
            }, step=state.global_step)

        model.train()


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── W&B ──────────────────────────────────────────────────
    wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        name=cfg.logging.run_name,
        tags=list(cfg.logging.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Tokenizer ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model (4-bit quantized) ──────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.model.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, cfg.model.quantization.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.model.quantization.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.training.gradient_checkpointing
    )

    # ── LoRA ─────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        lora_dropout=cfg.model.lora.lora_dropout,
        target_modules=list(cfg.model.lora.target_modules),
        bias=cfg.model.lora.bias,
        task_type=cfg.model.lora.task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Data ─────────────────────────────────────────────────
    train_dataset = load_dataset("json", data_files=cfg.data.train_file, split="train")
    eval_dataset  = load_dataset("json", data_files=cfg.data.val_file, split="train")

    if cfg.data.max_samples is not None:
        train_dataset = train_dataset.select(range(min(cfg.data.max_samples, len(train_dataset))))
        eval_dataset  = eval_dataset.select(range(min(cfg.data.max_samples, len(eval_dataset))))
        print(f"[smoke test] using {len(train_dataset)} train / {len(eval_dataset)} eval examples")

    formatting_func = build_formatter(tokenizer)
    train_dataset = train_dataset.map(lambda ex: {"text": formatting_func(ex)})
    eval_dataset_for_loss = eval_dataset.map(lambda ex: {"text": formatting_func(ex)})

    # ── Convert "every 0.25 epoch" into real step counts ─────
    effective_batch_size = (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    eval_steps = max(1, round(steps_per_epoch * cfg.training.eval_steps))
    save_steps = max(1, round(steps_per_epoch * cfg.training.save_steps))

    print(f"steps_per_epoch={steps_per_epoch}  eval_steps={eval_steps}  save_steps={save_steps}")

    # ── Training arguments ───────────────────────────────────
    sft_config = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        optim=cfg.training.optim,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=eval_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=save_steps,
        logging_steps=cfg.training.logging_steps,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        report_to=["wandb"],
        run_name=cfg.logging.run_name,
        max_seq_length=cfg.model.max_seq_length,
        dataset_text_field="text",
        packing=False,
        seed=cfg.seed,
    )

    # ── Trainer ──────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_for_loss,
        processing_class=tokenizer,
    )

    trainer.add_callback(
        GenerationEvalCallback(tokenizer, eval_dataset, n_samples=50)
    )

    # ── Train ────────────────────────────────────────────────
    trainer.train()

    # ── Save final adapter ───────────────────────────────────
    final_path = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved final model to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
