"""
QLoRA fine-tuning entry point for the DBpedia Hindi Chapter — Experiment 1.

Usage:
    python3 train.py                                  # uses default config
    python3 train.py training.learning_rate=1e-5       # override learning rate
    python3 train.py data.max_samples=200               # smoke test on a subset

Resuming after an interruption:
    Just re-run the exact same command. If a checkpoint already exists in
    output_dir, training automatically resumes from the latest one instead
    of starting over, and the zeroth-step baseline eval is skipped (since
    it already ran before the interruption).
"""

import os
import re
import json
import math
import glob
import random
import inspect

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
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
        [m for m in user_turns if m["role"] == "user"],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, reference


# ─────────────────────────────────────────────────────────────
# Checkpoint resume helper
# ─────────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir):
    """
    Looks for existing HuggingFace Trainer checkpoints (folders named
    'checkpoint-<step>') inside output_dir. Returns the path to the
    latest one (highest step number) if any exist, else None.

    This lets a re-run of the exact same command automatically resume
    training instead of starting over from scratch if the process was
    previously interrupted (crash, server issue, disconnection, etc.).
    """
    if not os.path.isdir(output_dir):
        return None

    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None

    def step_num(path):
        match = re.search(r"checkpoint-(\d+)$", path)
        return int(match.group(1)) if match else -1

    latest = max(checkpoints, key=step_num)
    return latest


# ─────────────────────────────────────────────────────────────
# Generation-based evaluation — pass@1 and valid_format_rate
#
# NOTE: this is no longer registered as a per-eval-step Trainer
# callback. Per mentor guidance (running it every 0.25 epoch added
# significant overhead for little signal, since early in training the
# model is "just generating, not really learning"), it is now only
# called manually: once before training starts (the zeroth-step
# baseline, so we know how the untrained model performs) and can be
# run again offline afterward, separately, against saved checkpoints
# (e.g. via evaluate.py), rather than blocking training itself.
# ─────────────────────────────────────────────────────────────

def run_generation_eval(model, tokenizer, eval_dataset, n_samples=50,
                         max_new_tokens=256, step_label=0):
    """
    Runs manual greedy-decoding generation on a sample of the validation
    set and returns (pass_at_1, valid_format_rate). Logs to W&B under the
    given step_label.

    We deliberately do NOT call model.generate() here: Gemma3's KV-cache
    handling has a confirmed, currently-open bug (huggingface/transformers
    #36815) that injects a stray extra dimension into the internal
    generate() loop and crashes mid-decode. Since we only ever need greedy
    decoding (do_sample=False), we implement it directly so every tensor
    shape is explicit and under our control, with no dependency on that
    internal code path.

    past_key_values is reused across steps so each step only processes the
    single newly generated token instead of re-running the whole growing
    sequence from scratch.
    """
    model.eval()
    correct = 0
    valid_format = 0

    n_samples = min(n_samples, len(eval_dataset))
    indices = random.sample(range(len(eval_dataset)), n_samples)
    samples = [eval_dataset[i] for i in indices]

    eos_id = tokenizer.eos_token_id

    for idx, example in enumerate(samples):
        print(f"[eval step={step_label}] generating sample {idx + 1}/{len(samples)}...", flush=True)

        prompt, reference = get_prompt_and_reference(example, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Squeeze away any unexpected extra dimensions defensively.
        if input_ids.dim() > 2:
            input_ids = input_ids.view(input_ids.shape[0], -1)
        if attention_mask.dim() > 2:
            attention_mask = attention_mask.view(attention_mask.shape[0], -1)

        prompt_len = input_ids.shape[1]
        generated = input_ids

        with torch.no_grad():
            past_key_values = None
            current_input = generated
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=current_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=-1
                )
                past_key_values = outputs.past_key_values
                current_input = next_token

                if next_token.item() == eos_id:
                    break

        output_ids = generated

        generated_text = tokenizer.decode(
            output_ids[0][prompt_len:],
            skip_special_tokens=True,
        )
        predicted = extract_final_answer(generated_text)

        if is_valid_slug_format(generated_text):
            valid_format += 1
        if predicted.strip() == reference.strip():
            correct += 1

    n = len(samples)
    pass_at_1 = correct / n if n else 0.0
    valid_format_rate = valid_format / n if n else 0.0

    print(f"[eval] step={step_label} pass@1={pass_at_1:.3f} "
          f"valid_format_rate={valid_format_rate:.3f}")

    if wandb.run is not None:
        wandb.log({
            "eval/pass_at_1": pass_at_1,
            "eval/valid_format_rate": valid_format_rate,
        }, step=step_label)

    model.train()
    return pass_at_1, valid_format_rate


# ─────────────────────────────────────────────────────────────
# Version-adaptive helpers
#
# trl's API has shifted between versions we've tested against
# (0.11.4 during initial development, then upgraded to 1.7.1 to get
# Gemma 3 support from a newer transformers release). Rather than
# hardcode argument names that may drift again, we introspect the
# installed trl at runtime and build kwargs dicts accordingly.
# ─────────────────────────────────────────────────────────────

def build_sft_config_kwargs(cfg, eval_steps, save_steps):
    kwargs = dict(
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
        dataset_text_field="text",
        packing=False,
        seed=cfg.seed,
    )

    sig_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sig_params:
        kwargs["max_length"] = cfg.model.max_seq_length
    elif "max_seq_length" in sig_params:
        kwargs["max_seq_length"] = cfg.model.max_seq_length
    else:
        print("[warn] Neither max_length nor max_seq_length found on SFTConfig "
              "in this trl version — sequence length will use the default.")

    # Drop any kwarg SFTConfig doesn't actually accept, rather than crash.
    kwargs = {k: v for k, v in kwargs.items() if k in sig_params}
    return kwargs


def build_trainer_kwargs(model, sft_config, train_dataset, eval_dataset, tokenizer):
    kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    sig_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sig_params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig_params:
        kwargs["tokenizer"] = tokenizer
    else:
        print("[warn] Neither processing_class nor tokenizer found on SFTTrainer "
              "— relying on the trainer's default tokenizer handling.")
    return kwargs


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Check for an existing checkpoint (resume support) ────
    # If output_dir already has checkpoint-* folders, this is a re-run
    # after an interruption. We resume training from the latest one, and
    # skip the zeroth-step baseline eval since it already ran previously.
    resume_checkpoint = find_latest_checkpoint(cfg.output_dir)
    is_resuming = resume_checkpoint is not None
    if is_resuming:
        print(f"Found existing checkpoint: {resume_checkpoint}")
        print("Resuming training from this checkpoint (skipping zeroth-step baseline eval).")
    else:
        print("No existing checkpoint found — starting fresh.")

    # ── W&B ──────────────────────────────────────────────────
    # Use a stable run id derived from the run name so a resumed run
    # continues logging into the SAME W&B run instead of creating a
    # new, disconnected one.
    wandb_run_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", cfg.logging.run_name)
    wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        name=cfg.logging.run_name,
        tags=list(cfg.logging.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
        id=wandb_run_id,
        resume="allow",
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
        attn_implementation="eager",
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

    # ── Zeroth-step baseline eval ─────────────────────────────
    # Run generation-based eval BEFORE any training happens, so we have
    # a "how does the untrained/base model do" reference point to compare
    # against later checkpoints. Requested by Aditya during the July 9
    # sync so pass@1 / valid_format_rate progress can be shown from a
    # true zero point, not just from step 1 onward.
    #
    # Skipped entirely when resuming from a checkpoint — it already ran
    # during the original (interrupted) launch, and re-running it here
    # would just waste time and log a confusing duplicate step=0 point.
    if not is_resuming:
        print("Running zeroth-step baseline evaluation (before training)...")
        run_generation_eval(
            model, tokenizer, eval_dataset,
            n_samples=50, max_new_tokens=256, step_label=0,
        )
    else:
        print("Skipping zeroth-step baseline eval (already logged before this resume).")

    # ── Training arguments (version-adaptive) ────────────────
    sft_config = SFTConfig(**build_sft_config_kwargs(cfg, eval_steps, save_steps))

    # ── Trainer (version-adaptive) ────────────────────────────
    trainer = SFTTrainer(**build_trainer_kwargs(
        model, sft_config, train_dataset, eval_dataset_for_loss, tokenizer
    ))

    # NOTE: the generation-based eval callback (pass@1 / valid_format_rate
    # every 0.25 epoch) has been intentionally removed here per mentor
    # guidance — it added significant overhead for little signal early in
    # training. The built-in SFTTrainer loss-based eval (fast, using
    # eval_dataset_for_loss) still runs on the normal eval_steps cadence.
    # Generation-based eval is now only run once here as a baseline, and
    # can be run again offline afterward against saved checkpoints
    # (e.g. via evaluate.py) without blocking training time.

    # ── Train ────────────────────────────────────────────────
    # resume_from_checkpoint=None is a no-op (normal fresh start); passing
    # the actual checkpoint path makes the Trainer restore model weights,
    # optimizer state, and step count, then continue from exactly where
    # it left off instead of restarting from step 0.
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ── Save final adapter ───────────────────────────────────
    final_path = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved final model to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
