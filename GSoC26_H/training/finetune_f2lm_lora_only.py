"""
finetune_f2lm_lora_only.py — Plain LoRA (no 4-bit quantization) fine-tuning
of F2LLM-v2-1.7B, trained fresh from the base model for 9 epochs — matching
the total epoch count of the existing QLoRA result (3+6=9) for a fair,
single-variable comparison, per Debarghya's request.

Only change vs. the QLoRA script: base model loads in bf16 instead of
4-bit, and no prepare_model_for_kbit_training call. Everything else —
LoraConfig (r=16, q_proj+v_proj), data, lr, batch size, loss — identical.

Memory note: bf16 base model ≈ 3.4GB (vs ~850MB for 4-bit), plus LoRA
adapter + gradients/optimizer (small) + activations. Should still fit
comfortably within the A2's 15.3GB.

Run:
    python3 finetune_f2lm_lora_only.py
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW

BASE_MODEL = "codefuse-ai/F2LLM-v2-1.7B"
TRAIN_FILE = "/home/nsingh/f2lm_finetune_train.jsonl"
TEST_FILE = "/home/nsingh/f2lm_finetune_test.jsonl"
CATALOG_FILE = "/home/nsingh/f2lm_property_catalog.json"
OUTPUT_DIR = "/home/nsingh/f2lm_finetuned_lora_only"
RESULTS_FILE = "/home/nsingh/f2lm_finetune_lora_only_results.json"

EPOCHS = 9
BATCH_SIZE = 16
GRAD_ACCUM = 4
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool(token_embeddings, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def encode_texts(model, tokenizer, texts, batch_size=32):
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt"
            ).to(DEVICE)
            outputs = model(**encoded)
            vecs = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            vecs = F.normalize(vecs, p=2, dim=-1)
            all_vecs.append(vecs.cpu().float().numpy())
    model.train()
    return np.vstack(all_vecs)


def load_catalog():
    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)
    uri_to_text = {}
    for entry in catalog:
        short_uri = "dbo:" + entry["property_uri"].split("/")[-1]
        labels = (entry.get("labels_en") or []) + (entry.get("alt_labels") or [])
        text = " | ".join(labels[:3]) if labels else short_uri
        comment = entry.get("comment_en")
        if comment:
            text = text + " | " + comment
        uri_to_text[short_uri] = text
    return uri_to_text, catalog


def load_train_pairs(uri_to_text):
    pairs = []
    skipped = 0
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            gold_dbo = entry["gold_dbo"]
            if gold_dbo == "NONE" or gold_dbo not in uri_to_text:
                skipped += 1
                continue
            pairs.append((entry["predicate"], uri_to_text[gold_dbo]))
    print(f"  Loaded {len(pairs)} training pairs, skipped {skipped}")
    return pairs


def load_test_entries():
    entries = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    print(f"  Loaded {len(entries)} test entries")
    return entries


class PairDataset(TorchDataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    anchors = [b[0] for b in batch]
    positives = [b[1] for b in batch]
    return anchors, positives


def mnrl_loss(anchor_vecs, positive_vecs):
    scores = anchor_vecs @ positive_vecs.T
    labels = torch.arange(len(scores), device=scores.device)
    return F.cross_entropy(scores * 20.0, labels)


def evaluate_precision(model, tokenizer, catalog, test_entries, top_k_check=(1, 5, 10)):
    catalog_texts = []
    catalog_uris = []
    for entry in catalog:
        short_uri = "dbo:" + entry["property_uri"].split("/")[-1]
        labels = (entry.get("labels_en") or []) + (entry.get("alt_labels") or [])
        text = " | ".join(labels[:3]) if labels else short_uri
        catalog_texts.append(text)
        catalog_uris.append(short_uri)

    catalog_vecs = encode_texts(model, tokenizer, catalog_texts)
    predicates = [e["predicate"] for e in test_entries]
    gold_uris = [e["gold_dbo"] for e in test_entries]
    query_vecs = encode_texts(model, tokenizer, predicates)

    hits = {k: 0 for k in top_k_check}
    max_k = max(top_k_check)

    for i, q_vec in enumerate(query_vecs):
        sims = catalog_vecs @ q_vec
        top_idx = np.argsort(-sims)[:max_k]
        top_uris = [catalog_uris[idx] for idx in top_idx]
        for k in top_k_check:
            if gold_uris[i] in top_uris[:k]:
                hits[k] += 1

    n = len(test_entries)
    return {k: hits[k] / n for k in top_k_check}


def main():
    print("=" * 60)
    print("F2LLM-v2-1.7B Plain LoRA Fine-tuning (no quantization)")
    print(f"Epochs: {EPOCHS} (matching total QLoRA epochs for fair comparison)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("\nLoading catalog...")
    uri_to_text, catalog = load_catalog()
    print(f"Catalog size: {len(catalog)}")

    print("\nLoading training pairs...")
    train_pairs = load_train_pairs(uri_to_text)

    print("\nLoading test entries (held out)...")
    test_entries = load_test_entries()
    print(f"\nTrain: {len(train_pairs)} pairs")
    print(f"Test:  {len(test_entries)} entries (never seen during training)")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in bf16 (no quantization)...")
    base_model = AutoModel.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"GPU memory free after model load: {free_gb:.1f}GB")

    print("\n--- Baseline: precision BEFORE fine-tuning ---")
    baseline = evaluate_precision(model, tokenizer, catalog, test_entries)
    for k in (1, 5, 10):
        print(f"  precision@{k}: {baseline[k]:.3f}")

    dataset = PairDataset(train_pairs)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
    )

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = (len(dataloader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nStarting plain LoRA fine-tuning...")
    print(f"  epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}, accum={GRAD_ACCUM}")
    print(f"  total_steps={total_steps}, warmup_steps={warmup_steps}")

    model.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        for step, (anchors, positives) in enumerate(dataloader):
            anchor_enc = tokenizer(
                anchors, padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt"
            ).to(DEVICE)
            positive_enc = tokenizer(
                positives, padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt"
            ).to(DEVICE)

            anchor_out = model(**anchor_enc)
            positive_out = model(**positive_enc)

            anchor_vecs = mean_pool(anchor_out.last_hidden_state, anchor_enc["attention_mask"])
            positive_vecs = mean_pool(positive_out.last_hidden_state, positive_enc["attention_mask"])

            anchor_vecs = F.normalize(anchor_vecs, p=2, dim=-1)
            positive_vecs = F.normalize(positive_vecs, p=2, dim=-1)

            loss = mnrl_loss(anchor_vecs, positive_vecs) / GRAD_ACCUM
            loss.backward()

            epoch_loss += loss.item() * GRAD_ACCUM
            num_batches += 1

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 25 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"  epoch={epoch+1} step={global_step} "
                          f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        print(f"Epoch {epoch+1} done. avg_loss={epoch_loss/num_batches:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved fine-tuned model to: {final_path}")

    print("\n--- After LoRA fine-tuning: precision on held-out test set ---")
    after = evaluate_precision(model, tokenizer, catalog, test_entries)
    for k in (1, 5, 10):
        change = after[k] - baseline[k]
        direction = "up" if change > 0 else "down"
        print(f"  precision@{k}: {after[k]:.3f}  "
              f"(was {baseline[k]:.3f}, {direction} {abs(change):.3f})")

    results = {
        "baseline": {str(k): v for k, v in baseline.items()},
        "after_finetuning": {str(k): v for k, v in after.items()},
        "train_size": len(train_pairs),
        "test_size": len(test_entries),
        "epochs": EPOCHS,
        "method": "plain_lora_no_quantization",
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
