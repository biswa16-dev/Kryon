import os
import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("[WARNING] peft not installed — running full fine-tune instead of LoRA.", flush=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.training.prepare_dataset import prepare_training_dataset


class AcademicQADataset(Dataset):
    def __init__(self, samples, tokenizer, max_input_len=320, max_target_len=160):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        inp_text = item["input_text"]
        tgt_text = item["target_text"]

        inp_enc = self.tokenizer(
            inp_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        tgt_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = tgt_enc.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inp_enc.input_ids.squeeze(0),
            "attention_mask": inp_enc.attention_mask.squeeze(0),
            "labels": labels
        }


def train_model(
    base_model_name: str = "google/flan-t5-base",
    output_dir: str = "models/kryon-academic-t5",
    epochs: int = 3,
    batch_size: int = 6,
    learning_rate: float = 3e-4,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    warmup_steps: int = 20,
):
    print("=" * 60, flush=True)
    print("[Kryon AI] Optimized LoRA Fine-Tuning on Academic Regulations", flush=True)
    print(f"Base Model: {base_model_name}", flush=True)
    print(f"LoRA: {'Enabled (r={}, alpha={})'.format(lora_r, lora_alpha) if use_lora and HAS_PEFT else 'Disabled (full fine-tune)'}", flush=True)
    print(f"Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}", flush=True)
    print("=" * 60, flush=True)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Prepare Dataset
    samples = prepare_training_dataset()
    print(f"Loaded {len(samples)} regulation training pairs.", flush=True)

    # 2. Load Model & Tokenizer
    print(f"Loading tokenizer and model weights for {base_model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    # 3. Apply LoRA adapters if available
    if use_lora and HAS_PEFT:
        print("Applying LoRA adapters to model...", flush=True)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q", "v"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    model.to(device)
    model.train()

    # 4. Create DataLoader
    dataset = AcademicQADataset(samples, tokenizer, max_input_len=320, max_target_len=160)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 5. Optimizer with warmup schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(dataloader) * epochs
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    epoch_losses = []

    print(f"\nStarting Training ({epochs} epochs, {len(dataloader)} batches/epoch):", flush=True)
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        batches = 0

        for step, batch in enumerate(dataloader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            batches += 1

            if step % 15 == 0 or step == len(dataloader):
                cur_loss = running_loss / batches
                print(f"  Epoch [{epoch}/{epochs}] Step [{step}/{len(dataloader)}] - Batch Loss: {loss.item():.4f} - Avg: {cur_loss:.4f}", flush=True)

        avg_loss = running_loss / max(1, batches)
        epoch_losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f"==> Epoch [{epoch}/{epochs}] Complete - Avg Loss: {avg_loss:.4f} - LR: {current_lr:.6f}", flush=True)

    total_training_time = time.time() - start_time
    print("-" * 60, flush=True)
    print(f"[SUCCESS] Training Complete in {total_training_time:.2f} seconds.", flush=True)
    print(f"   Initial Loss: {epoch_losses[0]:.4f} -> Final Loss: {epoch_losses[-1]:.4f}", flush=True)

    # 6. Save Model and Tokenizer (Safely handle Windows file handles)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Clean any locked or existing safetensor files in target directory
    for existing in out_path.glob("*.safetensors"):
        try:
            existing.unlink()
        except Exception:
            pass

    print(f"Saving fine-tuned model to {output_dir}...", flush=True)

    if use_lora and HAS_PEFT:
        # Merge LoRA weights into base model
        print("Merging LoRA adapters into base model for deployment...", flush=True)
        try:
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(str(out_path), safe_serialization=True)
        except Exception as e:
            print(f"Merged save warning ({e}), saving with standard PyTorch weights...", flush=True)
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(str(out_path), safe_serialization=False)
    else:
        model.save_pretrained(str(out_path))

    tokenizer.save_pretrained(str(out_path))

    # 7. Save Training Metrics Log for Academic Reporting
    metrics = {
        "model_name": "kryon-academic-t5",
        "base_model": base_model_name,
        "fine_tuning_method": "LoRA (r={}, alpha={})".format(lora_r, lora_alpha) if use_lora and HAS_PEFT else "Full Fine-Tune",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "samples_count": len(samples),
        "initial_loss": round(epoch_losses[0], 4),
        "final_loss": round(epoch_losses[-1], 4),
        "loss_history": [round(l, 4) for l in epoch_losses],
        "training_duration_seconds": round(total_training_time, 2),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    metrics_file = out_path / "training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved training report -> {metrics_file}", flush=True)
    print("=" * 60, flush=True)
    return metrics


if __name__ == "__main__":
    train_model()
