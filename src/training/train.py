import os
import sys
import time
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.training.prepare_dataset import prepare_training_dataset


class AcademicQADataset(Dataset):
    def __init__(self, samples, tokenizer, max_input_len=512, max_target_len=128):
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

        # Tokenize input
        inp_enc = self.tokenizer(
            inp_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target
        tgt_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = tgt_enc.input_ids.squeeze(0)
        # Replace padding token id with -100 so loss ignores it
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inp_enc.input_ids.squeeze(0),
            "attention_mask": inp_enc.attention_mask.squeeze(0),
            "labels": labels
        }


def train_model(
    base_model_name: str = "google/flan-t5-small",
    output_dir: str = "models/kryon-academic-t5",
    epochs: int = 4,
    batch_size: int = 4,
    learning_rate: float = 2e-4
):
    print("=" * 60)
    print("[Kryon AI] Fine-Tuning Sequence-to-Sequence Model on Regulations")
    print(f"Base Architecture: {base_model_name}")
    print(f"Device: {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Prepare Dataset
    samples = prepare_training_dataset()
    print(f"Loaded {len(samples)} regulation training pairs.")

    # 2. Load Model & Tokenizer
    print("Loading base tokenizer and model weights...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model.to(device)
    model.train()

    # 3. Create DataLoader
    dataset = AcademicQADataset(samples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Optimizer & Training Configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    epoch_losses = []

    print("\nStarting Training Loop:")
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

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg_loss = running_loss / max(1, batches)
        epoch_losses.append(avg_loss)
        print(f"  Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    total_training_time = time.time() - start_time
    print("-" * 60)
    print(f"[SUCCESS] Training Complete in {total_training_time:.2f} seconds.")
    print(f"   Initial Loss: {epoch_losses[0]:.4f} -> Final Loss: {epoch_losses[-1]:.4f}")

    # 5. Save Model and Tokenizer Artifacts
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving fine-tuned model checkpoint to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 6. Save Training Metrics Log for Academic Reporting
    metrics = {
        "model_name": "kryon-academic-t5",
        "base_model": base_model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
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

    print(f"Saved training report -> {metrics_file}")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    train_model()
