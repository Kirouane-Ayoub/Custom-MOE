import argparse
import time

import torch
from torch.utils.data import DataLoader

import settings
from dataset import TextDataset, new_tokenizer
from model import SMoELanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train SMoE Language Model")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="MOE_darija.pt",
        help="Path to save the model checkpoint",
    )
    parser.add_argument(
        "--total_steps", type=int, default=10000, help="Total number of training steps"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1000,
        help="Interval for evaluation and logging",
    )
    return parser.parse_args()


def train(
    model: SMoELanguageModel,
    train_data: DataLoader,
    val_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    total_steps: int = 10000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    clip_grad_norm: float = 1.0,
    lr_scheduler=None,
    eval_interval: int = 1000,
    save_path: str = "MOE_darija.pt",
):
    model = model.to(device)
    model.train()

    print("Training...")
    step = 0
    total_loss = 0.0
    start_time = time.time()

    while step < total_steps:
        for batch in train_data:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            _, loss = model(input_ids, labels)

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            # Update weights
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step(loss.detach().item())

            total_loss += loss.item()
            step += 1

            # Print step-wise loss and elapsed time periodically
            if step % eval_interval == 0:
                avg_loss = total_loss / eval_interval
                elapsed_time = time.time() - start_time
                print(
                    f"Step {step}/{total_steps} | Average loss: {avg_loss:.4f} | Elapsed time: {elapsed_time:.2f}s"
                )
                total_loss = 0.0
                start_time = time.time()

                # Evaluation Phase
                model.eval()
                eval_loss = 0
                with torch.no_grad():
                    for val_batch in val_data:
                        input_ids, labels = val_batch
                        input_ids, labels = input_ids.to(device), labels.to(device)
                        _, loss = model(input_ids, labels)
                        eval_loss += loss.item()

                avg_eval_loss = eval_loss / len(val_data)
                print(f"Step {step}, Evaluation Loss: {avg_eval_loss:.4f}")
                model.train()

            # Stop if total steps reached
            if step >= total_steps:
                break

        # Early stop if steps are already reached within epoch
        if step >= total_steps:
            break

    torch.save(model.state_dict(), save_path)
    print("Training complete!")


if __name__ == "__main__":
    args = parse_args()

    model = SMoELanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3
    )

    train_dataset = TextDataset(settings.TRAINING_FILE, new_tokenizer, max_length=512)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    eval_dataset = TextDataset(settings.EVAL_FILE, new_tokenizer, max_length=512)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    train(
        model,
        train_data=train_loader,
        val_data=eval_loader,
        optimizer=optimizer,
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        save_path=args.save_path,
        lr_scheduler=lr_scheduler,
    )
