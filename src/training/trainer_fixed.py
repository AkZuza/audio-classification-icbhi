"""Improved training logic with class weighting."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path


class Trainer:
    """Improved trainer with class weighting and better loss handling."""

    def __init__(self, model, train_dataset, val_dataset, config, device):
        self.model = model. to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device

        # Training parameters
        self.epochs = config["training"]["epochs"]
        self.batch_size = config["training"]["batch_size"]
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
        self.mixed_precision = config["training"]["mixed_precision"]
        self.early_stopping_patience = config["training"]["early_stopping_patience"]

        # Calculate class weights for imbalanced dataset
        self.class_weights = self._calculate_class_weights()
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config["device"]["num_workers"],
            pin_memory=config["device"]["pin_memory"],
            drop_last=True,  # Drop last incomplete batch
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self. batch_size,
            shuffle=False,
            num_workers=config["device"]["num_workers"],
            pin_memory=config["device"]["pin_memory"],
        )

        # Weighted loss function
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # Optimizer
        optimizer_name = config["training"]["optimizer"]. lower()
        if optimizer_name == "adam":
            self. optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self. learning_rate, 
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = torch.optim. SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )

        # Learning rate scheduler
        scheduler_name = config["training"]["scheduler"].lower()
        if scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif scheduler_name == "plateau":
            self. scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self. optimizer, mode="min", factor=0.5, patience=10
            )
        elif scheduler_name == "step": 
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            self.scheduler = None

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.mixed_precision else None

        # Checkpoint directory
        self.checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=config["training"]["log_dir"])

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced dataset."""
        labels = [self.train_dataset.data[i][1] for i in range(len(self.train_dataset))]
        labels = np.array(labels)
        
        # Calculate class frequencies
        class_counts = np. bincount(labels)
        total_samples = len(labels)
        
        # Calculate weights (inverse frequency)
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"\nClass distribution:")
        for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
            class_name = self.config["classes"][i]
            print(f"  {class_name}: {count} samples (weight: {weight:.3f})")
        
        return class_weights

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]")

        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Mixed precision forward pass
            if self.mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self. model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self. optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()

            # Statistics
            running_loss += loss. item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss":  running_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                }
            )

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """Validate the model."""
        self.model. eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Val]")

            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self. device), labels.to(self.device)

                # Forward pass
                if self.mixed_precision:
                    with torch.cuda.amp. autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self. model(inputs)
                    loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss. item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": running_loss / (batch_idx + 1),
                        "acc": 100.0 * correct / total,
                    }
                )

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate:  {self.learning_rate}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}\n")

        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self. validate(epoch)

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else: 
                    self.scheduler.step()

            # Log to TensorBoard
            self. writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]["lr"], epoch)

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            # Print epoch summary
            print(
                f"\nEpoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_path = self.checkpoint_dir / "best_model.pt"
                self.save_checkpoint(best_path, epoch, val_loss)
                print(f"✓ Best model saved with validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.early_stopping_patience})")

            # Save periodic checkpoint
            if (epoch + 1) % self.config["training"]["save_every"] == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, val_loss)

            # Early stopping
            if self.patience_counter >= self. early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print("\n✓ Training completed!")
        self.writer.close()

        return self.history

    def save_checkpoint(self, path, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "class_weights": self.class_weights,
        }
        torch.save(checkpoint, path)
