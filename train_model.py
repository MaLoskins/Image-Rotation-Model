# Licence headers and high-level comments are preserved.
import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.core.view import DatasetView
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import numpy as np
import random
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import time
import shutil
import json
from typing import Tuple, Dict, List, Optional
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
CONFIG = {
    "TEST_MODE": False,
    "DATASET_NAME": "coco-2017-rotated",
    "MODEL_PATH": Path("rotation_model_best.pth"),
    "CHECKPOINT_DIR": Path("checkpoints"),
    "UNFROZEN_LAYERS": ("layer4", "fc"),  # Back to original - only unfreeze final layers
    "LEARNING_RATE": 0.001,  # Back to original LR
    "WEIGHT_DECAY": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_WORKERS": 0 if os.name == 'nt' else 4,
    "VIZ_DIR": Path("visualizations"),
    "EARLY_STOPPING_PATIENCE": 5,
    "MIN_DELTA": 0.01,  # Minimum improvement to reset patience
    "GRADIENT_CLIP": 1.0,  # Gradient clipping value
}
CONFIG.update({
    "BATCH_SIZE": 4 if CONFIG["TEST_MODE"] else 1024,  # Back to original batch size
    "NUM_EPOCHS": 1 if CONFIG["TEST_MODE"] else 50,  # Increased epochs with early stopping
    "GRAD_CAM_COUNT": 2 if CONFIG["TEST_MODE"] else 10,
})


# --- HELPER FUNCTIONS ---
def circular_mae(pred_angles: torch.Tensor, true_angles: torch.Tensor) -> torch.Tensor:
    """Calculate mean absolute error for circular quantities (angles)."""
    diff = torch.abs(pred_angles - true_angles)
    return torch.min(diff, 360 - diff).mean()


def adjust_batch_size(initial_batch_size: int, device: str) -> int:
    """Dynamically adjust batch size based on available GPU memory."""
    if device == "cpu":
        return min(initial_batch_size, 32)
    
    # For GPU, try to maintain larger batch sizes
    try:
        # Start with the requested size
        test_size = initial_batch_size
        
        # Try progressively smaller sizes if needed
        sizes_to_try = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 1]
        sizes_to_try = [s for s in sizes_to_try if s <= initial_batch_size]
        
        for test_size in sizes_to_try:
            try:
                # Test allocation with some margin
                test_tensor = torch.randn(test_size, 3, 224, 224).to(device)
                test_output = torch.randn(test_size, 2).to(device)
                del test_tensor, test_output
                torch.cuda.empty_cache()
                return test_size
            except RuntimeError:
                torch.cuda.empty_cache()
                continue
        
        return 1
    except:
        return min(initial_batch_size, 32)


# --- VISUALIZATION HELPERS ---
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def register_hooks(self):
        """Register hooks for the target layer."""
        self.handles.append(self.target_layer.register_forward_hook(self.save_activation))
        self.handles.append(self.target_layer.register_full_backward_hook(self.save_gradient))
    
    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate GradCAM heatmap."""
        self.register_hooks()
        
        try:
            # Forward pass
            self.model.eval()
            output = self.model(x)
            
            # Use the sin component (index 1) for gradient computation
            if class_idx is None:
                class_idx = 1
            
            # Backward pass
            self.model.zero_grad()
            output[0][class_idx].backward(retain_graph=True)
            
            # Generate heatmap
            gradients = self.gradients.cpu().data.numpy()[0]
            activations = self.activations.cpu().data.numpy()[0]
            
            # Global average pooling
            weights = np.mean(gradients, axis=(1, 2))
            
            # Weighted combination
            cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
            cam = np.maximum(cam, 0)
            
            # Resize to input size
            cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
            
            # Normalize
            if cam.max() > 0:
                cam = cam / cam.max()
            
            return cam
        finally:
            self.remove_hooks()


def create_visualization_grid(images: List[np.ndarray], titles: List[str], 
                            save_path: Path, cols: int = 3):
    """Create a grid of visualization images."""
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_training_history(history: Dict[str, List[float]], save_path: Path):
    """Create comprehensive training history plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[0, 1].plot(history['val_mae'], label='Validation MAE', color='orange', linewidth=2)
    axes[0, 1].set_title('Validation MAE', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (degrees)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(bottom=0)
    
    # Learning rate plot
    if 'lr' in history:
        axes[1, 0].plot(history['lr'], label='Learning Rate', color='green', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Loss ratio plot (train/val)
    loss_ratio = np.array(history['train_loss']) / (np.array(history['val_loss']) + 1e-8)
    axes[1, 1].plot(loss_ratio, label='Train/Val Loss Ratio', color='red', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Overfitting Indicator', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_error_analysis(true_angles: np.ndarray, pred_angles: np.ndarray, save_path: Path):
    """Create detailed error analysis plots."""
    errors = (pred_angles - true_angles + 180) % 360 - 180
    abs_errors = np.abs(errors)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Error distribution
    axes[0, 0].hist(errors, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_title('Prediction Error Distribution', fontsize=14)
    axes[0, 0].set_xlabel('Error (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Scatter plot
    axes[0, 1].scatter(true_angles, pred_angles, alpha=0.5, s=10)
    axes[0, 1].plot([0, 360], [0, 360], 'r--', label='Perfect Prediction')
    axes[0, 1].set_title('True vs. Predicted Angles', fontsize=14)
    axes[0, 1].set_xlabel('True Angle (degrees)')
    axes[0, 1].set_ylabel('Predicted Angle (degrees)')
    axes[0, 1].set_xlim(0, 360)
    axes[0, 1].set_ylim(0, 360)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error vs true angle
    axes[0, 2].scatter(true_angles, abs_errors, alpha=0.5, s=10, color='orange')
    axes[0, 2].set_title('Absolute Error vs True Angle', fontsize=14)
    axes[0, 2].set_xlabel('True Angle (degrees)')
    axes[0, 2].set_ylabel('Absolute Error (degrees)')
    axes[0, 2].set_xlim(0, 360)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Error percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    error_percentiles = [np.percentile(abs_errors, p) for p in percentiles]
    axes[1, 0].bar(range(len(percentiles)), error_percentiles, 
                   tick_label=[f"{p}th" for p in percentiles])
    axes[1, 0].set_title('Error Percentiles', fontsize=14)
    axes[1, 0].set_ylabel('Absolute Error (degrees)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Cumulative error distribution
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    axes[1, 1].plot(sorted_errors, cumulative, linewidth=2)
    axes[1, 1].set_title('Cumulative Error Distribution', fontsize=14)
    axes[1, 1].set_xlabel('Absolute Error (degrees)')
    axes[1, 1].set_ylabel('Percentage of Samples (%)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, max(30, np.percentile(abs_errors, 99)))
    
    # Error statistics text
    stats_text = f"""Error Statistics:
    Mean Error: {np.mean(errors):.2f}°
    Mean Abs Error: {np.mean(abs_errors):.2f}°
    Std Dev: {np.std(errors):.2f}°
    Median Abs Error: {np.median(abs_errors):.2f}°
    90th Percentile: {np.percentile(abs_errors, 90):.2f}°
    Max Error: {np.max(abs_errors):.2f}°
    
    Samples within:
    ±5°: {np.sum(abs_errors <= 5) / len(abs_errors) * 100:.1f}%
    ±10°: {np.sum(abs_errors <= 10) / len(abs_errors) * 100:.1f}%
    ±15°: {np.sum(abs_errors <= 15) / len(abs_errors) * 100:.1f}%"""
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --- DATASET AND MODEL ---
class RotationDataset(Dataset):
    """PyTorch Dataset for loading rotated images and their angles."""
    def __init__(self, samples: List, transform=None):
        self.samples = samples
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        image = Image.open(sample.filepath).convert("RGB")
        
        # Get rotation angle - no augmentation
        angle = sample.rotation_angle
        
        # Convert to radians and create label
        angle_rad = np.deg2rad(angle)
        label_tensor = torch.tensor([np.cos(angle_rad), np.sin(angle_rad)], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_tensor, sample.filepath





def get_model(unfrozen_layers: Tuple[str, ...], dropout_rate: float = 0.25) -> nn.Module:
    """Create and configure the model with flexible unfreezing."""
    # Load pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze specified layers
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in unfrozen_layers):
            param.requires_grad = True
    
    # Replace final layer with simple head (like original)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_features, 2)  # Direct output of cos and sin
    )
    
    return model


def get_dataloaders(dataset: fo.Dataset, config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with random split."""
    # Get rotated samples and shuffle
    rotated_view = dataset.match(F("is_rotated") == True)
    rotated_view.shuffle(seed=51)  # Fixed seed for reproducibility
    
    # Simple random split (like original)
    val_split = int(0.2 * len(rotated_view))
    val_view = rotated_view.take(val_split)
    train_view = rotated_view.skip(val_split)
    
    # Get samples
    train_samples = list(train_view)
    val_samples = list(val_view)
    
    print(f"Training on {len(train_samples)} samples, validating on {len(val_samples)} samples.")
    
    # Define transforms (simplified, more like original)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets - no rotation augmentation
    train_dataset = RotationDataset(train_samples, train_transform)
    val_dataset = RotationDataset(val_samples, val_transform)
    
    # Adjust batch size if needed
    actual_batch_size = adjust_batch_size(config["BATCH_SIZE"], config["DEVICE"])
    if actual_batch_size != config["BATCH_SIZE"]:
        print(f"Adjusted batch size from {config['BATCH_SIZE']} to {actual_batch_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=actual_batch_size, 
        shuffle=True, 
        num_workers=config["NUM_WORKERS"],
        pin_memory=True if config["DEVICE"] == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=actual_batch_size, 
        shuffle=False, 
        num_workers=config["NUM_WORKERS"],
        pin_memory=True if config["DEVICE"] == "cuda" else False
    )
    
    return train_loader, val_loader


# --- TRAINING AND EVALUATION ---
def trig_to_deg(tensor: torch.Tensor) -> torch.Tensor:
    """Convert cos/sin representation to degrees."""
    return (torch.rad2deg(torch.atan2(tensor[:, 1], tensor[:, 0])) + 360) % 360


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                   optimizer: optim.Optimizer, device: str, epoch: int, 
                   gradient_clip: float = 1.0) -> float:
    """Train for one epoch with gradient clipping and progress bar."""
    model.train()
    running_loss = 0.0
    
    # Create progress bar with custom description
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}', 
                unit='batch', leave=True, position=0)
    
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item() * images.size(0)
        
        # Update progress bar
        avg_loss = running_loss / ((batch_idx + 1) * images.size(0))
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{avg_loss:.4f}'})
    
    return running_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
            device: str, epoch: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model on validation set with progress bar."""
    model.eval()
    running_loss = 0.0
    all_true_deg, all_pred_deg = [], []
    
    # Create progress bar
    pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}', 
                unit='batch', leave=True, position=0)
    
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # Convert to degrees
            true_deg = trig_to_deg(labels)
            pred_deg = trig_to_deg(outputs)
            
            all_true_deg.append(true_deg.cpu().numpy())
            all_pred_deg.append(pred_deg.cpu().numpy())
            
            # Update progress bar
            avg_loss = running_loss / ((batch_idx + 1) * images.size(0))
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{avg_loss:.4f}'})
    
    # Concatenate all predictions
    all_true_deg = np.concatenate(all_true_deg)
    all_pred_deg = np.concatenate(all_pred_deg)
    
    # Calculate MAE
    mae = circular_mae(torch.tensor(all_pred_deg), torch.tensor(all_true_deg)).item()
    
    return (running_loss / len(dataloader.dataset), mae, all_true_deg, all_pred_deg)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


def run_training(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                config: Dict) -> Dict[str, List[float]]:
    """Run the complete training loop with early stopping."""
    device = config["DEVICE"]
    model.to(device)
    
    # Setup training components - use standard MSE loss like original
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"]
    )
    
    # Learning rate scheduling - use StepLR like original
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["EARLY_STOPPING_PATIENCE"],
        min_delta=config["MIN_DELTA"]
    )
    
    # Create checkpoint directory
    config["CHECKPOINT_DIR"].mkdir(exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'lr': []
    }
    
    best_mae = float('inf')
    
    print("\n--- Starting Training ---")
    print(f"Device: {device}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total batches per epoch: {len(train_loader)}")
    
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['NUM_EPOCHS']}")
        print(f"{'='*60}")
        
        # Training
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1,
            config["GRADIENT_CLIP"]
        )
        
        # Validation
        val_loss, val_mae, _, _ = evaluate(model, val_loader, criterion, device, epoch+1)
        
        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val MAE: {val_mae:.2f}°")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            # Convert config paths to strings to avoid PyTorch 2.6 loading issues
            config_to_save = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae,
                'config': config_to_save
            }, config["MODEL_PATH"])
            print(f"  ✓ New best model saved with MAE: {best_mae:.2f}°")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = config["CHECKPOINT_DIR"] / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_path)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if early_stopping(val_mae):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\n✓ Training complete. Best model saved with MAE: {best_mae:.2f}°")
    return history


def generate_visualizations(model: nn.Module, val_loader: DataLoader, 
                          history: Dict[str, List[float]], config: Dict):
    """Generate comprehensive visualizations."""
    print("\n--- Generating Visualizations ---")
    viz_dir = config["VIZ_DIR"]
    viz_dir.mkdir(exist_ok=True)
    
    device = config["DEVICE"]
    
    # Load best model - fix for PyTorch 2.6
    checkpoint = torch.load(config["MODEL_PATH"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 1. Training history plots
    plot_training_history(history, viz_dir / "training_history.png")
    
    # 2. Error analysis
    criterion = nn.MSELoss()  # Use standard MSE loss
    print("\nAnalyzing model performance...")
    _, _, true_angles, pred_angles = evaluate(model, val_loader, criterion, device, epoch=0)
    
    if len(true_angles) > 0:
        plot_error_analysis(true_angles, pred_angles, viz_dir / "error_analysis.png")
    
    # 3. Grad-CAM visualizations
    if len(val_loader.dataset) > 0:
        print("\nGenerating Grad-CAM heatmaps...")
        grad_cam = GradCAM(model, target_layer=model.layer4[-1])
        
        gradcam_images = []
        gradcam_titles = []
        
        # Sample diverse angles
        angle_bins = np.linspace(0, 360, config["GRAD_CAM_COUNT"] + 1)
        
        # Create progress bar for GradCAM generation
        pbar = tqdm(range(min(config["GRAD_CAM_COUNT"], len(val_loader.dataset))), 
                   desc="Generating GradCAM", unit="image")
        
        for i in pbar:
            # Try to get samples from different angle ranges
            target_angle = (angle_bins[i] + angle_bins[i+1]) / 2
            
            # Find sample closest to target angle
            best_idx = 0
            best_diff = 360
            for idx in range(len(val_loader.dataset)):
                _, label, _ = val_loader.dataset[idx]
                angle = trig_to_deg(label.unsqueeze(0)).item()
                diff = abs(angle - target_angle)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
            
            # Get sample
            img_tensor, label, img_path = val_loader.dataset[best_idx]
            
            # Generate heatmap
            heatmap = grad_cam(img_tensor.unsqueeze(0).to(device))
            
            # Create visualization
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_img = cv2.resize(original_img, (224, 224))
            
            # Apply colormap to heatmap
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Superimpose
            superimposed = cv2.addWeighted(heatmap_colored, 0.4, original_img, 0.6, 0)
            
            # Predict angle
            model.eval()
            with torch.no_grad():
                pred = model(img_tensor.unsqueeze(0).to(device))
                pred_angle = trig_to_deg(pred).item()
                true_angle = trig_to_deg(label.unsqueeze(0)).item()
            
            gradcam_images.append(superimposed)
            gradcam_titles.append(f"True: {true_angle:.1f}°, Pred: {pred_angle:.1f}°")
        
        # Save grid
        if gradcam_images:
            create_visualization_grid(
                gradcam_images, gradcam_titles,
                viz_dir / "gradcam_grid.png",
                cols=min(3, len(gradcam_images))
            )
    
    print(f"\n✓ All visualizations saved to '{viz_dir}'")
    
    # 4. Save training summary
    config_to_save = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    summary = {
        "final_train_loss": history['train_loss'][-1],
        "final_val_loss": history['val_loss'][-1],
        "final_val_mae": history['val_mae'][-1],
        "best_val_mae": checkpoint['best_mae'],
        "total_epochs": len(history['train_loss']),
        "config": config_to_save
    }
    
    with open(viz_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    """Main execution function."""
    print(f"--- Configuration ---")
    print(json.dumps({k: str(v) if isinstance(v, Path) else v 
                     for k, v in CONFIG.items()}, indent=2))
    
    start_time = time.time()
    
    # Check dataset exists
    if not fo.dataset_exists(CONFIG["DATASET_NAME"]):
        print(f"\nError: Dataset '{CONFIG['DATASET_NAME']}' not found.")
        print("Please run 'prepare_data.py' first.")
        return
    
    # Load dataset
    dataset = fo.load_dataset(CONFIG["DATASET_NAME"])
    print(f"\nLoaded dataset with {len(dataset)} samples")
    print(f"Rotated samples: {len(dataset.match_tags('rotated'))}")
    
    # Create data loaders
    train_loader, val_loader = get_dataloaders(dataset, CONFIG)
    
    # Create model
    model = get_model(unfrozen_layers=CONFIG["UNFROZEN_LAYERS"])
    
    # Train model
    history = run_training(model, train_loader, val_loader, CONFIG)
    
    # Generate visualizations
    generate_visualizations(model, val_loader, history, CONFIG)
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"--- Training Summary ---")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Final validation MAE: {history['val_mae'][-1]:.2f}°")
    print(f"Best validation MAE: {min(history['val_mae']):.2f}°")
    
    if CONFIG["TEST_MODE"]:
        print("\n--- Running Self-Contained Test ---")
        assert CONFIG["MODEL_PATH"].exists(), "Model file not found"
        assert (CONFIG["VIZ_DIR"] / "training_history.png").exists(), "Training history plot not found"
        assert (CONFIG["VIZ_DIR"] / "error_analysis.png").exists(), "Error analysis plot not found"
        print("--- All Tests Passed ---")
        
        # Cleanup
        CONFIG["MODEL_PATH"].unlink()
        shutil.rmtree(CONFIG["VIZ_DIR"])
        shutil.rmtree(CONFIG["CHECKPOINT_DIR"])
        print("--- Test Cleanup Complete ---")


if __name__ == "__main__":
    main()