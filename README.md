# Rotation Detection Neural Network

A deep learning system that accurately predicts the rotation angle of images using a modified ResNet18 architecture with trigonometric output representation.

## Architecture Overview

```mermaid
flowchart TB
    subgraph Input["INPUT LAYER"]
        style Input fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
        I[Image<br/>3×224×224]
        style I fill:#2196f3,stroke:#1565c0,color:#fff
    end
    
    subgraph Conv1["CONV BLOCK 1"]
        style Conv1 fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
        C1[Conv 7×7<br/>64 filters<br/>stride 2]
        BN1[BatchNorm]
        R1[ReLU]
        MP1[MaxPool 3×3<br/>stride 2]
        
        style C1 fill:#81c784,stroke:#4caf50
        style BN1 fill:#a5d6a7,stroke:#66bb6a
        style R1 fill:#c8e6c9,stroke:#81c784
        style MP1 fill:#81c784,stroke:#4caf50
    end
    
    subgraph Layer1["LAYER 1"]
        style Layer1 fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
        L1A[BasicBlock<br/>64 channels]
        L1B[BasicBlock<br/>64 channels]
        
        style L1A fill:#a5d6a7,stroke:#66bb6a
        style L1B fill:#a5d6a7,stroke:#66bb6a
    end
    
    subgraph Layer2["LAYER 2"]
        style Layer2 fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
        L2A[BasicBlock<br/>128 channels]
        L2B[BasicBlock<br/>128 channels]
        DS2[Downsample<br/>stride 2]
        
        style L2A fill:#a5d6a7,stroke:#66bb6a
        style L2B fill:#a5d6a7,stroke:#66bb6a
        style DS2 fill:#81c784,stroke:#4caf50
    end
    
    subgraph Layer3["LAYER 3"]
        style Layer3 fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
        L3A[BasicBlock<br/>256 channels]
        L3B[BasicBlock<br/>256 channels]
        DS3[Downsample<br/>stride 2]
        
        style L3A fill:#a5d6a7,stroke:#66bb6a
        style L3B fill:#a5d6a7,stroke:#66bb6a
        style DS3 fill:#81c784,stroke:#4caf50
    end
    
    subgraph Layer4["LAYER 4"]
        style Layer4 fill:#ffebee,stroke:#ef5350,stroke-width:3px
        L4A[BasicBlock<br/>512 channels]
        L4B[BasicBlock<br/>512 channels]
        DS4[Downsample<br/>stride 2]
        
        style L4A fill:#ef5350,stroke:#e53935,color:#fff
        style L4B fill:#ef5350,stroke:#e53935,color:#fff
        style DS4 fill:#f44336,stroke:#d32f2f,color:#fff
    end
    
    subgraph Head["CUSTOM HEAD"]
        style Head fill:#fff3e0,stroke:#ff6f00,stroke-width:3px
        GAP[Global<br/>Avg Pool<br/>7×7→1×1]
        FL[Flatten<br/>512]
        DO[Dropout<br/>p=0.25]
        FC[Linear<br/>512→2]
        OUT((cos θ<br/>sin θ))
        
        style GAP fill:#ffb74d,stroke:#ff9800
        style FL fill:#ffa726,stroke:#fb8c00
        style DO fill:#ff9800,stroke:#f57c00,color:#fff
        style FC fill:#ff7043,stroke:#ff5722,color:#fff
        style OUT fill:#ff5722,stroke:#d84315,color:#fff
    end
    
    I --> C1
    C1 --> BN1
    BN1 --> R1
    R1 --> MP1
    
    MP1 --> L1A
    L1A --> L1B
    
    L1B --> L2A
    L2A --> L2B
    L2B --> DS2
    
    DS2 --> L3A
    L3A --> L3B
    L3B --> DS3
    
    DS3 --> L4A
    L4A --> L4B
    L4B --> DS4
    
    DS4 --> GAP
    GAP --> FL
    FL --> DO
    DO --> FC
    FC --> OUT
    
    %% Annotations
    I -.- A1[Batch × 3 × 224 × 224]:::annot
    MP1 -.- A2[Batch × 64 × 56 × 56]:::annot
    DS2 -.- A3[Batch × 128 × 28 × 28]:::annot
    DS3 -.- A4[Batch × 256 × 14 × 14]:::annot
    DS4 -.- A5[Batch × 512 × 7 × 7]:::annot
    FL -.- A6[Batch × 512]:::annot
    OUT -.- A7[Batch × 2]:::annot
    
    classDef annot fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray: 5 5
```

## Key Features

- **Rotation Range**: Full 0-360° angle detection
- **Output Representation**: Trigonometric (cos, sin) for continuous circular values
- **Architecture**: Modified ResNet18 with selective layer unfreezing
- **Performance**: ~13.8° Mean Absolute Error
- **Training Time**: ~15-20 minutes on RTX 5080 GPU for 1000 initial samples
- **Robustness**: Early stopping, gradient clipping, and learning rate scheduling

## Requirements

```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
fiftyone>=0.22.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=10.0.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rotation-detection-nn.git
cd rotation-detection-nn

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

```bash
# Prepare the rotated dataset
python prepare_data.py

# Configuration options in prepare_data.py:
# - SOURCE_MAX_SAMPLES: 20000 (number of original images)
# - NUM_ROTATIONS_PER_IMAGE: 6 (augmentation factor)
# - PADDING_STRATEGY: 'reflect' (options: crop, reflect, random_bg, alpha_mask)
# - ANGLE_RANGE: (0, 360) (full rotation range)
```

## Training

```bash
# Train the model
python train_model.py

# Key configuration in train_model.py:
# - BATCH_SIZE: 1024
# - NUM_EPOCHS: 50 (with early stopping)
# - LEARNING_RATE: 0.001
# - UNFROZEN_LAYERS: ("layer4", "fc")
```

## Model Details

### Architecture Modifications

1. **Backbone**: ResNet18 pretrained on ImageNet
2. **Frozen Layers**: All layers except layer4 and fc
3. **Custom Head**: 
   - Dropout (p=0.25)
   - Linear: 512 → 2 (cos, sin output)

### Loss Function

- **Type**: Mean Squared Error (MSE)
- **Target**: [cos(θ), sin(θ)] representation
- **Advantages**: 
  - Continuous representation (no discontinuity at 0°/360°)
  - Equal weight to all angles
  - Smooth gradients

### Training Strategy

1. **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
2. **LR Schedule**: StepLR (÷10 every 12 epochs)
3. **Gradient Clipping**: max_norm=1.0
4. **Early Stopping**: patience=5, min_delta=0.01°
5. **Batch Size**: 1024 (dynamically adjusted for GPU memory)

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Best Validation MAE | 13.82° |
| Training Time | 22 minutes |
| Model Parameters | 8.4M (trainable) |
| Total Parameters | 11.2M |

### Error Distribution

- **50th percentile**: ~10° error
- **90th percentile**: ~25° error
- **95th percentile**: ~35° error
- **Samples within ±15°**: ~65%

## Visualizations

The training script generates comprehensive visualizations:

1. **Training History**: Loss curves, MAE progression, learning rate schedule
2. **Error Analysis**: Distribution plots, scatter plots, percentile analysis
3. **Grad-CAM**: Attention heatmaps showing model focus areas
4. **Statistics**: Detailed error metrics and performance breakdowns

## Use Cases

- **Image Orientation Correction**: Automatically detect and correct image rotation
- **Quality Control**: Verify correct orientation in image processing pipelines
- **Data Augmentation**: Generate accurately labeled rotated datasets
- **Computer Vision Research**: Baseline for rotation-invariant algorithms

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- COCO dataset for providing diverse training images
- PyTorch and torchvision teams for the pretrained models
- FiftyOne for excellent dataset management tools