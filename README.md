# NeuralIDE

NeuralIDE is an interactive architecture design studio and bidirectional PyTorch development environment. It is designed to bridge the gap between mental models of deep learning architectures and executable production code.

---

## 1. The Core Problem and Design Philosophy

### The Friction in Deep Learning Engineering

When developing deep learning models, engineers and researchers face recurring ergonomic hurdles:

1. **Runtime Dimension Failures**: Neural networks are directed acyclic graphs of tensor transformations. Yet, standard development relies on imperative Python code. A single typo in stride, padding, kernel size, or flatten boundaries produces silent dimensional bugs that only crash at runtime when the first forward pass executes:
   ```text
   RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x2048 and 1024x10)
   ```
2. **Mental Arithmetic Overhead**: Calculating output shapes, receptive fields, layer parameter sizes, and memory footprints across 50+ layer stacks is tedious and error-prone.
3. **The Diagram-to-Code Disconnect**: Publication diagrams in machine learning literature and actual model code exist in separate silos. Diagram software does not generate runnable PyTorch code, and raw Python code cannot be visually inspected or dynamically manipulated.

### The NeuralIDE Approach

NeuralIDE treats neural network architecture design as an integrated, visual, and mathematical discipline:

- **Deterministic Dimensional Inference**: Every layer modification triggers instant, deterministic shape inference (`Channels x Height x Width` or `Vector Units`) through the entire network DAG.
- **Bidirectional Synchronization**: The visual graph and the Python source code are two synchronized representations of the same underlying architecture. Changes made on the canvas update the code immediately. Edits written directly in the Python editor can be parsed back into canvas nodes with a single click.
- **Static Guardrails**: A rule engine performs static analysis on the graph before any code runs, flagging rank mismatches, missing flatten operations, unviable pooling windows, and invalid branch mergers.
- **Zero Heavy Abstractions**: Built with pure React 19, native Web APIs (HTML5 Drag and Drop, SVG, Canvas), and standard Vanilla CSS, eliminating heavy external dependencies while delivering instant responsiveness.

---

## 2. End-to-End Workflow

NeuralIDE structures architecture development into four natural engineering phases:

### Phase 1: Structural Composition
Start from an empty canvas or load a canonical baseline (such as ResNet-18, Swin Transformer, or MobileNetV2). Assemble layers using drag-and-drop mechanics or the layer toolbox. Wire residual skip paths or multi-branch concatenations.

### Phase 2: Static Verification and Analytics
As layers are placed, NeuralIDE verifies dimension compatibility, computes parameter counts in millions (M Params), and estimates forward activation memory consumption and theoretical FLOPs. The static analyzer highlights architectural mistakes with actionable error messages.

### Phase 3: Simulated Dry Run (Model Insights)
Switch to the Model Insights tab to inspect the network as a publication-ready hierarchical diagram. Select test samples from benchmark datasets (CIFAR-10, MNIST, Fashion-MNIST) to simulate tensor propagation across stages, visualizing multi-channel spatial activation heatmaps and output probability distributions.

### Phase 4: Code Export and Bidirectional Refinement
Navigate to the PyTorch Code tab to inspect the generated `torch.nn.Module`. Hand-edit training parameters, add custom layers, or modify connections. Use the "Sync to Graph" parser to reflect Python edits back into the visual canvas, or copy the self-contained script directly into your training pipeline.

---

## 3. Core Modules and System Architecture

### 3.1 Architecture Flow Canvas

The primary workspace provides an environment for constructing computation graphs:

- **Layer Library**: Over 50 layer types spanning eight functional categories:
  - *Input and Data*: Input, Normalize, Augmentation, Reshape, Flatten.
  - *Feature Extraction*: Conv2d, Depthwise Conv, Dilated Conv, 1x1 Pointwise Conv, Transposed Conv, MaxPool2d, AveragePool, AdaptiveAvgPool, GlobalAvgPool.
  - *Activations*: ReLU, LeakyReLU, GELU, SiLU / Swish, ELU, SELU, Tanh, Sigmoid, Softplus.
  - *Normalization*: BatchNorm, LayerNorm, GroupNorm, InstanceNorm.
  - *Regularization*: Dropout, Dropout2D, DropPath.
  - *Classifiers*: Dense / Linear, Bilinear, ClassifierHead.
  - *Connections*: Add, Multiply, Concatenate, Residual, Skip, Split, Merge.
  - *Attention and Advanced Blocks*: Self Attention, Multi-Head Attention, Cross Attention, Channel Attention, Spatial Attention, Squeeze-and-Excitation (SE Block), CBAM.
  - *Output*: Softmax.
- **Directed Graph Routing**: Connect layers non-sequentially to create residual shortcuts or branch merges. If layers are reordered or removed, an automated pruning routine strips backward-pointing or orphan connections to maintain valid topological ordering.
- **Interactive Inspector**: Select any layer to modify kernel dimensions, channel counts, stride, dilation, padding, units, or regularization rates. Action buttons permit duplicating, reordering, or removing layers.
- **Activation Microscope**: Inspect mathematical curves for any activation function across the `[-3, 3]` domain with an interactive test slider showing output values in real time.
- **Training Setup Panel**: Configure optimizer selection (AdamW, SGD, RMSprop), learning rate, batch size, and epoch count, which are directly incorporated into the synthesized PyTorch training loop.

### 3.2 Resizable Workspace Subsystem

To accommodate varying screen resolutions and complex inspector panels, the left sidebar is resizable:

- **Dynamic Layout Engine**: Driven by a CSS custom property (`--sidebar-width`) attached to the main CSS grid layout, decoupling layout state from heavy style recalculations.
- **Boundary Clamping**: Clamped between a minimum width of `220px` (preserving input control legibility) and a maximum of `550px` (or viewport-relative constraints preventing middle-canvas collapse).
- **Sticky Grab Pill**: Positioned via `position: sticky; top: 50%` so the handle remains centered within the user viewport regardless of vertical scroll depth.
- **State Persistence**: Current widths are automatically synced to `localStorage` (`neuralide_sidebar_width`).
- **Keyboard and Touch Support**: Full keyboard accessibility via Left and Right arrow keys (+/- 10px step), Home/Enter reset, double-click reset to default (270px), and mobile media query overrides.

### 3.3 Static Design Rule Engine

The shape inference engine continuously runs a rule-based linter against the active graph:

- **Rank Violations**: Detects when spatial operations (convolutions, pooling) receive flat vector inputs, or when dense layers are applied before vector flattening.
- **Spatial Geometry Limits**: Flags kernels or pooling windows whose dimensions exceed the spatial resolution of the current feature map.
- **Normalization Constraints**: Enforces that channel counts are divisible by the group count in `GroupNorm`.
- **Branch Dimension Mismatches**: Validates that tensor shapes entering an `Add` or `Concatenate` node match across channel or spatial dimensions.
- **Output Alignment**: Confirms that the final dense layer matches the target class count of the selected dataset.

### 3.4 Model Insights: Live Dry Run and Feature Maps

The Model Insights workspace presents the model through an academic publication lens:

- **Hierarchical Stage Breakdown**: Automatically partitions the architecture into semantic blocks (Stem / Patch Partition, Stage 1 through Stage 4, Classifier Head) with color-coded tensor pathways.
- **Procedural Activation Heatmaps**: Simulates multi-channel spatial activation heatmaps across each convolutional and pooling stage. This illustrates how low-level edge filters in early layers combine into structural representations and high-level class-discriminative feature maps in deep layers.
- **Dataset Priors**: Choose from CIFAR-10, MNIST, or Fashion-MNIST to seed input dimensions and class vocabularies.
- **Softmax Probability Output**: Displays live simulated class probability distributions and confidence scores corresponding to the selected input category.
- **Computational Profile**: Calculates total trainable parameters, activation memory footprint in megabytes, and theoretical FLOPs/MACs.

### 3.5 Bidirectional PyTorch Code Studio

The Code Studio provides an embedded PyTorch IDE with synchronization:

- **Canvas-to-Code Generation**: Generates clean, idiomatic Python code implementing standard `torch.nn.Module` classes:
  - Formatted declarations in `__init__` with explicit parameter arguments.
  - Forward pass logic that tracks intermediate tensors in a `features` dictionary, allowing arbitrary residual routing.
  - Automated inclusion of standalone helper classes (e.g. `SelfAttentionBlock`) when attention layers are detected.
  - Complete training script including dataset initialization, loss criterion (`CrossEntropyLoss`), optimizer setup, and epoch training loop.
- **Code-to-Canvas Parser (`codeParser.js`)**:
  - Employs a robust Python parser capable of handling diverse coding styles:
    - Arbitrary layer attribute naming (`self.conv1`, `self.fc`, `self.backbone`, etc.).
    - Positional arguments, explicit keyword arguments (`in_channels=3, out_channels=64`), and trailing parameters (`bias=False`).
    - Multi-line layer instantiations.
    - Forward pass extraction supporting both indexed features and direct residual arithmetic (`x = x + ...`).
    - Automatic extraction of training hyperparameters and dataset selections.
- **Live Sync Controller**: A dedicated toggle allows engineers to pause synchronization when performing substantial code refactoring, and re-enable live sync when switching back to canvas design.
- **Dual-Mode Display**:
  - *View Mode*: PrismJS syntax-highlighted presentation with standard text selection and one-click script downloading.
  - *Edit Mode*: Interactive text editor with synchronized line numbers and matched font metrics.

---

## 4. Canonical Research Model Presets

NeuralIDE includes built-in configurations of influential deep learning architectures. Each preset sets up layer definitions, residual paths, recommended datasets, and training configurations:

| Architecture | Reference Paper | Primary Structural Feature | Parameters | Default Dataset | Weights Configuration |
|---|---|---|---|---|---|
| **Swin Transformer (Swin-T)** | Liu et al., ICCV 2021 | Shifted Window Self-Attention (W-MSA / SW-MSA) | ~28.29M | CIFAR-10 | `Swin_T_Weights.DEFAULT` |
| **ResNet-18** | He et al., CVPR 2016 | Identity Residual Skip Connections `F(x) + x` | ~11.69M | CIFAR-10 | `ResNet18_Weights.DEFAULT` |
| **MobileNetV2** | Sandler et al., CVPR 2018 | Inverted Residuals and Linear Bottlenecks | ~3.50M | CIFAR-10 | `MobileNet_V2_Weights.DEFAULT` |
| **Vision Transformer (ViT-Tiny)** | Dosovitskiy et al., ICLR 2021 | Non-overlapping Patch Partitions and Global Attention | ~5.72M | CIFAR-10 | `ViT_B_16_Weights.DEFAULT` |
| **VGG-11** | Simonyan & Zisserman, ICLR 2015 | Deep Homogeneous 3x3 Convolutions | ~132.86M | CIFAR-10 | `VGG11_Weights.DEFAULT` |
| **AlexNet** | Krizhevsky et al., NeurIPS 2012 | Landmark Deep Convolutional Network | ~61.10M | CIFAR-10 | `AlexNet_Weights.DEFAULT` |
| **LeNet-5** | LeCun et al., 1998 | Classical Digit Recognition Convolutional Stack | ~0.06M | MNIST | Scratch Initialization |
| **ConvNet Baseline** | Standard Benchmark | Three-Stage Conv-Pool Feature Extractor | ~0.58M | CIFAR-10 | Scratch Initialization |
| **Multi-Layer Perceptron (MLP)** | Classical Baseline | Dense Feedforward Network with Dropout | ~0.11M | MNIST | Scratch Initialization |

### Pretrained Weights Integration

For architectures supported by `torchvision.models`, NeuralIDE includes a one-click toggle between:
1. **Transfer Learning**: Automatically incorporates pretrained ImageNet-1K weights, freezing or adapting the backbone while reinitializing the classification head.
2. **Scratch Training**: Configures default He (Kaiming) or Xavier uniform initializations for training models from scratch.

---

## 5. Mathematical Formulations

NeuralIDE uses standard deep learning formulas for shape inference and resource estimation:

### Spatial Output Dimension (2D Convolution)

For input spatial dimension $I$, kernel size $K$, padding $P$, dilation $D$, and stride $S$:

$$O = \left\lfloor \frac{I + 2P - D(K - 1) - 1}{S} \right\rfloor + 1$$

### Spatial Output Dimension (Transposed Convolution)

For input spatial dimension $I$, kernel size $K$, padding $P$, output padding $P_{\text{out}}$, and stride $S$:

$$O = (I - 1) \times S - 2P + K + P_{\text{out}}$$

### Parameter Computation

- **Standard Conv2d (with bias)**:
  $$\text{Params} = C_{\text{in}} \times C_{\text{out}} \times K_h \times K_w + C_{\text{out}}$$
- **Depthwise Separable Conv2d**:
  $$\text{Params} = (C_{\text{in}} \times 1 \times K_h \times K_w + C_{\text{in}}) + (C_{\text{in}} \times C_{\text{out}} \times 1 \times 1 + C_{\text{out}})$$
- **Linear Layer**:
  $$\text{Params} = N_{\text{in}} \times N_{\text{out}} + N_{\text{out}}$$
- **Batch Normalization**:
  $$\text{Params} = 2 \times C \quad (\gamma \text{ weight, } \beta \text{ bias})$$
- **Layer Normalization**:
  $$\text{Params} = 2 \times \prod \text{normalized\_shape}$$

### Theoretical Computation (FLOPs / MACs)

For standard 2D convolution:

$$\text{FLOPs} \approx 2 \times H_{\text{out}} \times W_{\text{out}} \times (C_{\text{in}} \times K_h \times K_w) \times C_{\text{out}}$$

---

## 6. Engineering Implementation Details

- **Language and Framework**: JavaScript (ES Modules), React 19.
- **Build Tool**: Vite 8 with Hot Module Replacement.
- **Syntax Engine**: PrismJS integrated with customized token definitions for Python and PyTorch keywords.
- **Design System**: Responsive dark mode implemented using Vanilla CSS tokens, CSS Grid, and custom variables without runtime CSS-in-JS overhead.
- **DOM Event Performance**: Sidebar resizing employs throttled window-level listeners with active cursor locks and selection suppression on `document.body` during drag operations.

---

## 7. Project Structure

```
NeuralIDE/
├── public/                     # Static application assets
├── src/
│   ├── components/
│   │   ├── CodeStudio.jsx      # PyTorch IDE, dual-mode editor, sync controller
│   │   └── PaperDiagram.jsx    # Publication diagram renderer and activation heatmaps
│   ├── data/
│   │   └── modelPresets.js     # Canonical model definitions (Swin, ResNet, ViT, etc.)
│   ├── utils/
│   │   ├── architectureAnalytics.js # Parameter, FLOPs, memory, and heatmap calculation
│   │   └── codeParser.js       # Python AST/regex parser for Code-to-Canvas sync
│   ├── App.jsx                 # Main application controller, workspace state, resizer
│   ├── App.css                 # Application layout, design tokens, and glassmorphism styling
│   ├── index.css               # Global base typography and reset rules
│   └── main.jsx                # Application root entry point
├── package.json                # Manifest and dependencies
├── vite.config.js              # Vite bundler configuration
└── README.md                   # Technical documentation
```

---

## 8. Getting Started

### Prerequisites

- Node.js 18.0.0 or higher
- npm, pnpm, or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mukut02/NeuralIDE.git
   cd NeuralIDE
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Launch the development server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.

4. Build production bundle:
   ```bash
   npm run build
   ```
   The compiled distribution files will be located in the `dist/` directory.

---

## 9. License

This project is open-source software licensed under the MIT License.
