import { parseCodeToArchitecture } from "../src/utils/codeParser.js";
import { MODEL_PRESETS } from "../src/data/modelPresets.js";
import {
  calculateLayerParams,
  calculateLayerFlops,
  calculateTensorBytes,
  formatParams,
  formatBytes,
  formatFlops,
} from "../src/utils/architectureAnalytics.js";

const DATASETS = {
  MNIST: { name: "MNIST", channels: 1, size: 28, classes: 10, description: "Handwritten digits" },
  "Fashion-MNIST": { name: "Fashion-MNIST", channels: 1, size: 28, classes: 10, description: "Clothing images" },
  "CIFAR-10": { name: "CIFAR-10", channels: 3, size: 32, classes: 10, description: "Colour object images" },
};

const LAYER_LIBRARY = {
  Input: { label: "Input", group: "Input & data", params: {} },
  Normalize: { label: "Normalize", group: "Input & data", params: {} },
  Augmentation: { label: "Augmentation", group: "Input & data", params: {} },
  Reshape: { label: "Reshape", group: "Input & data", params: { channels: 1, size: 28 } },
  Flatten: { label: "Flatten", group: "Input & data", params: {} },
  Conv2d: { label: "Convolution", group: "Feature extraction", params: { filters: 32, kernel: 3, padding: 1, stride: 1 } },
  DepthwiseConv: { label: "Depthwise Conv", group: "Feature extraction", params: { multiplier: 1, kernel: 3, padding: 1 } },
  DilatedConv: { label: "Dilated Conv", group: "Feature extraction", params: { filters: 32, kernel: 3, dilation: 2, padding: 2 } },
  PointwiseConv: { label: "1×1 Convolution", group: "Feature extraction", params: { filters: 32 } },
  TransposedConv: { label: "Transposed Conv", group: "Feature extraction", params: { filters: 32, kernel: 2, stride: 2, padding: 0 } },
  ReLU: { label: "ReLU", group: "Activations", params: {} },
  LeakyReLU: { label: "Leaky ReLU", group: "Activations", params: { slope: 0.1 } },
  GELU: { label: "GELU", group: "Activations", params: {} },
  SiLU: { label: "SiLU / Swish", group: "Activations", params: {} },
  ELU: { label: "ELU", group: "Activations", params: {} },
  SELU: { label: "SELU", group: "Activations", params: {} },
  Tanh: { label: "Tanh", group: "Activations", params: {} },
  Sigmoid: { label: "Sigmoid", group: "Activations", params: {} },
  Softplus: { label: "Softplus", group: "Activations", params: {} },
  MaxPool2d: { label: "Max Pool", group: "Feature extraction", params: { kernel: 2 } },
  AveragePool: { label: "Average Pool", group: "Feature extraction", params: { kernel: 2 } },
  AdaptiveAvgPool: { label: "Adaptive Avg Pool", group: "Feature extraction", params: { size: 1 } },
  GlobalAvgPool: { label: "Global Avg Pool", group: "Feature extraction", params: {} },
  BatchNorm: { label: "BatchNorm", group: "Normalization", params: {} },
  LayerNorm: { label: "LayerNorm", group: "Normalization", params: {} },
  GroupNorm: { label: "GroupNorm", group: "Normalization", params: { groups: 8 } },
  InstanceNorm: { label: "InstanceNorm", group: "Normalization", params: {} },
  Dropout: { label: "Dropout", group: "Regularization", params: { rate: 0.25 } },
  Dropout2D: { label: "Dropout2D", group: "Regularization", params: { rate: 0.25 } },
  DropPath: { label: "DropPath", group: "Regularization", params: { rate: 0.1 } },
  Dense: { label: "Dense", group: "Classifier", params: { units: 128 } },
  Linear: { label: "Linear", group: "Classifier", params: { units: 128 } },
  Bilinear: { label: "Bilinear", group: "Classifier", params: { units: 128 } },
  ClassifierHead: { label: "Classifier Head", group: "Classifier", params: { units: 10 } },
  Add: { label: "Add", group: "Connections", params: {} },
  Multiply: { label: "Multiply", group: "Connections", params: {} },
  Concatenate: { label: "Concatenate", group: "Connections", params: {} },
  Residual: { label: "Residual Connection", group: "Connections", params: {} },
  Skip: { label: "Skip Connection", group: "Connections", params: {} },
  Split: { label: "Split", group: "Connections", params: {} },
  Merge: { label: "Merge", group: "Connections", params: {} },
  SelfAttention: { label: "Self Attention", group: "Attention / advanced", params: { heads: 4 } },
  MultiHeadAttention: { label: "Multi-Head Attention", group: "Attention / advanced", params: { heads: 8 } },
  CrossAttention: { label: "Cross Attention", group: "Attention / advanced", params: { heads: 8 } },
  ChannelAttention: { label: "Channel Attention", group: "Attention / advanced", params: {} },
  SpatialAttention: { label: "Spatial Attention", group: "Attention / advanced", params: {} },
  SEBlock: { label: "SE Block", group: "Attention / advanced", params: { reduction: 16 } },
  CBAM: { label: "CBAM", group: "Attention / advanced", params: { reduction: 16 } },
  Softmax: { label: "Softmax", group: "Output", params: {} },
  LogSoftmax: { label: "LogSoftmax", group: "Output", params: {} },
  RegressionHead: { label: "Regression Head", group: "Output", params: { units: 1 } },
  ClassificationHead: { label: "Classification Head", group: "Output", params: { units: 10 } },
};

function inferArchitecture(layers, dataset, connections = []) {
  let shape = { kind: "image", c: dataset.channels, h: dataset.size, w: dataset.size };
  const result = [];
  layers.forEach((layer) => {
    const input = { ...shape };
    if (layer.type === "Conv2d") {
      const p = layer.params;
      const span = Math.floor((shape.h + 2 * (p.padding || 0) - (p.kernel || 3) + 1) / (p.stride || 1));
      shape = { kind: "image", c: p.filters, h: Math.max(1, span), w: Math.max(1, span) };
    } else if (layer.type === "Flatten") {
      shape = { kind: "vector", n: shape.c * shape.h * shape.w };
    } else if (layer.type === "Linear") {
      shape = { kind: "vector", n: layer.params.units };
    }
    result.push({ ...layer, input, output: { ...shape } });
  });
  return { result };
}

function makeCode(layers, dataset, connections, training, usePretrainedWeights = false, currentPreset = null) {
  let channels = dataset.channels;
  const definitions = [];
  const forward = [];
  const byTarget = new Map(connections.map((connection) => [connection.to, connection]));
  const activationMap = { ReLU: "ReLU()", LeakyReLU: "LeakyReLU(0.1)", GELU: "GELU()", SiLU: "SiLU()", ELU: "ELU()", SELU: "SELU()", Tanh: "Tanh()", Sigmoid: "Sigmoid()", Softplus: "Softplus()" };

  layers.forEach((layer, index) => {
    const p = layer.params;
    const name = `layer_${index + 1}`;
    const source = byTarget.get(layer.id);
    const branch = source ? `features[${layers.findIndex((item) => item.id === source.from) + 1}]` : null;
    forward.push(`# ${String(index + 1).padStart(2, "0")} · ${LAYER_LIBRARY[layer.type]?.label || layer.type}`);

    if (layer.type === "Conv2d") { definitions.push(`self.${name} = nn.Conv2d(${channels}, ${p.filters}, ${p.kernel}, stride=${p.stride ?? 1}, padding=${p.padding})`); forward.push(`x = self.${name}(x)`); channels = p.filters; }
    else if (activationMap[layer.type]) { definitions.push(`self.${name} = nn.${activationMap[layer.type]}`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "MaxPool2d") { definitions.push(`self.${name} = nn.MaxPool2d(${p.kernel})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "Flatten") forward.push("x = torch.flatten(x, 1)");
    else if (["Linear", "Dense"].includes(layer.type)) { definitions.push(`self.${name} = nn.LazyLinear(${p.units})`); forward.push(`x = self.${name}(x)`); }
    else if (["Add", "Residual"].includes(layer.type)) forward.push(branch ? `x = x + ${branch}` : "# attach a branch to complete this merge");
    forward.push(`features[${index + 1}] = x`);
  });

  const optimizerCall = `torch.optim.${training.optimizer || "AdamW"}(model.parameters(), lr=${training.learningRate})`;
  const isPretrained = Boolean(usePretrainedWeights && currentPreset?.torchvisionModel);

  if (isPretrained) {
    return `import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import ${currentPreset.torchvisionModel}
# Dataset: ${dataset.name}
train_set = datasets.${dataset.name === "CIFAR-10" ? "CIFAR10" : dataset.name.replace("-", "")}(root="./data")
model = ${currentPreset.torchvisionModel}(weights=None)
class CustomArchitectureBlueprint(nn.Module):
    def __init__(self):
        super().__init__()
        ${definitions.join("\n        ") || "pass"}
    def forward(self, x):
        features = {}
        ${forward.join("\n        ")}
        return x
optimizer = ${optimizerCall}
for epoch in range(${training.epochs}):
    pass
`;
  }

  return `import torch
import torch.nn as nn
from torchvision import datasets
# Dataset: ${dataset.name}
train_set = datasets.${dataset.name === "CIFAR-10" ? "CIFAR10" : dataset.name.replace("-", "")}(root="./data")
optimizer = ${optimizerCall}
batch_size = ${training.batchSize}
for epoch in range(${training.epochs}):
    pass

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ${definitions.join("\n        ") || "pass"}

    def forward(self, x):
        features = {}
        ${forward.join("\n        ")}
        return x
`;
}

let passed = 0;
let failed = 0;

function check(testName, condition) {
  if (condition) {
    console.log(`[PASS] ${testName}`);
    passed++;
  } else {
    console.error(`[FAIL] ${testName}`);
    failed++;
  }
}

console.log("=================================================");
console.log("TEST SUITE: BIDIRECTIONAL SYNC & CODE VISIBILITY");
console.log("=================================================\n");

// 1. Initial State Consistency
console.log("1. Testing Initial State Consistency:");
const swinPreset = MODEL_PRESETS.find((p) => p.id === "swin_tiny");
check("Swin-T preset exists", Boolean(swinPreset));
check("Swin-T recommended dataset is CIFAR-10", swinPreset.recommendedDataset === "CIFAR-10");
check("Swin-T canonical params format in Millions", formatParams(swinPreset.canonicalParams).includes("M"));
console.log(`   Swin-T Params: ${formatParams(swinPreset.canonicalParams)}`);

// 2. Visual Architecture -> Code Generation
console.log("\n2. Testing Architecture -> Code Reflection:");
const testLayers = [
  { id: 1, type: "Conv2d", params: { filters: 64, kernel: 5, padding: 2, stride: 2 } },
  { id: 2, type: "ReLU", params: {} },
  { id: 3, type: "Linear", params: { units: 10 } },
];
const genCode = makeCode(testLayers, DATASETS["CIFAR-10"], [], { optimizer: "SGD", learningRate: 0.05, batchSize: 32, epochs: 20 });
check("Generated code reflects Conv2d filters 64", genCode.includes("64, 5, stride=2, padding=2"));
check("Generated code reflects SGD optimizer", genCode.includes("torch.optim.SGD"));
check("Generated code reflects lr 0.05", genCode.includes("lr=0.05"));
check("Generated code reflects CIFAR-10 dataset", genCode.includes("datasets.CIFAR10"));

// 3. Code -> Visual Architecture Sync
console.log("\n3. Testing Code -> Architecture Parsing:");
const editedCode = `
import torch
import torch.nn as nn
from torchvision import datasets

train_set = datasets.FashionMNIST(root="./data")
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.002)
batch_size = 64
for epoch in range(15):
    pass

class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(in_features=9408, out_features=120, bias=False)
        self.out = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.out(x)
        return x
`;
const parsed = parseCodeToArchitecture(editedCode, DATASETS, LAYER_LIBRARY);
check("Parsed dataset changed to Fashion-MNIST", parsed.datasetName === "Fashion-MNIST");
check("Parsed optimizer changed to RMSprop", parsed.training.optimizer === "RMSprop");
check("Parsed learningRate changed to 0.002", parsed.training.learningRate === 0.002);
check("Parsed batchSize changed to 64", parsed.training.batchSize === 64);
check("Parsed epochs changed to 15", parsed.training.epochs === 15);
check("Parsed layer count is 6", parsed.layers.length === 6);
check("First layer is Conv2d with 48 filters", parsed.layers[0].type === "Conv2d" && parsed.layers[0].params.filters === 48);
check("Second layer is GELU", parsed.layers[1].type === "GELU");
check("Third layer is MaxPool2d", parsed.layers[2].type === "MaxPool2d");
check("Fourth layer is Flatten", parsed.layers[3].type === "Flatten");
check("Fifth layer is Linear with 120 units", parsed.layers[4].type === "Linear" && parsed.layers[4].params.units === 120);
check("Sixth layer is Linear with 10 units", parsed.layers[5].type === "Linear" && parsed.layers[5].params.units === 10);

// 4. Presets Roundtrip Test
console.log("\n4. Testing Presets Roundtrip Sync:");
for (const preset of MODEL_PRESETS) {
  const code = makeCode(preset.layers, DATASETS[preset.recommendedDataset] || DATASETS["CIFAR-10"], preset.connections || [], preset.training || {}, preset.pretrainedWeights, preset);
  const pResult = parseCodeToArchitecture(code, DATASETS, LAYER_LIBRARY);
  check(`Preset [${preset.id}] parsed successfully (${pResult?.layers?.length} layers)`, Boolean(pResult && pResult.layers.length > 0));
}

console.log(`\n=================================================`);
console.log(`TOTAL: ${passed} PASSED, ${failed} FAILED`);
console.log(`=================================================`);
