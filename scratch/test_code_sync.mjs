import { parseCodeToArchitecture } from "../src/utils/codeParser.js";
import { MODEL_PRESETS } from "../src/data/modelPresets.js";

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
  GELU: { label: "GELU", group: "Activations", params: {} }, SiLU: { label: "SiLU / Swish", group: "Activations", params: {} }, ELU: { label: "ELU", group: "Activations", params: {} }, SELU: { label: "SELU", group: "Activations", params: {} }, Tanh: { label: "Tanh", group: "Activations", params: {} }, Sigmoid: { label: "Sigmoid", group: "Activations", params: {} }, Softplus: { label: "Softplus", group: "Activations", params: {} },
  MaxPool2d: { label: "Max Pool", group: "Feature extraction", params: { kernel: 2 } },
  AveragePool: { label: "Average Pool", group: "Feature extraction", params: { kernel: 2 } },
  AdaptiveAvgPool: { label: "Adaptive Avg Pool", group: "Feature extraction", params: { size: 1 } },
  GlobalAvgPool: { label: "Global Avg Pool", group: "Feature extraction", params: {} },
  BatchNorm: { label: "BatchNorm", group: "Normalization", params: {} }, LayerNorm: { label: "LayerNorm", group: "Normalization", params: {} }, GroupNorm: { label: "GroupNorm", group: "Normalization", params: { groups: 8 } }, InstanceNorm: { label: "InstanceNorm", group: "Normalization", params: {} },
  Dropout: { label: "Dropout", group: "Regularization", params: { rate: 0.25 } }, Dropout2D: { label: "Dropout2D", group: "Regularization", params: { rate: 0.25 } }, DropPath: { label: "DropPath", group: "Regularization", params: { rate: 0.1 } },
  Dense: { label: "Dense", group: "Classifier", params: { units: 128 } },
  Linear: { label: "Linear", group: "Classifier", params: { units: 128 } }, Bilinear: { label: "Bilinear", group: "Classifier", params: { units: 128 } }, ClassifierHead: { label: "Classifier Head", group: "Classifier", params: { units: 10 } },
  Add: { label: "Add", group: "Connections", params: {} }, Multiply: { label: "Multiply", group: "Connections", params: {} }, Concatenate: { label: "Concatenate", group: "Connections", params: {} }, Residual: { label: "Residual Connection", group: "Connections", params: {} }, Skip: { label: "Skip Connection", group: "Connections", params: {} }, Split: { label: "Split", group: "Connections", params: {} }, Merge: { label: "Merge", group: "Connections", params: {} },
  SelfAttention: { label: "Self Attention", group: "Attention / advanced", params: { heads: 4 } }, MultiHeadAttention: { label: "Multi-Head Attention", group: "Attention / advanced", params: { heads: 8 } }, CrossAttention: { label: "Cross Attention", group: "Attention / advanced", params: { heads: 8 } }, ChannelAttention: { label: "Channel Attention", group: "Attention / advanced", params: {} }, SpatialAttention: { label: "Spatial Attention", group: "Attention / advanced", params: {} }, SEBlock: { label: "SE Block", group: "Attention / advanced", params: { reduction: 16 } }, CBAM: { label: "CBAM", group: "Attention / advanced", params: { reduction: 16 } },
  Softmax: { label: "Softmax", group: "Output", params: {} },
  LogSoftmax: { label: "LogSoftmax", group: "Output", params: {} }, RegressionHead: { label: "Regression Head", group: "Output", params: { units: 1 } }, ClassificationHead: { label: "Classification Head", group: "Output", params: { units: 10 } },
};

const INITIAL_LAYERS = [
  { id: 1, type: "Conv2d", params: { filters: 32, kernel: 3, padding: 1, stride: 1 } },
  { id: 2, type: "ReLU", params: {} },
  { id: 3, type: "MaxPool2d", params: { kernel: 2 } },
  { id: 4, type: "Flatten", params: {} },
  { id: 5, type: "Linear", params: { units: 128 } },
  { id: 6, type: "ReLU", params: {} },
  { id: 7, type: "Linear", params: { units: 10 } },
];

function inferArchitecture(layers, dataset, connections = []) {
  let shape = { kind: "image", c: dataset.channels, h: dataset.size, w: dataset.size };
  const result = [];
  layers.forEach((layer) => {
    const input = { ...shape };
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
  const hasSelfAttention = layers.some((layer) => layer.type === "SelfAttention" || (layer.type === "MultiHeadAttention" && !byTarget.has(layer.id)));
  const hasCrossAttention = layers.some((layer) => layer.type === "CrossAttention" || (layer.type === "MultiHeadAttention" && byTarget.has(layer.id)));
  const architecture = inferArchitecture(layers, dataset, connections).result;

  layers.forEach((layer, index) => {
    const p = layer.params;
    const name = `layer_${index + 1}`;
    const source = byTarget.get(layer.id);
    const branch = source ? `features[${layers.findIndex((item) => item.id === source.from) + 1}]` : null;
    const attentionChannels = architecture[index]?.input?.kind === "image" ? architecture[index].input.c : 1;
    forward.push(`# ${String(index + 1).padStart(2, "0")} · ${LAYER_LIBRARY[layer.type].label}`);

    if (layer.type === "Conv2d") { definitions.push(`self.${name} = nn.Conv2d(${channels}, ${p.filters}, ${p.kernel}, stride=${p.stride ?? 1}, padding=${p.padding})`); forward.push(`x = self.${name}(x)`); channels = p.filters; }
    else if (layer.type === "DepthwiseConv") { definitions.push(`self.${name} = nn.Conv2d(${channels}, ${channels * p.multiplier}, ${p.kernel}, padding=${p.padding}, groups=${channels})`); forward.push(`x = self.${name}(x)`); channels *= p.multiplier; }
    else if (layer.type === "DilatedConv") { definitions.push(`self.${name} = nn.Conv2d(${channels}, ${p.filters}, ${p.kernel}, padding=${p.padding}, dilation=${p.dilation})`); forward.push(`x = self.${name}(x)`); channels = p.filters; }
    else if (layer.type === "PointwiseConv") { definitions.push(`self.${name} = nn.Conv2d(${channels}, ${p.filters}, 1)`); forward.push(`x = self.${name}(x)`); channels = p.filters; }
    else if (layer.type === "TransposedConv") { definitions.push(`self.${name} = nn.ConvTranspose2d(${channels}, ${p.filters}, ${p.kernel}, stride=${p.stride}, padding=${p.padding})`); forward.push(`x = self.${name}(x)`); channels = p.filters; }
    else if (activationMap[layer.type]) { definitions.push(`self.${name} = nn.${activationMap[layer.type]}`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "MaxPool2d") { definitions.push(`self.${name} = nn.MaxPool2d(${p.kernel})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "AveragePool") { definitions.push(`self.${name} = nn.AvgPool2d(${p.kernel})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "AdaptiveAvgPool") { definitions.push(`self.${name} = nn.AdaptiveAvgPool2d((${p.size}, ${p.size}))`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "GlobalAvgPool") { definitions.push(`self.${name} = nn.AdaptiveAvgPool2d(1)`); forward.push(`x = torch.flatten(self.${name}(x), 1)`); }
    else if (layer.type === "Flatten") forward.push("x = torch.flatten(x, 1)");
    else if (["Linear", "Dense", "Bilinear", "ClassifierHead", "ClassificationHead", "RegressionHead"].includes(layer.type)) { definitions.push(`self.${name} = nn.LazyLinear(${p.units})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "Dropout") { definitions.push(`self.${name} = nn.Dropout(${p.rate})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "Dropout2D") { definitions.push(`self.${name} = nn.Dropout2d(${p.rate})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "DropPath") { definitions.push(`self.${name} = nn.Dropout(${p.rate})  # DropPath approximated as Dropout`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "BatchNorm") { definitions.push(`self.${name} = nn.BatchNorm2d(${channels})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "InstanceNorm") { definitions.push(`self.${name} = nn.InstanceNorm2d(${channels})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "GroupNorm") { definitions.push(`self.${name} = nn.GroupNorm(${p.groups}, ${channels})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "LayerNorm") forward.push("x = F.layer_norm(x, x.shape[1:])");
    else if (layer.type === "Reshape") forward.push(`x = x.reshape(x.shape[0], ${p.channels}, ${p.size}, ${p.size})`);
    else if (["Normalize", "Augmentation", "Input"].includes(layer.type)) forward.push(`# ${LAYER_LIBRARY[layer.type].label} is applied in the dataset transform`);
    else if (["Add", "Residual", "Merge"].includes(layer.type)) forward.push(branch ? `x = x + ${branch}` : "# attach a branch to complete this merge");
    else if (layer.type === "Multiply") forward.push(branch ? `x = x * ${branch}` : "# attach a branch to multiply tensors");
    else if (layer.type === "Concatenate") forward.push(branch ? `x = torch.cat((x, ${branch}), dim=1)` : "# attach a branch to concatenate tensors");
    else if (layer.type === "Skip") forward.push(branch ? `x = ${branch}  # skip route` : "# attach a source for this skip route");
    else if (layer.type === "Split") forward.push("skip = x  # preserve this tensor for a later merge");
    else if (["SelfAttention", "MultiHeadAttention"].includes(layer.type)) {
      if (layer.type === "MultiHeadAttention" && branch) {
        definitions.push(`self.${name} = CrossAttentionBlock(channels=${attentionChannels}, heads=${p.heads})`);
        forward.push(`x = self.${name}(x, ${branch})`);
      } else {
        definitions.push(`self.${name} = SelfAttentionBlock(channels=${attentionChannels}, heads=${p.heads})`);
        forward.push(`x = self.${name}(x)`);
      }
    }
    else if (layer.type === "CrossAttention") {
      definitions.push(`self.${name} = CrossAttentionBlock(channels=${attentionChannels}, heads=${p.heads})`);
      forward.push(branch ? `x = self.${name}(x, ${branch})` : "# attach an earlier context layer before running cross attention");
    }
    else if (["ChannelAttention", "SpatialAttention", "SEBlock", "CBAM"].includes(layer.type)) forward.push(`# ${LAYER_LIBRARY[layer.type].label} needs a task-specific image attention block`);
    else if (layer.type === "Softmax") forward.push("x = torch.softmax(x, dim=1)");
    else if (layer.type === "LogSoftmax") forward.push("x = torch.log_softmax(x, dim=1)");
    forward.push(`features[${index + 1}] = x`);
  });

  const optimizerCall = `torch.optim.AdamW(model.parameters(), lr=${training.learningRate})`;
  const isPretrained = Boolean(usePretrainedWeights && currentPreset?.torchvisionModel);

  if (isPretrained) {
    return `import torch
import torch.nn as nn
# Pretrained ${currentPreset.name}
class CustomArchitectureBlueprint(nn.Module):
    def __init__(self):
        super().__init__()
        ${definitions.join("\n        ") || "pass"}

    def forward(self, x):
        features = {}
        ${forward.join("\n        ")}
        return x
`;
  }

  return `import torch
import torch.nn as nn
from torchvision import datasets
train_set = datasets.MNIST(root="./data")
# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=${training.learningRate})
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

// TEST 1: Initial Layers
const code1 = makeCode(INITIAL_LAYERS, DATASETS.MNIST, [], { optimizer: "AdamW", learningRate: 0.001, batchSize: 128, epochs: 10 });
console.log("=== Generated Code (First 15 lines) ===");
console.log(code1.split("\n").slice(0, 15).join("\n"));
const parsed1 = parseCodeToArchitecture(code1, DATASETS, LAYER_LIBRARY);
console.log("\n=== Parsed Result ===");
console.log("Dataset:", parsed1?.datasetName);
console.log("Training:", parsed1?.training);
console.log("Layers count:", parsed1?.layers?.length);
console.log("Layers parsed:", parsed1?.layers);

// TEST 2: Presets
for (const preset of MODEL_PRESETS) {
  const code = makeCode(preset.layers, DATASETS[preset.recommendedDataset] || DATASETS.MNIST, preset.connections || [], { optimizer: "AdamW", learningRate: 0.001, batchSize: 128, epochs: 10 }, true, preset);
  const parsed = parseCodeToArchitecture(code, DATASETS, LAYER_LIBRARY);
  console.log(`Preset [${preset.id}] (${preset.layers.length} layers) -> Parsed: ${parsed?.layers?.length} layers`);
}
