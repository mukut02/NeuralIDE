import { parseCodeToArchitecture } from "../src/utils/codeParser.js";

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

let testsPassed = 0;
let testsFailed = 0;

function assert(condition, name) {
  if (condition) {
    console.log(`  PASS: ${name}`);
    testsPassed++;
  } else {
    console.error(`  FAIL: ${name}`);
    testsFailed++;
  }
}

console.log("=== RUNNING CODE PARSER & TWO-WAY SYNC UNIT TESTS ===\n");

// TEST 1: User edits filters and kernel size in Conv2d
{
  console.log("[Test 1] Parse custom Conv2d filter and kernel edits");
  const code = `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Conv2d(1, 64, 5, stride=2, padding=2)
    def forward(self, x):
        # 01 · Convolution
        x = self.layer_1(x)
        features[1] = x
        return x
  `;
  const parsed = parseCodeToArchitecture(code, DATASETS, LAYER_LIBRARY);
  assert(parsed !== null, "Result is not null");
  assert(parsed?.layers?.length === 1, "Has 1 layer");
  assert(parsed?.layers?.[0]?.params?.filters === 64, "Filters updated to 64");
  assert(parsed?.layers?.[0]?.params?.kernel === 5, "Kernel updated to 5");
  assert(parsed?.layers?.[0]?.params?.stride === 2, "Stride updated to 2");
  assert(parsed?.layers?.[0]?.params?.padding === 2, "Padding updated to 2");
}

// TEST 2: User changes hyperparameters (learningRate, batch_size, epochs, optimizer)
{
  console.log("\n[Test 2] Parse training hyperparameters");
  const code = `
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9
)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
for epoch in range(25):
    pass

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.LazyLinear(64)
    def forward(self, x):
        # 01 · Linear
        x = self.layer_1(x)
        return x
  `;
  const parsed = parseCodeToArchitecture(code, DATASETS, LAYER_LIBRARY);
  assert(parsed?.training?.learningRate === 0.005, "learningRate parsed as 0.005");
  assert(parsed?.training?.batchSize === 64, "batchSize parsed as 64");
  assert(parsed?.training?.epochs === 25, "epochs parsed as 25");
  assert(parsed?.training?.optimizer === "SGD", "optimizer parsed as SGD");
}

// TEST 3: User changes Dataset to Fashion-MNIST or CIFAR-10
{
  console.log("\n[Test 3] Parse dataset changes");
  const codeFashion = `train_set = datasets.FashionMNIST(root="./data")\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layer_1 = nn.ReLU()\n    def forward(self, x):\n        # 01 · ReLU\n        x = self.layer_1(x)\n        return x`;
  const parsedFashion = parseCodeToArchitecture(codeFashion, DATASETS, LAYER_LIBRARY);
  assert(parsedFashion?.datasetName === "Fashion-MNIST", "Dataset identified as Fashion-MNIST");

  const codeCIFAR = `train_set = datasets.CIFAR10(root="./data")\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layer_1 = nn.ReLU()\n    def forward(self, x):\n        # 01 · ReLU\n        x = self.layer_1(x)\n        return x`;
  const parsedCIFAR = parseCodeToArchitecture(codeCIFAR, DATASETS, LAYER_LIBRARY);
  assert(parsedCIFAR?.datasetName === "CIFAR-10", "Dataset identified as CIFAR-10");
}

// TEST 4: User changes Activation (e.g. ReLU -> GELU)
{
  console.log("\n[Test 4] Parse activation changes");
  const code = `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.GELU()
    def forward(self, x):
        # 01 · GELU
        x = self.layer_1(x)
        return x
  `;
  const parsed = parseCodeToArchitecture(code, DATASETS, LAYER_LIBRARY);
  assert(parsed?.layers?.[0]?.type === "GELU", "Layer parsed as GELU");
}

// TEST 5: User adds a residual skip connection (Add)
{
  console.log("\n[Test 5] Parse residual connection (Add)");
  const code = `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.layer_2 = nn.ReLU()
        self.layer_3 = nn.Conv2d(32, 32, 3, padding=1)
    def forward(self, x):
        features = {}
        # 01 · Convolution
        x = self.layer_1(x)
        features[1] = x
        # 02 · ReLU
        x = self.layer_2(x)
        features[2] = x
        # 03 · Add
        x = x + features[1]
        features[3] = x
        return x
  `;
  const parsed = parseCodeToArchitecture(code, DATASETS, LAYER_LIBRARY);
  assert(parsed?.layers?.length === 3, "Has 3 layers");
  assert(parsed?.layers?.[2]?.type === "Add", "Layer 3 is Add");
  assert(parsed?.connections?.length === 1, "Connection detected");
  assert(parsed?.connections?.[0]?.from === 1 && parsed?.connections?.[0]?.to === 3, "Connection from 1 to 3");
}

// TEST 6: User writes code without forward comments (parse directly from __init__)
{
  console.log("\n[Test 6] Parse from __init__ when user deletes step comments");
  const code = `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.layer_2 = nn.ReLU()
        self.layer_3 = nn.MaxPool2d(2)
        self.layer_4 = nn.Linear(128, 10)

    def forward(self, x):
        return self.layer_4(self.layer_3(self.layer_2(self.layer_1(x))))
  `;
  const parsed = parseCodeToArchitecture(code, DATASETS, LAYER_LIBRARY);
  assert(parsed !== null, "Fallback parse succeeded");
  assert(parsed?.layers?.length === 4, `Parsed 4 layers (got ${parsed?.layers?.length})`);
  assert(parsed?.layers?.[0]?.type === "Conv2d", "Layer 1 is Conv2d");
  assert(parsed?.layers?.[1]?.type === "ReLU", "Layer 2 is ReLU");
  assert(parsed?.layers?.[2]?.type === "MaxPool2d", "Layer 3 is MaxPool2d");
  assert(parsed?.layers?.[3]?.type === "Linear", "Layer 4 is Linear");
}

// TEST 7: User writes custom layer variable names (self.conv1, self.fc)
{
  console.log("\n[Test 7] Parse arbitrary layer variable names (self.conv1, self.fc)");
  const code = `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.fc1(x)
        return x
  `;
  const parsed = parseCodeToArchitecture(code, DATASETS, LAYER_LIBRARY);
  assert(parsed !== null && parsed.layers?.length === 4, "Arbitrary variable names parsed correctly (4 layers)");
  assert(parsed?.layers?.[0]?.type === "Conv2d", "First is Conv2d");
  assert(parsed?.layers?.[3]?.type === "Linear", "Last is Linear");
}

// TEST 8: Multiline definitions and keyword arguments (in_channels, out_channels, kernel_size)
{
  console.log("\n[Test 8] Parse multiline definitions and keyword arguments");
  const code = `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2
        )
        self.layer_2 = nn.Linear(
            in_features=512,
            out_features=256,
            bias=False
        )

    def forward(self, x):
        # 01 · Convolution
        x = self.layer_1(x)
        # 02 · Linear
        x = self.layer_2(x)
        return x
  `;
  const parsed = parseCodeToArchitecture(code, DATASETS, LAYER_LIBRARY);
  assert(parsed !== null, "Multiline parsed");
  assert(parsed?.layers?.[0]?.params?.filters === 64, "Conv2d out_channels=64 parsed");
  assert(parsed?.layers?.[0]?.params?.kernel === 5, "Conv2d kernel_size=5 parsed");
  assert(parsed?.layers?.[0]?.params?.stride === 2, "Conv2d stride=2 parsed");
  assert(parsed?.layers?.[1]?.params?.units === 256, "Linear out_features=256 parsed with bias=False");
}

// TEST 9: Dropout with p keyword and MaxPool with kernel_size keyword
{
  console.log("\n[Test 9] Parse Dropout rate and MaxPool kernel with keyword args");
  const code = `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.MaxPool2d(kernel_size=3)
        self.layer_2 = nn.Dropout(p=0.45)
    def forward(self, x):
        # 01 · Max Pool
        x = self.layer_1(x)
        # 02 · Dropout
        x = self.layer_2(x)
        return x
  `;
  const parsed = parseCodeToArchitecture(code, DATASETS, LAYER_LIBRARY);
  assert(parsed?.layers?.[0]?.params?.kernel === 3, "MaxPool kernel_size=3 parsed");
  assert(parsed?.layers?.[1]?.params?.rate === 0.45, "Dropout p=0.45 parsed");
}

console.log(`\n=== SUMMARY: ${testsPassed} passed, ${testsFailed} failed ===`);
