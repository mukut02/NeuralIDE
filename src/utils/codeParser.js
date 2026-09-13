// Parser to extract neural network architecture, dataset, and training configuration
// from PyTorch code, enabling two-way synchronization between Code and Canvas.

export function parseCodeToArchitecture(code, currentDatasets = {}, layerLibrary = {}) {
  if (!code || typeof code !== "string") return null;

  const result = {
    datasetName: null,
    training: {},
    layers: [],
    connections: [],
  };

  try {
    // 1. Parse Dataset
    if (/datasets\.CIFAR10/i.test(code) || /CIFAR-10/i.test(code)) {
      result.datasetName = "CIFAR-10";
    } else if (/datasets\.FashionMNIST/i.test(code) || /Fashion-MNIST/i.test(code)) {
      result.datasetName = "Fashion-MNIST";
    } else if (/datasets\.MNIST/i.test(code) || /MNIST/i.test(code)) {
      result.datasetName = "MNIST";
    }

    // 2. Parse Training Hyperparameters
    const lrMatch = code.match(/lr\s*=\s*([0-9.eE+-]+)/);
    if (lrMatch) {
      const lr = parseFloat(lrMatch[1]);
      if (!isNaN(lr) && lr > 0) result.training.learningRate = lr;
    }

    const batchMatch = code.match(/batch_size\s*=\s*(\d+)/);
    if (batchMatch) {
      const bs = parseInt(batchMatch[1], 10);
      if (!isNaN(bs) && bs > 0) result.training.batchSize = bs;
    }

    const epochMatch = code.match(/range\((\d+)\)/);
    if (epochMatch) {
      const ep = parseInt(epochMatch[1], 10);
      if (!isNaN(ep) && ep > 0) result.training.epochs = ep;
    }

    const optMatch = code.match(/torch\.optim\.(AdamW|SGD|RMSprop|Adam)/);
    if (optMatch) {
      result.training.optimizer = optMatch[1] === "Adam" ? "AdamW" : optMatch[1];
    }

    // 3. Parse Layer Definitions in __init__
    // Matches self.conv1 = nn.Conv2d(...) or self.layer_1 = ... across multiple lines
    const layerDefs = new Map();
    const orderedDefs = [];
    const defRegex = /self\.([a-zA-Z0-9_]+)\s*=\s*(?:nn\.)?([a-zA-Z0-9_]+)\s*\(([\s\S]*?)\)(?=\s*(?:#|self\.|\n\s*def|\n\s*return|\n\s*$))/g;
    let defMatch;

    while ((defMatch = defRegex.exec(code)) !== null) {
      const varName = defMatch[1];
      const typeName = defMatch[2];
      const argsStr = defMatch[3].trim();
      const defObj = { varName, typeName, argsStr };
      layerDefs.set(varName, defObj);
      orderedDefs.push(defObj);
    }

    // Helper to resolve layer type and parameters from definition
    function resolveDef(def) {
      if (!def) return { layerType: "Linear", params: {} };
      const { typeName, argsStr } = def;
      let layerType = "Linear";
      let params = {};

      if (typeName === "Conv2d") {
        const outChannelsM = argsStr.match(/(?:out_channels|filters)\s*=\s*(\d+)/);
        const kernelM = argsStr.match(/(?:kernel_size|kernel)\s*=\s*(?:\(?\s*(\d+)|(\d+))/);
        const strideM = argsStr.match(/stride\s*=\s*(\d+)/);
        const padM = argsStr.match(/padding\s*=\s*(\d+)/);
        const dilM = argsStr.match(/dilation\s*=\s*(\d+)/);
        const grpM = argsStr.match(/groups\s*=\s*(\d+)/);

        const rawArgs = argsStr.split(",").map((s) => s.trim()).filter((s) => s && !s.includes("="));
        const positionalOut = rawArgs[1] ? parseInt(rawArgs[1], 10) : undefined;
        const positionalKernel = rawArgs[2] ? parseInt(rawArgs[2], 10) : undefined;
        const positionalStride = rawArgs[3] ? parseInt(rawArgs[3], 10) : undefined;
        const positionalPad = rawArgs[4] ? parseInt(rawArgs[4], 10) : undefined;

        const filters = outChannelsM ? parseInt(outChannelsM[1], 10) : (positionalOut || 32);
        const kernel = (kernelM ? parseInt(kernelM[1] || kernelM[2], 10) : positionalKernel) || 3;
        const stride = strideM ? parseInt(strideM[1], 10) : (positionalStride || 1);
        const padding = padM ? parseInt(padM[1], 10) : (positionalPad !== undefined ? positionalPad : 1);
        const dilation = dilM ? parseInt(dilM[1], 10) : 1;
        const groups = grpM ? parseInt(grpM[1], 10) : 1;

        if (groups > 1) {
          layerType = "DepthwiseConv";
          params = { multiplier: 1, kernel, padding };
        } else if (dilation > 1) {
          layerType = "DilatedConv";
          params = { filters, kernel, dilation, padding };
        } else if (kernel === 1 && stride === 1) {
          layerType = "PointwiseConv";
          params = { filters };
        } else {
          layerType = "Conv2d";
          params = { filters, kernel, stride, padding };
        }
      } else if (typeName === "ConvTranspose2d") {
        layerType = "TransposedConv";
        const outChannelsM = argsStr.match(/(?:out_channels|filters)\s*=\s*(\d+)/);
        const kernelM = argsStr.match(/(?:kernel_size|kernel)\s*=\s*(\d+)/);
        const strideM = argsStr.match(/stride\s*=\s*(\d+)/);
        const padM = argsStr.match(/padding\s*=\s*(\d+)/);

        const rawArgs = argsStr.split(",").map((s) => s.trim()).filter((s) => s && !s.includes("="));
        params = {
          filters: outChannelsM ? parseInt(outChannelsM[1], 10) : (parseInt(rawArgs[1], 10) || 32),
          kernel: kernelM ? parseInt(kernelM[1], 10) : (parseInt(rawArgs[2], 10) || 2),
          stride: strideM ? parseInt(strideM[1], 10) : 2,
          padding: padM ? parseInt(padM[1], 10) : 0,
        };
      } else if (typeName === "MaxPool2d") {
        layerType = "MaxPool2d";
        const kM = argsStr.match(/(?:kernel_size|kernel)\s*=\s*(\d+)/) || argsStr.match(/(\d+)/);
        params = { kernel: kM ? parseInt(kM[1], 10) : 2 };
      } else if (typeName === "AvgPool2d") {
        layerType = "AveragePool";
        const kM = argsStr.match(/(?:kernel_size|kernel)\s*=\s*(\d+)/) || argsStr.match(/(\d+)/);
        params = { kernel: kM ? parseInt(kM[1], 10) : 2 };
      } else if (typeName === "AdaptiveAvgPool2d") {
        if (/\(?\s*1\s*(?:,\s*1\s*)?\)?/.test(argsStr) && !/size\s*=\s*[2-9]/.test(argsStr)) {
          layerType = "GlobalAvgPool";
          params = {};
        } else {
          layerType = "AdaptiveAvgPool";
          const sM = argsStr.match(/(?:output_size|size)\s*=\s*(\d+)/) || argsStr.match(/(\d+)/);
          params = { size: sM ? parseInt(sM[1], 10) : 1 };
        }
      } else if (typeName === "Linear" || typeName === "LazyLinear") {
        layerType = "Linear";
        const outM = argsStr.match(/(?:out_features|units)\s*=\s*(\d+)/);
        if (outM) {
          params = { units: parseInt(outM[1], 10) };
        } else {
          const rawArgs = argsStr.split(",").map((s) => s.trim()).filter((s) => s && !s.includes("="));
          if (rawArgs.length === 1 && /^\d+$/.test(rawArgs[0])) {
            params = { units: parseInt(rawArgs[0], 10) };
          } else if (rawArgs.length >= 2 && /^\d+$/.test(rawArgs[1])) {
            params = { units: parseInt(rawArgs[1], 10) };
          } else {
            const numM = argsStr.match(/(\d+)/);
            params = { units: numM ? parseInt(numM[1], 10) : 128 };
          }
        }
      } else if (typeName === "Dropout") {
        layerType = "Dropout";
        const rM = argsStr.match(/(?:p|rate)\s*=\s*([0-9.]+)/) || argsStr.match(/([0-9.]+)/);
        params = { rate: rM ? parseFloat(rM[1]) : 0.25 };
      } else if (typeName === "Dropout2d") {
        layerType = "Dropout2D";
        const rM = argsStr.match(/(?:p|rate)\s*=\s*([0-9.]+)/) || argsStr.match(/([0-9.]+)/);
        params = { rate: rM ? parseFloat(rM[1]) : 0.25 };
      } else if (typeName === "BatchNorm2d" || typeName === "BatchNorm") {
        layerType = "BatchNorm";
      } else if (typeName === "InstanceNorm2d" || typeName === "InstanceNorm") {
        layerType = "InstanceNorm";
      } else if (typeName === "GroupNorm") {
        layerType = "GroupNorm";
        const gM = argsStr.match(/(?:num_groups|groups)\s*=\s*(\d+)/) || argsStr.match(/(\d+)/);
        params = { groups: gM ? parseInt(gM[1], 10) : 8 };
      } else if (typeName === "LayerNorm") {
        layerType = "LayerNorm";
      } else if (typeName === "ReLU") {
        layerType = "ReLU";
      } else if (typeName === "LeakyReLU") {
        layerType = "LeakyReLU";
        const slM = argsStr.match(/(?:negative_slope|slope)\s*=\s*([0-9.]+)/) || argsStr.match(/([0-9.]+)/);
        params = { slope: slM ? parseFloat(slM[1]) : 0.1 };
      } else if (typeName === "GELU") {
        layerType = "GELU";
      } else if (typeName === "SiLU") {
        layerType = "SiLU";
      } else if (typeName === "ELU") {
        layerType = "ELU";
      } else if (typeName === "SELU") {
        layerType = "SELU";
      } else if (typeName === "Tanh") {
        layerType = "Tanh";
      } else if (typeName === "Sigmoid") {
        layerType = "Sigmoid";
      } else if (typeName === "Softplus") {
        layerType = "Softplus";
      } else if (typeName === "SelfAttentionBlock" || typeName === "SelfAttention") {
        layerType = "SelfAttention";
        const hM = argsStr.match(/(?:heads|num_heads)\s*=\s*(\d+)/) || argsStr.match(/,\s*(\d+)/);
        params = { heads: hM ? parseInt(hM[1], 10) : 4 };
      } else if (typeName === "CrossAttentionBlock" || typeName === "CrossAttention") {
        layerType = "CrossAttention";
        const hM = argsStr.match(/(?:heads|num_heads)\s*=\s*(\d+)/) || argsStr.match(/,\s*(\d+)/);
        params = { heads: hM ? parseInt(hM[1], 10) : 8 };
      } else if (layerLibrary && layerLibrary[typeName]) {
        layerType = typeName;
      }

      return { layerType, params };
    }

    // 4. Method A: Parse annotated Forward pass steps (# 01 · ...)
    const stepRegex = /#\s*(\d+)\s*·\s*([^\n\r]+)([\s\S]*?)(?=(?:#\s*\d+\s*·)|return\s+x|$)/g;
    let stepMatch;
    let stepCounter = 0;

    while ((stepMatch = stepRegex.exec(code)) !== null) {
      stepCounter++;
      const stepIdx = stepCounter;
      const stepLabel = stepMatch[2].trim();
      const stepBody = stepMatch[3];

      const callMatch = stepBody.match(/self\.([a-zA-Z0-9_]+)\s*\(/);
      const varName = callMatch ? callMatch[1] : null;
      const def = varName ? layerDefs.get(varName) : null;

      let layerType = "Linear";
      let params = {};

      // Match connection operations
      if (/x\s*\+\s*features\[(\d+)\]/.test(stepBody)) {
        layerType = "Add";
        const fromStep = parseInt(stepBody.match(/features\[(\d+)\]/)[1], 10);
        result.connections.push({ from: fromStep, to: stepIdx });
      } else if (/x\s*\*\s*features\[(\d+)\]/.test(stepBody)) {
        layerType = "Multiply";
        const fromStep = parseInt(stepBody.match(/features\[(\d+)\]/)[1], 10);
        result.connections.push({ from: fromStep, to: stepIdx });
      } else if (/torch\.cat\(.*features\[(\d+)\]/.test(stepBody)) {
        layerType = "Concatenate";
        const fromStep = parseInt(stepBody.match(/features\[(\d+)\]/)[1], 10);
        result.connections.push({ from: fromStep, to: stepIdx });
      } else if (/x\s*=\s*features\[(\d+)\]/.test(stepBody)) {
        layerType = "Skip";
        const fromStep = parseInt(stepBody.match(/features\[(\d+)\]/)[1], 10);
        result.connections.push({ from: fromStep, to: stepIdx });
      } else if (/torch\.flatten/.test(stepBody) || /Flatten/i.test(stepLabel)) {
        layerType = "Flatten";
      } else if (/torch\.softmax/.test(stepBody)) {
        layerType = "Softmax";
      } else if (/torch\.log_softmax/.test(stepBody)) {
        layerType = "LogSoftmax";
      } else if (/F\.layer_norm/.test(stepBody)) {
        layerType = "LayerNorm";
      } else if (def) {
        const resolved = resolveDef(def);
        layerType = resolved.layerType;
        params = resolved.params;
      } else {
        // Fallback matching from stepLabel
        for (const [type, info] of Object.entries(layerLibrary || {})) {
          if (info.label?.toLowerCase() === stepLabel.toLowerCase()) {
            layerType = type;
            params = { ...info.params };
            break;
          }
        }
      }

      const defaultParams = layerLibrary && layerLibrary[layerType] ? layerLibrary[layerType].params : {};
      result.layers.push({
        id: stepIdx,
        type: layerType,
        params: { ...defaultParams, ...params },
      });
    }

    // 5. Method B: If no forward step comments, parse calls from forward method body
    if (result.layers.length === 0) {
      const forwardMethodMatch = code.match(/def\s+forward\s*\([^)]*\):([\s\S]*?)(?=\n\s*def|\n\s*class|\n\s*device|\n\s*model\s*=|\n\s*optimizer\s*=|root=|$)/);
      if (forwardMethodMatch) {
        const forwardLines = forwardMethodMatch[1].split("\n");
        let step = 0;

        for (const rawLine of forwardLines) {
          const line = rawLine.trim();
          if (!line || line.startsWith("#") || line.startsWith("return") || line.startsWith("features =")) continue;

          step++;
          let layerType = "Linear";
          let params = {};

          if (/x\s*\+\s*features\[(\d+)\]/.test(line)) {
            layerType = "Add";
            const fromStep = parseInt(line.match(/features\[(\d+)\]/)[1], 10);
            result.connections.push({ from: fromStep, to: step });
          } else if (/torch\.flatten/.test(line)) {
            layerType = "Flatten";
          } else if (/F\.layer_norm/.test(line)) {
            layerType = "LayerNorm";
          } else if (/torch\.softmax/.test(line)) {
            layerType = "Softmax";
          } else if (/torch\.log_softmax/.test(line)) {
            layerType = "LogSoftmax";
          } else {
            const callMatch = line.match(/self\.([a-zA-Z0-9_]+)\s*\(/);
            if (callMatch) {
              const def = layerDefs.get(callMatch[1]);
              const resolved = resolveDef(def);
              layerType = resolved.layerType;
              params = resolved.params;
            } else {
              continue;
            }
          }

          const defaultParams = layerLibrary && layerLibrary[layerType] ? layerLibrary[layerType].params : {};
          result.layers.push({
            id: step,
            type: layerType,
            params: { ...defaultParams, ...params },
          });
        }
      }
    }

    // 6. Method C: Fallback to __init__ layer definitions in order if forward could not be parsed
    if (result.layers.length === 0 && orderedDefs.length > 0) {
      orderedDefs.forEach((def, idx) => {
        const resolved = resolveDef(def);
        const defaultParams = layerLibrary && layerLibrary[resolved.layerType] ? layerLibrary[resolved.layerType].params : {};
        result.layers.push({
          id: idx + 1,
          type: resolved.layerType,
          params: { ...defaultParams, ...resolved.params },
        });
      });
    }

    // Return null if no valid layers could be parsed
    if (result.layers.length === 0) return null;

    return result;
  } catch (err) {
    console.error("parseCodeToArchitecture error:", err);
    return null;
  }
}
