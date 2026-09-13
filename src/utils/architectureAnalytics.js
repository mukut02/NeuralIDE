/**
 * Neural Network Architecture Analytics & Synthetic Dry Run Engine
 *
 * Computes:
 * - Trainable parameter counts per layer & total
 * - Forward activation memory consumption
 * - Theoretical FLOPs / MACs
 * - Simulated activation feature heatmaps for interactive dry run
 */

export function calculateLayerParams(layer, inputShape, outputShape) {
  if (!layer) return 0;
  if (layer.params?.paramCount !== undefined) {
    return Number(layer.params.paramCount);
  }
  if (layer.paramCount !== undefined) {
    return Number(layer.paramCount);
  }
  if (!inputShape || !outputShape) return 0;
  const p = layer.params || {};

  switch (layer.type) {
    case "Conv2d":
    case "DilatedConv":
    case "TransposedConv": {
      const cIn = inputShape.kind === "image" ? inputShape.c : 1;
      const cOut = outputShape.kind === "image" ? outputShape.c : (p.filters || 32);
      const k = p.kernel || 3;
      // Weights: cIn * cOut * k * k + bias: cOut
      return cIn * cOut * k * k + cOut;
    }
    case "PointwiseConv": {
      const cIn = inputShape.kind === "image" ? inputShape.c : 1;
      const cOut = outputShape.kind === "image" ? outputShape.c : (p.filters || 32);
      return cIn * cOut * 1 * 1 + cOut;
    }
    case "DepthwiseConv": {
      const cIn = inputShape.kind === "image" ? inputShape.c : 1;
      const mult = p.multiplier || 1;
      const k = p.kernel || 3;
      // Depthwise: cIn * mult * k * k + cIn * mult
      return cIn * mult * k * k + cIn * mult;
    }
    case "Linear":
    case "Dense":
    case "ClassifierHead":
    case "ClassificationHead":
    case "RegressionHead": {
      const nIn = inputShape.kind === "vector" ? inputShape.n : (inputShape.c * (inputShape.h || 1) * (inputShape.w || 1));
      const nOut = outputShape.kind === "vector" ? outputShape.n : (p.units || 10);
      return nIn * nOut + nOut;
    }
    case "Bilinear": {
      const nIn = inputShape.kind === "vector" ? inputShape.n : 128;
      const nOut = outputShape.kind === "vector" ? outputShape.n : (p.units || 10);
      return nIn * nIn * nOut + nOut;
    }
    case "BatchNorm":
    case "InstanceNorm": {
      const c = inputShape.kind === "image" ? inputShape.c : (inputShape.n || 32);
      return 2 * c; // gamma + beta
    }
    case "LayerNorm": {
      const dim = inputShape.kind === "vector" ? inputShape.n : (inputShape.c || 32);
      return 2 * dim;
    }
    case "GroupNorm": {
      const c = inputShape.kind === "image" ? inputShape.c : 32;
      return 2 * c;
    }
    case "SelfAttention":
    case "MultiHeadAttention": {
      const c = inputShape.kind === "image" ? inputShape.c : (inputShape.n || 64);
      // Q, K, V projections + Output projection = 4 * c * c + 4 * c
      return 4 * c * c + 4 * c;
    }
    case "CrossAttention": {
      const c = inputShape.kind === "image" ? inputShape.c : (inputShape.n || 64);
      return 4 * c * c + 4 * c;
    }
    case "SEBlock":
    case "CBAM": {
      const c = inputShape.kind === "image" ? inputShape.c : 32;
      const r = p.reduction || 16;
      return 2 * Math.floor(c * (c / r));
    }
    default:
      // Activations, Pooling, Flatten, Reshape, Skip, Add, Multiply, Concatenate = 0 params
      return 0;
  }
}

export function calculateLayerFlops(layer, inputShape, outputShape) {
  if (!layer || !inputShape || !outputShape) return 0;
  const p = layer.params || {};

  if (["Conv2d", "DilatedConv", "PointwiseConv"].includes(layer.type) && outputShape.kind === "image") {
    const cIn = inputShape.kind === "image" ? inputShape.c : 1;
    const k = layer.type === "PointwiseConv" ? 1 : (p.kernel || 3);
    return 2 * cIn * k * k * outputShape.c * outputShape.h * outputShape.w;
  }

  if (["Linear", "Dense", "ClassifierHead", "ClassificationHead"].includes(layer.type)) {
    const nIn = inputShape.kind === "vector" ? inputShape.n : 128;
    const nOut = outputShape.kind === "vector" ? outputShape.n : 10;
    return 2 * nIn * nOut;
  }

  if (["MaxPool2d", "AveragePool"].includes(layer.type) && outputShape.kind === "image") {
    const k = p.kernel || 2;
    return inputShape.c * outputShape.h * outputShape.w * k * k;
  }

  return 0;
}

export function calculateTensorBytes(shape, batchSize = 1, dtypeBytes = 4) {
  if (!shape) return 0;
  if (shape.kind === "image") {
    return batchSize * shape.c * shape.h * shape.w * dtypeBytes;
  }
  if (shape.kind === "vector") {
    return batchSize * shape.n * dtypeBytes;
  }
  return 0;
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function formatParams(num) {
  if (!num || num === 0) return "0.00M";
  const inMillions = num / 1000000;
  if (inMillions >= 100) return `${inMillions.toFixed(1)}M`;
  if (inMillions >= 1) return `${inMillions.toFixed(2)}M`;
  if (inMillions >= 0.01) return `${inMillions.toFixed(2)}M`;
  return `${inMillions.toFixed(3)}M`;
}

export function formatParamsDetailed(num) {
  if (!num || num === 0) return "0.00M (0 params)";
  const formattedM = formatParams(num);
  return `${formattedM} (${num.toLocaleString()} params)`;
}

export function formatFlops(num) {
  if (num === 0) return "—";
  if (num < 1000) return `${num} FLOPs`;
  if (num < 1000000) return `${(num / 1000).toFixed(1)} KFLOPs`;
  if (num < 1000000000) return `${(num / 1000000).toFixed(1)} MFLOPs`;
  return `${(num / 1000000000).toFixed(2)} GFLOPs`;
}

/**
 * Procedural synthesis of authentic input sample patterns per dataset class
 */
export function generateInputPattern(datasetName, classIdx = 0, gridDim = 16, channel = 0) {
  const grid = [];
  const cx = gridDim / 2;
  const cy = gridDim / 2;

  for (let y = 0; y < gridDim; y++) {
    const row = [];
    const ny = (y - cy) / (gridDim / 2); // -1.0 to 1.0

    for (let x = 0; x < gridDim; x++) {
      const nx = (x - cx) / (gridDim / 2); // -1.0 to 1.0
      let val;

      if (datasetName === "MNIST") {
        switch (classIdx) {
          case 0: { // Oval loop
            const r = Math.hypot(nx * 1.15, ny);
            val = Math.abs(r - 0.65) < 0.22 ? 0.95 : 0.05;
            break;
          }
          case 1: { // Vertical stem with top hook
            const isStem = Math.abs(nx - 0.05) < 0.16 && ny >= -0.75 && ny <= 0.75;
            const isSerif = ny < -0.45 && nx >= -0.35 && nx <= 0.05 && Math.abs(ny - (-0.75 - (nx - 0.05) * 0.8)) < 0.2;
            val = isStem || isSerif ? 0.95 : 0.05;
            break;
          }
          case 2: { // Digit 2
            const topArch = ny < -0.1 && Math.hypot(nx, ny + 0.4) < 0.45 && Math.hypot(nx, ny + 0.4) > 0.22 && nx > -0.35;
            const diag = ny >= -0.1 && ny <= 0.6 && Math.abs(nx - (-0.4 + (0.6 - ny) * 0.9)) < 0.18;
            const base = ny > 0.55 && ny < 0.8 && nx > -0.6 && nx < 0.6;
            val = topArch || diag || base ? 0.95 : 0.05;
            break;
          }
          case 3: { // Digit 3
            const top = ny < 0 && Math.hypot(nx, ny + 0.35) < 0.44 && Math.hypot(nx, ny + 0.35) > 0.22 && nx > -0.25;
            const bot = ny >= 0 && Math.hypot(nx, ny - 0.35) < 0.46 && Math.hypot(nx, ny - 0.35) > 0.22 && nx > -0.3;
            val = top || bot ? 0.95 : 0.05;
            break;
          }
          case 4: { // Digit 4
            const vRight = Math.abs(nx - 0.25) < 0.16 && ny > -0.75 && ny < 0.75;
            const hBar = Math.abs(ny - 0.15) < 0.16 && nx > -0.65 && nx < 0.45;
            const vLeft = Math.abs(nx - (-0.45 + (0.15 - ny) * 0.5)) < 0.16 && ny > -0.75 && ny <= 0.15;
            val = vRight || hBar || vLeft ? 0.95 : 0.05;
            break;
          }
          case 5: { // Digit 5
            const topBar = ny > -0.75 && ny < -0.55 && nx > -0.55 && nx < 0.55;
            const stem = nx > -0.55 && nx < -0.35 && ny >= -0.65 && ny <= -0.05;
            const botLoop = ny > -0.15 && Math.hypot(nx, ny - 0.3) < 0.48 && Math.hypot(nx, ny - 0.3) > 0.24 && nx > -0.4;
            val = topBar || stem || botLoop ? 0.95 : 0.05;
            break;
          }
          case 6: { // Digit 6
            const loop = ny > -0.1 && Math.hypot(nx, ny - 0.3) < 0.46 && Math.hypot(nx, ny - 0.3) > 0.22;
            const spine = nx < 0 && Math.abs(nx - (-0.35 - (ny - 0.2) * 0.4)) < 0.18 && ny >= -0.75 && ny <= 0.3;
            val = loop || spine ? 0.95 : 0.05;
            break;
          }
          case 7: { // Digit 7
            const isTopBar = ny >= -0.75 && ny <= -0.55 && nx >= -0.6 && nx <= 0.6;
            const isDiag = Math.abs(nx - (0.55 - (ny + 0.6) * 0.75)) < 0.18 && ny >= -0.6 && ny <= 0.75;
            val = isTopBar || isDiag ? 0.95 : 0.05;
            break;
          }
          case 8: { // Digit 8
            const top8 = Math.hypot(nx, ny + 0.35);
            const bot8 = Math.hypot(nx, ny - 0.35);
            const isTop = Math.abs(top8 - 0.36) < 0.18;
            const isBot = Math.abs(bot8 - 0.42) < 0.20;
            val = isTop || isBot ? 0.95 : 0.05;
            break;
          }
          default: { // Digit 9
            const loop9 = ny < 0.1 && Math.hypot(nx, ny + 0.3) < 0.46 && Math.hypot(nx, ny + 0.3) > 0.22;
            const spine9 = nx > 0 && Math.abs(nx - (0.35 - (ny + 0.3) * 0.2)) < 0.18 && ny >= -0.5 && ny <= 0.75;
            val = loop9 || spine9 ? 0.95 : 0.05;
            break;
          }
        }
      } else if (datasetName === "Fashion-MNIST") {
        switch (classIdx) {
          case 0: { // T-shirt
            const collar = Math.hypot(nx, ny + 0.75) < 0.28;
            const torso = Math.abs(nx) < 0.48 && ny > -0.65 && ny < 0.75;
            const sleeves = ny > -0.65 && ny < -0.15 && Math.abs(nx) < 0.85;
            val = (torso || sleeves) && !collar ? 0.92 : 0.05;
            break;
          }
          case 1: { // Trouser
            const waist = Math.abs(nx) < 0.5 && ny > -0.75 && ny < -0.45;
            const leftLeg = nx > -0.48 && nx < -0.06 && ny >= -0.45 && ny < 0.82;
            const rightLeg = nx > 0.06 && nx < 0.48 && ny >= -0.45 && ny < 0.82;
            val = waist || leftLeg || rightLeg ? 0.92 : 0.05;
            break;
          }
          case 2: { // Pullover
            const torso = Math.abs(nx) < 0.52 && ny > -0.7 && ny < 0.75;
            const armL = nx > -0.85 && nx < -0.5 && ny > -0.65 && ny < 0.65;
            const armR = nx > 0.5 && nx < 0.85 && ny > -0.65 && ny < 0.65;
            val = torso || armL || armR ? 0.92 : 0.05;
            break;
          }
          case 3: { // Dress
            const bodice = Math.abs(nx) < 0.35 && ny > -0.75 && ny < -0.15;
            const skirt = ny >= -0.15 && ny < 0.75 && Math.abs(nx) < (0.35 + (ny + 0.15) * 0.5);
            val = bodice || skirt ? 0.92 : 0.05;
            break;
          }
          case 4: { // Coat
            const body = Math.abs(nx) < 0.58 && ny > -0.75 && ny < 0.82;
            const slit = Math.abs(nx) < 0.06 && ny > -0.5 && ny < 0.82;
            val = body && !slit ? 0.92 : 0.05;
            break;
          }
          case 5: { // Sandal
            const sole = ny > 0.45 && ny < 0.68 && nx > -0.75 && nx < 0.75;
            const strap1 = Math.abs(nx + 0.35) < 0.12 && ny > 0.15 && ny < 0.5;
            const strap2 = Math.abs(nx - 0.25) < 0.12 && ny > 0.15 && ny < 0.5;
            val = sole || strap1 || strap2 ? 0.92 : 0.05;
            break;
          }
          case 6: { // Shirt
            const body = Math.abs(nx) < 0.52 && ny > -0.7 && ny < 0.75;
            const buttons = Math.abs(nx) < 0.04 && ny > -0.6 && ny < 0.7;
            val = body && !buttons ? 0.92 : 0.05;
            break;
          }
          case 7: { // Sneaker
            const sole = ny > 0.45 && ny < 0.7 && nx > -0.8 && nx < 0.8;
            const body = ny > 0.1 && ny <= 0.45 && nx > -0.75 && nx < 0.45;
            val = sole || body ? 0.92 : 0.05;
            break;
          }
          case 8: { // Bag
            const body = Math.abs(nx) < 0.58 && ny > -0.2 && ny < 0.75;
            const handle = Math.abs(Math.hypot(nx, ny + 0.2) - 0.38) < 0.12 && ny < -0.2;
            val = body || handle ? 0.92 : 0.05;
            break;
          }
          default: { // Ankle Boot
            const sole = ny > 0.48 && ny < 0.75 && nx > -0.75 && nx < 0.75;
            const foot = ny > 0.15 && ny <= 0.48 && nx > -0.75 && nx < 0.65;
            const shaft = nx > -0.75 && nx < -0.15 && ny > -0.65 && ny <= 0.15;
            val = sole || foot || shaft ? 0.92 : 0.05;
            break;
          }
        }
      } else {
        // CIFAR-10 (with authentic 3-channel RGB distribution)
        switch (classIdx) {
          case 0: { // Airplane
            const fuselage = Math.abs(ny) < 0.18 && nx > -0.75 && nx < 0.75;
            const wings = Math.abs(nx + 0.1) < 0.22 && Math.abs(ny) < 0.75;
            const tail = nx < -0.55 && Math.abs(ny) < 0.45;
            const isPlane = fuselage || wings || tail;
            if (isPlane) {
              val = channel === 0 ? 0.95 : channel === 1 ? 0.92 : 0.98; // silver plane
            } else {
              val = channel === 2 ? 0.82 : channel === 1 ? 0.52 : 0.22; // sky blue
            }
            break;
          }
          case 1: { // Automobile
            const cabin = ny > -0.55 && ny <= -0.05 && nx > -0.45 && nx < 0.4;
            const body = ny > -0.05 && ny < 0.45 && nx > -0.78 && nx < 0.78;
            const wheel1 = Math.hypot(nx + 0.45, ny - 0.45) < 0.22;
            const wheel2 = Math.hypot(nx - 0.45, ny - 0.45) < 0.22;
            const isCar = (cabin || body) && !wheel1 && !wheel2;
            if (isCar) {
              val = channel === 0 ? 0.95 : channel === 1 ? 0.35 : 0.28; // red car body
            } else if (wheel1 || wheel2) {
              val = 0.12; // dark rubber wheels
            } else {
              val = channel === 2 ? 0.42 : 0.25; // pavement
            }
            break;
          }
          case 2: { // Bird
            const body = Math.hypot(nx, ny) < 0.42;
            const head = Math.hypot(nx - 0.35, ny + 0.35) < 0.25;
            const wing = nx < 0 && ny < 0.2 && Math.abs(nx + ny) < 0.5;
            const isBird = body || head || wing;
            val = isBird ? (channel === 1 ? 0.92 : channel === 0 ? 0.75 : 0.28) : (channel === 2 ? 0.6 : 0.15);
            break;
          }
          case 3: { // Cat
            const head = Math.hypot(nx, ny - 0.05) < 0.46;
            const earL = nx > -0.45 && nx < -0.15 && ny > -0.65 && ny < -0.25;
            const earR = nx > 0.15 && nx < 0.45 && ny > -0.65 && ny < -0.25;
            const eyeL = Math.hypot(nx + 0.18, ny - 0.05) < 0.09;
            const eyeR = Math.hypot(nx - 0.18, ny - 0.05) < 0.09;
            const isCat = (head || earL || earR) && !eyeL && !eyeR;
            val = isCat ? (channel === 0 ? 0.88 : channel === 1 ? 0.65 : 0.4) : (eyeL || eyeR ? 0.98 : 0.08);
            break;
          }
          case 4: { // Deer
            const body = Math.abs(nx + 0.1) < 0.45 && ny > -0.1 && ny < 0.35;
            const neck = nx > 0.2 && nx < 0.5 && ny > -0.65 && ny < 0.0;
            const legs = (Math.abs(nx + 0.4) < 0.08 || Math.abs(nx + 0.15) < 0.08 || Math.abs(nx - 0.25) < 0.08) && ny >= 0.35 && ny < 0.8;
            const isDeer = body || neck || legs;
            val = isDeer ? (channel === 0 ? 0.85 : channel === 1 ? 0.55 : 0.25) : (channel === 1 ? 0.45 : 0.1);
            break;
          }
          case 5: { // Dog
            const body = Math.abs(nx + 0.15) < 0.5 && ny > -0.1 && ny < 0.4;
            const head = Math.hypot(nx - 0.38, ny + 0.15) < 0.32;
            const snout = nx > 0.45 && nx < 0.75 && ny > 0.05 && ny < 0.3;
            const isDog = body || head || snout;
            val = isDog ? (channel === 0 ? 0.82 : channel === 1 ? 0.58 : 0.35) : 0.1;
            break;
          }
          case 6: { // Frog
            const body = Math.hypot(nx * 0.9, ny) < 0.52;
            const eyeL = Math.hypot(nx + 0.3, ny + 0.38) < 0.18;
            const eyeR = Math.hypot(nx - 0.3, ny + 0.38) < 0.18;
            const isFrog = body || eyeL || eyeR;
            val = isFrog ? (channel === 1 ? 0.96 : channel === 0 ? 0.35 : 0.25) : 0.08;
            break;
          }
          case 7: { // Horse
            const torso = Math.abs(nx) < 0.52 && ny > -0.15 && ny < 0.35;
            const neck = nx > 0.25 && nx < 0.65 && ny > -0.7 && ny < 0.0;
            const leg1 = Math.abs(nx + 0.4) < 0.1 && ny >= 0.35 && ny < 0.82;
            const leg2 = Math.abs(nx - 0.35) < 0.1 && ny >= 0.35 && ny < 0.82;
            const isHorse = torso || neck || leg1 || leg2;
            val = isHorse ? (channel === 0 ? 0.75 : channel === 1 ? 0.5 : 0.32) : 0.12;
            break;
          }
          case 8: { // Ship
            const hull = ny >= 0.15 && ny < 0.55 && Math.abs(nx) < (0.75 - (ny - 0.15) * 0.3);
            const mast = Math.abs(nx + 0.1) < 0.08 && ny > -0.65 && ny < 0.15;
            const isShip = hull || mast;
            const isSea = ny >= 0.55;
            if (isShip) {
              val = channel === 0 ? 0.88 : 0.4;
            } else if (isSea) {
              val = channel === 2 ? 0.92 : channel === 1 ? 0.55 : 0.15; // ocean water
            } else {
              val = channel === 2 ? 0.55 : 0.35;
            }
            break;
          }
          default: { // Truck
            const cargo = nx > -0.8 && nx < 0.15 && ny > -0.65 && ny < 0.45;
            const cab = nx >= 0.15 && nx < 0.75 && ny > -0.25 && ny < 0.45;
            const wheels = (Math.hypot(nx + 0.5, ny - 0.48) < 0.2 || Math.hypot(nx + 0.1, ny - 0.48) < 0.2 || Math.hypot(nx - 0.5, ny - 0.48) < 0.2);
            const isTruck = (cargo || cab) && !wheels;
            val = isTruck ? (channel === 0 ? 0.9 : channel === 1 ? 0.6 : 0.2) : (wheels ? 0.15 : 0.08);
            break;
          }
        }
      }

      row.push(Math.max(0.02, Math.min(0.98, val)));
    }
    grid.push(row);
  }
  return grid;
}

/**
 * Generate simulated activation slices for the visual dry run
 * Dynamically reacts to the chosen dataset and selectedClass index
 */
export function generateSyntheticFeatureMaps(stepIndex, totalSteps, shape, dataset, selectedClass = 0) {
  if (!shape) return null;

  const datasetName = dataset?.name || "MNIST";
  const numClasses = dataset?.classes || 10;
  const validClass = Math.max(0, Math.min(numClasses - 1, selectedClass || 0));
  const classLabel = dataset?.classesList?.[validClass] || `Class ${validClass}`;

  // For 1D vector representations (Classification Head / Dense Layers)
  if (shape.kind === "vector") {
    const isFinal = stepIndex === totalSteps;
    const size = Math.min(shape.n, numClasses);

    // Target class receives top-1 probability; other classes receive realistic lower distribution
    if (isFinal) {
      // 1. Synthesize unnormalized logits z_i
      const rawLogits = [];
      for (let i = 0; i < size; i++) {
        if (i === validClass) {
          // Target ground truth class receives dominant logit
          rawLogits.push(4.8 + Math.sin(validClass * 2) * 0.25);
        } else {
          // Secondary classes with small plausible confusions based on semantic distance
          const diff = Math.abs(i - validClass);
          rawLogits.push(Math.max(-1.8, 0.6 / (diff + 1) + Math.sin((i + validClass) * 1.5) * 0.25));
        }
      }

      // 2. Exact mathematical Softmax: σ(z)_i = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
      const maxLogit = Math.max(...rawLogits);
      const expScores = rawLogits.map((z) => Math.exp(z - maxLogit));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      const softmaxProbs = expScores.map((e) => e / sumExp);

      return {
        kind: "vector",
        values: softmaxProbs,
        rawLogits,
        targetClass: validClass,
        targetClassLabel: classLabel,
      };
    }

    // For intermediate hidden vectors (Flatten / Dense layers before final output)
    const hiddenActivations = [];
    for (let i = 0; i < size; i++) {
      const activation = Math.sin((i * 1.8 + validClass * 2.4 + stepIndex) * 0.75) * 0.45 + 0.5;
      hiddenActivations.push(Math.max(0.04, Math.min(0.96, activation)));
    }

    return {
      kind: "vector",
      values: hiddenActivations,
      targetClass: null,
      targetClassLabel: null,
    };
  }

  // For 4D image feature maps (channels x height x width)
  const previewChannels = Math.min(shape.c, 4);
  const gridDim = Math.min(Math.max(shape.h, 6), 16);
  const slices = [];

  // Generate input pattern as baseline for this class
  const baseInputPattern = generateInputPattern(datasetName, validClass, gridDim, 0);

  for (let c = 0; c < previewChannels; c++) {
    const channelInput = previewChannels > 1 && datasetName === "CIFAR-10"
      ? generateInputPattern(datasetName, validClass, gridDim, c)
      : baseInputPattern;

    const grid = [];
    for (let y = 0; y < gridDim; y++) {
      const row = [];
      for (let x = 0; x < gridDim; x++) {
        let val;

        if (stepIndex === 0) {
          // Step 0: Input Tensor directly renders the authentic class sample
          val = channelInput[y][x];
        } else if (stepIndex === 1) {
          // Step 1: Low-Level Convolutions (directional edge gradients & oriented filters)
          const cur = channelInput[y][x];
          const prevY = channelInput[Math.max(0, y - 1)][x];
          const prevX = channelInput[y][Math.max(0, x - 1)];
          const prevDiag = channelInput[Math.max(0, y - 1)][Math.max(0, x - 1)];

          if (c === 0) {
            val = Math.abs(cur - prevY) * 2.5 + 0.04;
          } else if (c === 1) {
            val = Math.abs(cur - prevX) * 2.5 + 0.04;
          } else if (c === 2) {
            val = Math.abs(cur - prevDiag) * 2.8 + 0.04;
          } else {
            val = Math.hypot(cur - prevX, cur - prevY) * 2.2 + 0.04;
          }
        } else if (stepIndex === 2) {
          // Step 2: Second Conv / Activation (corner junctions, curvature, laplacian contours)
          const cur = channelInput[y][x];
          const nextY = channelInput[Math.min(gridDim - 1, y + 1)][x];
          const nextX = channelInput[y][Math.min(gridDim - 1, x + 1)];
          const laplacian = Math.abs(4 * cur - nextX - nextY - channelInput[Math.max(0, y - 1)][x] - channelInput[y][Math.max(0, x - 1)]);
          const phase = (c + 1) * 1.2;

          if (c === 0) {
            val = Math.min(0.95, laplacian * 2.2 + 0.05);
          } else if (c === 1) {
            val = Math.max(0.04, Math.sin(cur * Math.PI + phase) * 0.75 + 0.15);
          } else if (c === 2) {
            val = Math.abs(cur - (nextX + nextY) * 0.5) * 2.6 + 0.05;
          } else {
            val = Math.max(0.04, (cur > 0.3 ? 0.85 : 0.08) * (1 + Math.sin(phase) * 0.2));
          }
        } else if (stepIndex <= 5) {
          // Steps 3–5: Mid-Level Hierarchy (receptive field expanding, motif grouping)
          const inputVal = channelInput[y][x];
          const distFromCenter = Math.hypot(x - gridDim / 2, y - gridDim / 2) / (gridDim / 2);
          const frequency = 2.4 + stepIndex * 0.6;
          const harmonic = Math.cos(distFromCenter * frequency + (c + stepIndex) * 0.85);
          const partFocus = inputVal > 0.25 ? 0.7 + (stepIndex * 0.04) : 0.05;
          val = Math.max(0.04, partFocus * 0.72 + harmonic * 0.28);
        } else if (stepIndex <= 8) {
          // Steps 6–8: Deep Convolutional Representations (semantic object part detectors)
          const inputVal = channelInput[y][x];
          const distFromCenter = Math.hypot(x - gridDim / 2, y - gridDim / 2) / (gridDim / 2);
          const localizedPart = Math.exp(-Math.pow(distFromCenter - 0.25 * (c + 1), 2) * 6.0);
          val = Math.max(0.03, inputVal * 0.62 + localizedPart * 0.38 + Math.sin(stepIndex * 1.4 + c) * 0.08);
        } else {
          // Deepest Layers / Attention / Classifier Head: Highly discriminative token attention hotspots
          const inputVal = channelInput[y][x];
          const attentionCenter = inputVal > 0.35 ? 0.94 : 0.05;
          const tokenPerturbation = Math.sin((x * 2.3 + y * 1.9 + c * 3 + stepIndex) * 0.7) * 0.08;
          val = Math.max(0.02, Math.min(0.98, attentionCenter + tokenPerturbation));
        }

        row.push(Math.max(0.02, Math.min(0.98, val)));
      }
      grid.push(row);
    }
    slices.push(grid);
  }

  // Channel color and title metadata
  const channelNames = datasetName === "CIFAR-10" && stepIndex === 0
    ? ["Red Channel (R)", "Green Channel (G)", "Blue Channel (B)", "Alpha / Mix"]
    : stepIndex === 1
      ? ["Horizontal Gradients", "Vertical Gradients", "Diagonal Contours", "Ridge Contrast"]
      : stepIndex === 2
        ? ["Corner Junctions", "Curvature & Contours", "Texture Harmonics", "Sparsity & Contrast"]
        : stepIndex <= 5
          ? [`Mid Motif ${1}`, `Structural Part ${2}`, `Spatial Filter ${3}`, `Contour Part ${4}`]
          : stepIndex <= 8
            ? [`Semantic Part ${1}`, `Object Feature ${2}`, `Invariant Node ${3}`, `Receptive Cluster ${4}`]
            : [`Attention Focus ${1}`, `Discriminative Token ${2}`, `Class Representation ${3}`, `Semantic Saliency ${4}`];

  return {
    kind: "image",
    slices,
    channelNames,
    previewChannels,
    totalChannels: shape.c,
    spatial: `${shape.h}×${shape.w}`,
    classLabel,
    selectedClass: validClass,
  };
}

