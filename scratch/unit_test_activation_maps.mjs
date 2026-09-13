import assert from "node:assert/strict";
import {
  generateInputPattern,
  generateSyntheticFeatureMaps,
  calculateLayerParams,
  calculateLayerFlops,
} from "../src/utils/architectureAnalytics.js";
import { MODEL_PRESETS } from "../src/data/modelPresets.js";

const DATASETS = {
  MNIST: {
    name: "MNIST",
    channels: 1,
    size: 28,
    classes: 10,
    classesList: ["0 - Zero", "1 - One", "2 - Two", "3 - Three", "4 - Four", "5 - Five", "6 - Six", "7 - Seven", "8 - Eight", "9 - Nine"],
  },
  "Fashion-MNIST": {
    name: "Fashion-MNIST",
    channels: 1,
    size: 28,
    classes: 10,
    classesList: ["T-shirt / top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
  },
  "CIFAR-10": {
    name: "CIFAR-10",
    channels: 3,
    size: 32,
    classes: 10,
    classesList: ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"],
  },
};

console.log("=================================================");
console.log("UNIT TEST: Dynamic Activation Maps & Dry Run Flow");
console.log("=================================================\n");

let testsPassed = 0;
let testsTotal = 0;

function runTest(name, fn) {
  testsTotal++;
  try {
    fn();
    console.log(`✅ PASS: ${name}`);
    testsPassed++;
  } catch (err) {
    console.error(`❌ FAIL: ${name}`);
    console.error(err);
  }
}

// -------------------------------------------------------------
// Test 1: Input pattern changes across all 10 classes in MNIST
// -------------------------------------------------------------
runTest("MNIST: All 10 classes generate distinct, non-zero input patterns", () => {
  const patterns = [];
  for (let c = 0; c < 10; c++) {
    const grid = generateInputPattern("MNIST", c, 16, 0);
    assert.equal(grid.length, 16, "Grid height must be 16");
    assert.equal(grid[0].length, 16, "Grid width must be 16");

    let sum = 0;
    for (const r of grid) {
      for (const v of r) {
        sum += v;
      }
    }
    assert.ok(sum > 1.0, `Class ${c} pattern must have significant non-zero activations (got sum=${sum})`);
    patterns.push(grid);
  }

  // Verify all pairs of classes produce distinct matrices
  for (let i = 0; i < 10; i++) {
    for (let j = i + 1; j < 10; j++) {
      let diff = 0;
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          diff += Math.abs(patterns[i][y][x] - patterns[j][y][x]);
        }
      }
      assert.ok(diff > 2.0, `Classes ${i} and ${j} must have distinctly different patterns (diff=${diff})`);
    }
  }
});

// -------------------------------------------------------------------
// Test 2: Input pattern changes across all 10 classes in Fashion-MNIST
// -------------------------------------------------------------------
runTest("Fashion-MNIST: All 10 classes generate distinct, non-zero input patterns", () => {
  const patterns = [];
  for (let c = 0; c < 10; c++) {
    const grid = generateInputPattern("Fashion-MNIST", c, 16, 0);
    let sum = 0;
    for (const r of grid) {
      for (const v of r) sum += v;
    }
    assert.ok(sum > 1.0, `Fashion-MNIST Class ${c} pattern must have activations (sum=${sum})`);
    patterns.push(grid);
  }

  for (let i = 0; i < 10; i++) {
    for (let j = i + 1; j < 10; j++) {
      let diff = 0;
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          diff += Math.abs(patterns[i][y][x] - patterns[j][y][x]);
        }
      }
      assert.ok(diff > 2.0, `Fashion classes ${i} and ${j} must be distinct (diff=${diff})`);
    }
  }
});

// -------------------------------------------------------------------
// Test 3: Input pattern changes across all 10 classes in CIFAR-10 with RGB
// -------------------------------------------------------------------
runTest("CIFAR-10: Multi-channel RGB input maps differ by channel & class", () => {
  for (let c = 0; c < 10; c++) {
    const rMap = generateInputPattern("CIFAR-10", c, 16, 0);
    const gMap = generateInputPattern("CIFAR-10", c, 16, 1);
    const bMap = generateInputPattern("CIFAR-10", c, 16, 2);

    let diffRG = 0;
    let diffRB = 0;
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        diffRG += Math.abs(rMap[y][x] - gMap[y][x]);
        diffRB += Math.abs(rMap[y][x] - bMap[y][x]);
      }
    }
    assert.ok(diffRG > 0.5, `Class ${c} R and G channels must show distinct chromatic distribution`);
    assert.ok(diffRB > 0.5, `Class ${c} R and B channels must show distinct chromatic distribution`);
  }
});

// -------------------------------------------------------------------
// Test 4: generateSyntheticFeatureMaps responds to selectedClass at step 0
// -------------------------------------------------------------------
runTest("Dry Run Step 0: Switching selectedClass alters activation heatmaps", () => {
  const dataset = DATASETS["CIFAR-10"];
  const shape = { kind: "image", c: 3, h: 32, w: 32 };

  // Compare Class 3 (Cat) vs Class 0 (Airplane)
  const catFeatures = generateSyntheticFeatureMaps(0, 10, shape, dataset, 3);
  const planeFeatures = generateSyntheticFeatureMaps(0, 10, shape, dataset, 0);

  assert.equal(catFeatures.kind, "image");
  assert.equal(planeFeatures.kind, "image");
  assert.equal(catFeatures.classLabel, "Cat");
  assert.equal(planeFeatures.classLabel, "Airplane");
  assert.equal(catFeatures.slices.length, 3, "3 preview slices for 3-channel input");

  // Sum of differences between Cat and Airplane
  let totalDelta = 0;
  for (let s = 0; s < 3; s++) {
    const sliceA = catFeatures.slices[s];
    const sliceB = planeFeatures.slices[s];
    for (let y = 0; y < sliceA.length; y++) {
      for (let x = 0; x < sliceA[0].length; x++) {
        totalDelta += Math.abs(sliceA[y][x] - sliceB[y][x]);
      }
    }
  }

  assert.ok(totalDelta > 15.0, `Changing class from Airplane to Cat must significantly alter activation map (delta=${totalDelta})`);
});

// -------------------------------------------------------------------
// Test 5: Early conv layers transform activations per class
// -------------------------------------------------------------------
runTest("Dry Run Step 1 (Early Conv): Edge filters vary across classes", () => {
  const dataset = DATASETS["MNIST"];
  const shape = { kind: "image", c: 32, h: 28, w: 28 };

  const digit0 = generateSyntheticFeatureMaps(1, 8, shape, dataset, 0);
  const digit1 = generateSyntheticFeatureMaps(1, 8, shape, dataset, 1);

  let delta = 0;
  for (let s = 0; s < digit0.slices.length; s++) {
    for (let y = 0; y < digit0.slices[s].length; y++) {
      for (let x = 0; x < digit0.slices[s][0].length; x++) {
        delta += Math.abs(digit0.slices[s][y][x] - digit1.slices[s][y][x]);
      }
    }
  }

  assert.ok(delta > 5.0, `Early conv activations must reflect edge differences between digit 0 and digit 1 (delta=${delta})`);
});

// -------------------------------------------------------------------
// Test 5b: Inter-layer evolution across successive Conv2d layers
// -------------------------------------------------------------------
runTest("Inter-layer Evolution: Feature activations change between successive Conv layers (Step 1 -> Step 2)", () => {
  const dataset = DATASETS["CIFAR-10"];
  const shape1 = { kind: "image", c: 64, h: 32, w: 32 };
  const shape2 = { kind: "image", c: 128, h: 16, w: 16 };

  // Step 1: Early edge detection
  const step1Features = generateSyntheticFeatureMaps(1, 8, shape1, dataset, 3);
  // Step 2: Second conv (corner junctions & curvature)
  const step2Features = generateSyntheticFeatureMaps(2, 8, shape2, dataset, 3);

  assert.notEqual(step1Features.channelNames[0], step2Features.channelNames[0], "Channel names must reflect hierarchical level");

  let layerDelta = 0;
  const slice1 = step1Features.slices[0];
  const slice2 = step2Features.slices[0];
  const minDim = Math.min(slice1.length, slice2.length);

  for (let y = 0; y < minDim; y++) {
    for (let x = 0; x < minDim; x++) {
      layerDelta += Math.abs(slice1[y][x] - slice2[y][x]);
    }
  }

  assert.ok(layerDelta > 4.0, `Step 1 and Step 2 activations must distinctly evolve (layerDelta=${layerDelta})`);
});

// -------------------------------------------------------------------
// Test 6: Final layer logits predict chosen class with >85% confidence
// -------------------------------------------------------------------
runTest("Dry Run Final Step: Softmax logits correctly predict selected class with >85% confidence", () => {
  const vectorShape = { kind: "vector", n: 10 };

  for (const [dsName, dataset] of Object.entries(DATASETS)) {
    for (let c = 0; c < 10; c++) {
      const result = generateSyntheticFeatureMaps(10, 10, vectorShape, dataset, c);
      assert.equal(result.kind, "vector");
      assert.equal(result.targetClass, c, `Dataset ${dsName} Class ${c} targetClass must match`);

      // Find argmax
      let maxIdx = 0;
      let maxVal = -1;
      result.values.forEach((v, idx) => {
        if (v > maxVal) {
          maxVal = v;
          maxIdx = idx;
        }
      });

      assert.equal(maxIdx, c, `Argmax prediction (${maxIdx}) must equal selected class (${c}) for ${dsName}`);
      assert.ok(maxVal >= 0.85, `Target class confidence must be >= 85% (got ${(maxVal * 100).toFixed(1)}%)`);

      // Check sum to 1.0 (valid probability distribution)
      const sum = result.values.reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(sum - 1.0) < 0.001, "Probabilities must sum to 1.0");
    }
  }
});

// -------------------------------------------------------------------
// Test 7: Stage names in all presets are non-empty and well-formed
// -------------------------------------------------------------------
runTest("Model Presets: All layers have valid, non-colliding stage names", () => {
  for (const preset of MODEL_PRESETS) {
    const stageNames = new Set();
    for (const layer of preset.layers) {
      assert.ok(layer.stage && layer.stage.trim().length > 0, `Preset ${preset.id} layer ${layer.id} must have stage`);
      stageNames.add(layer.stage);
    }
    assert.ok(stageNames.size >= 2, `Preset ${preset.id} must have multiple stages (got ${stageNames.size})`);
  }
});

console.log(`\n=================================================`);
console.log(`RESULTS: ${testsPassed} / ${testsTotal} tests passed`);
console.log(`=================================================`);

if (testsPassed !== testsTotal) {
  process.exit(1);
}
