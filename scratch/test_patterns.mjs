// Test procedural image generation for MNIST, Fashion-MNIST, CIFAR-10
function generateInputPattern(datasetName, classIdx, gridDim, channel = 0) {
  const grid = [];
  const cx = gridDim / 2;
  const cy = gridDim / 2;

  for (let y = 0; y < gridDim; y++) {
    const row = [];
    const ny = (y - cy) / (gridDim / 2); // -1 to 1

    for (let x = 0; x < gridDim; x++) {
      const nx = (x - cx) / (gridDim / 2); // -1 to 1
      let val = 0.05;

      if (datasetName === "MNIST") {
        switch (classIdx) {
          case 0: { // Oval ring
            const r = Math.hypot(nx * 1.1, ny);
            val = Math.abs(r - 0.65) < 0.22 ? 0.95 : 0.05;
            break;
          }
          case 1: { // Vertical stroke
            const isStem = Math.abs(nx - 0.05) < 0.18 && ny >= -0.75 && ny <= 0.75;
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
            const top = ny < 0 && Math.hypot(nx, ny + 0.35) < 0.42 && Math.hypot(nx, ny + 0.35) > 0.2 && nx > -0.2;
            const bot = ny >= 0 && Math.hypot(nx, ny - 0.35) < 0.45 && Math.hypot(nx, ny - 0.35) > 0.2 && nx > -0.25;
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
        // CIFAR-10 (with channel-specific RGB modulation: channel 0=R, 1=G, 2=B)
        switch (classIdx) {
          case 0: { // Airplane
            const fuselage = Math.abs(ny) < 0.18 && nx > -0.75 && nx < 0.75;
            const wings = Math.abs(nx + 0.1) < 0.22 && Math.abs(ny) < 0.75;
            const tail = nx < -0.55 && Math.abs(ny) < 0.45;
            const isPlane = fuselage || wings || tail;
            if (isPlane) {
              val = channel === 0 ? 0.95 : channel === 1 ? 0.92 : 0.98; // bright silver/white
            } else {
              val = channel === 2 ? 0.75 : channel === 1 ? 0.45 : 0.15; // sky blue
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
              val = channel === 0 ? 0.95 : channel === 1 ? 0.35 : 0.3; // sporty red
            } else if (wheel1 || wheel2) {
              val = 0.15; // dark rubber wheels
            } else {
              val = channel === 2 ? 0.4 : 0.2; // road/asphalt
            }
            break;
          }
          case 2: { // Bird
            const body = Math.hypot(nx, ny) < 0.42;
            const head = Math.hypot(nx - 0.35, ny + 0.35) < 0.25;
            const wing = nx < 0 && ny < 0.2 && Math.abs(nx + ny) < 0.5;
            const isBird = body || head || wing;
            val = isBird ? (channel === 1 ? 0.92 : channel === 0 ? 0.75 : 0.3) : 0.15;
            break;
          }
          case 3: { // Cat
            const head = Math.hypot(nx, ny - 0.05) < 0.46;
            const earL = nx > -0.45 && nx < -0.15 && ny > -0.65 && ny < -0.25 && (nx + 0.45) * 1.5 > (-ny - 0.25);
            const earR = nx > 0.15 && nx < 0.45 && ny > -0.65 && ny < -0.25 && (-nx + 0.45) * 1.5 > (-ny - 0.25);
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
            val = isHorse ? (channel === 0 ? 0.65 : channel === 1 ? 0.45 : 0.28) : 0.12;
            break;
          }
          case 8: { // Ship
            const hull = ny >= 0.15 && ny < 0.55 && Math.abs(nx) < (0.75 - (ny - 0.15) * 0.3);
            const mast = Math.abs(nx + 0.1) < 0.08 && ny > -0.65 && ny < 0.15;
            const isShip = hull || mast;
            const isSea = ny >= 0.55;
            if (isShip) {
              val = channel === 0 ? 0.85 : 0.4;
            } else if (isSea) {
              val = channel === 2 ? 0.88 : channel === 1 ? 0.55 : 0.15; // deep sea blue
            } else {
              val = channel === 2 ? 0.55 : 0.35; // sky
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

      row.push(Math.max(0, Math.min(1, val)));
    }
    grid.push(row);
  }
  return grid;
}

console.log("Testing generation for all classes...");
for (let c = 0; c < 10; c++) {
  const g1 = generateInputPattern("MNIST", c, 16);
  const g2 = generateInputPattern("Fashion-MNIST", c, 16);
  const g3 = generateInputPattern("CIFAR-10", c, 16, 0);
  const sum1 = g1.flat().reduce((a, b) => a + b, 0);
  const sum2 = g2.flat().reduce((a, b) => a + b, 0);
  const sum3 = g3.flat().reduce((a, b) => a + b, 0);
  console.log(`Class ${c}: MNIST sum=${sum1.toFixed(1)}, Fashion sum=${sum2.toFixed(1)}, CIFAR sum=${sum3.toFixed(1)}`);
}
