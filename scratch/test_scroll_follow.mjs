import assert from "node:assert";

console.log("=== Testing Synchronous Scroll-Spy & Floating Inspector Logic ===");

// 1. Test target layer resolution under various scroll positions
const mockCards = [
  { id: 901, top: 120, bottom: 200 },
  { id: 902, top: 220, bottom: 300 },
  { id: 903, top: 320, bottom: 400 },
  { id: 904, top: 420, bottom: 500 },
  { id: 905, top: 520, bottom: 600 },
  { id: 906, top: 620, bottom: 700 },
  { id: 907, top: 720, bottom: 800 },
  { id: 908, top: 820, bottom: 900 },
];

function resolveActiveLayer({ scrollY, innerHeight, scrollHeight, cards }) {
  if (scrollY < 50) {
    return cards[0].id;
  }
  if (innerHeight + scrollY >= scrollHeight - 50) {
    return cards[cards.length - 1].id;
  }

  const targetY = innerHeight * 0.38;
  let minDistance = Infinity;
  let closestId = null;

  for (const card of cards) {
    if (card.bottom > 60 && card.top < innerHeight - 60) {
      const dist = Math.abs(card.top - targetY);
      if (dist < minDistance) {
        minDistance = dist;
        closestId = card.id;
      }
    }
  }

  return closestId;
}

// Case 1: At top of page
assert.strictEqual(
  resolveActiveLayer({ scrollY: 10, innerHeight: 900, scrollHeight: 2500, cards: mockCards }),
  901,
  "At top of page (scrollY < 50), layer 901 must be active"
);

// Case 2: At bottom of page
assert.strictEqual(
  resolveActiveLayer({ scrollY: 1560, innerHeight: 900, scrollHeight: 2500, cards: mockCards }),
  908,
  "At bottom of page, last layer (908) must be active"
);

// Case 3: Middle scroll spy
// Target Y = 900 * 0.38 = 342. Card 903 (top: 320) is closest (|320 - 342| = 22 vs |220 - 342| = 122).
assert.strictEqual(
  resolveActiveLayer({ scrollY: 400, innerHeight: 900, scrollHeight: 2500, cards: mockCards }),
  903,
  "In mid-scroll, the card nearest 38% viewport height must be active"
);

// 2. Test Inspector Offset Calculation & Boundary Clamping
function calculateInspectorOffset({ selectedTop, workspaceTop, workspaceHeight, inspectorHeight }) {
  const rawOffset = selectedTop - workspaceTop - 12;
  const maxOffset = Math.max(0, workspaceHeight - inspectorHeight - 24);
  return Math.round(Math.max(0, Math.min(rawOffset, maxOffset)));
}

// Case 4: First layer at workspace top
const offset1 = calculateInspectorOffset({
  selectedTop: 80,
  workspaceTop: 70,
  workspaceHeight: 2000,
  inspectorHeight: 500,
});
assert.strictEqual(offset1, 0, "Top layer offset clamped to >= 0");

// Case 5: Layer 4 in the middle
const offset2 = calculateInspectorOffset({
  selectedTop: 450,
  workspaceTop: 70,
  workspaceHeight: 2000,
  inspectorHeight: 500,
});
assert.strictEqual(offset2, 368, "Layer 4 offset calculated accurately (450 - 70 - 12 = 368)");

// Case 6: Bottom boundary clamping
const offset3 = calculateInspectorOffset({
  selectedTop: 1950,
  workspaceTop: 70,
  workspaceHeight: 2000,
  inspectorHeight: 500,
});
assert.strictEqual(offset3, 1476, "Bottom boundary clamped to workspaceHeight - inspectorHeight - 24 (2000 - 500 - 24 = 1476)");

console.log("All scroll-spy and floating inspector tests passed successfully!");
