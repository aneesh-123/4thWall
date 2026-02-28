/**
 * mastery.js
 * ──────────────────────────────────────────────────────────────────
 * A headless mastery tracking system with two core functions:
 *
 *   1. updateMastery(result)
 *      Takes a question result and updates subtopic mastery scores.
 *
 *   2. getRecommendations(options?)
 *      Returns a weighted focus breakdown for what to study next.
 *
 * Storage is a plain JS object (masteryStore) you can persist however
 * you like (localStorage, a DB, a file, etc.).
 * ──────────────────────────────────────────────────────────────────
 */

// ─── Mastery Store ────────────────────────────────────────────────────────────
// Each key is a subtopic string. Value is a score from 0.0 to 1.0 and array of weaknesses.
// You can seed this with prior data or start empty.
let masteryStore = {};

// ─── Config ───────────────────────────────────────────────────────────────────
const CONFIG = {
  // How strongly a correct answer pushes mastery up
  correctWeight: 0.15,
  // How strongly an incorrect answer pulls mastery down
  incorrectWeight: 0.12,
  // How much a partial answer moves mastery (multiplied by correctness 0–1)
  partialScale: 0.13,
  // Starting mastery for unseen subtopics
  defaultMastery: 0.0,
  // Floor/ceiling
  min: 0.0,
  max: 1.0,
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
function clamp(val) {
  return Math.max(CONFIG.min, Math.min(CONFIG.max, val));
}

function getScore(subtopic) {
  if (!masteryStore[subtopic]) {
    return CONFIG.defaultMastery;
  }
  return masteryStore[subtopic].score;
}

// ─── 1. updateMastery ─────────────────────────────────────────────────────────
/**
 * Updates mastery scores based on a question attempt.
 *
 * @param {Object} result
 * @param {[subtopic: string]: {"improved": string[], "weak": string[]}} result.subtopics
 *   List of subtopic keys this question covers, areas for the student to improve in and areas they have improved in.
 *   e.g. {"binary_trees.balanced":{"improved": ["understands balance"], "weak": ["calculating tree balance incorrectly, confusing right vs left height"]}}
 *
 * @param {"correct" | "incorrect" | "partial"} result.outcome
 *   Whether the student got it right.
 *
 * @param {number} [result.correctness]
 *   For "partial" outcomes: a 0–1 score of how correct the answer was.
 *   Ignored for "correct" / "incorrect". Defaults to 0.5.
 *
 * @param {number} [result.weight]
 *   Optional multiplier (0–1) for how much this question should count.
 *   Useful for weighting easier vs harder questions. Defaults to 1.
 *
 * @returns {{ [subtopic: string]: { before: number, after: number } }}
 *   A diff showing what changed.
 *
 * Example:
 *   updateMastery({
 *     subtopics: ["binary_trees.balanced", "binary_trees.full"],
 *     outcome: "partial",
 *     correctness: 0.6,
 *   })
 */
function updateMastery({ subtopics, outcome, correctness = 0.5, weight = 1 }) {
  if (!subtopics || subtopics.length === 0) {
    throw new Error("updateMastery: subtopics array is required and must not be empty.");
  }
  if (!["correct", "incorrect", "partial"].includes(outcome)) {
    throw new Error(`updateMastery: outcome must be "correct", "incorrect", or "partial". Got: "${outcome}"`);
  }

  const diff = {};

  for (const subtopic in subtopics) {
    if (!masteryStore[subtopic]) {
        masteryStore[subtopic] = {
            score: CONFIG.defaultMastery,
            weaknesses: []
        };
    }

    const { improved = [], weak = [] } = subtopics[subtopic];

    const before = getScore(subtopic);
    let delta = 0;

    if (outcome === "correct") {
      // Scale the boost down as mastery approaches 1 (diminishing returns)
      delta = CONFIG.correctWeight * (1 - before) * weight;
    } else if (outcome === "incorrect") {
      // Scale the penalty down as mastery approaches 0 (floor protection)
      delta = -CONFIG.incorrectWeight * before * weight;
    } else {
      // Partial: interpolate between correct and incorrect delta
      const upDelta = CONFIG.partialScale * (1 - before) * weight;
      const downDelta = -CONFIG.partialScale * before * weight;
      delta = upDelta * correctness + downDelta * (1 - correctness);
    }

    const after = clamp(before + delta);
    if (after > 0.95 && masteryStore[subtopic].weaknesses.length > 0) {
        after = 0.95
    }
    masteryStore[subtopic].score = after;
    diff[subtopic] = { before: round(before), after: round(after) };

    // Remove fixed weaknesses
    for (const item of improved) {
        const index = masteryStore[subtopic].weaknesses.indexOf(item);
        if (index > -1) {
            masteryStore[subtopic].weaknesses.splice(index, 1);
        }
    }

    // Add new weaknesses (avoid duplicates)
    for (const item of weak) {
        if (!masteryStore[subtopic].weaknesses.includes(item)) {
            masteryStore[subtopic].weaknesses.push(item);
        }
    }
  }

  return diff;
}

// ─── 2. getRecommendations ────────────────────────────────────────────────────
/**
 * Returns a percentage breakdown of how much to focus on each subtopic.
 *
 * Subtopics with lower mastery get higher focus weights.
 * The output always sums to 100%.
 *
 * @param {Object} [options]
 * @param {string[]} [options.subtopics]
 *   Restrict recommendations to a specific list of subtopics.
 *   If omitted, all tracked subtopics are used.
 *
 * @param {string[]} [options.exclude]
 *   Subtopics to exclude from the output (e.g. ones already mastered).
 *
 * @param {number} [options.masteredThreshold]
 *   Subtopics with mastery >= this value are excluded automatically.
 *   Defaults to 1.0 (only perfect mastery is excluded).
 *   Set to e.g. 0.85 to ignore nearly-mastered topics.
 *
 * @param {number} [options.topN]
 *   If set, only return the top N weakest subtopics.
 *
 * @returns {{ [subtopic: string]: string }}
 *   e.g. { "binary_trees.balanced": "68%", "binary_trees.full": "22%", ... }
 *
 * Example:
 *   getRecommendations({ subtopics: ["binary_trees.balanced", "binary_trees.full", "binary_trees.complete"] })
 *   // → { "binary_trees.balanced": "71%", "binary_trees.full": "19%", "binary_trees.complete": "10%" }
 */
function getRecommendations({
  subtopics,
  exclude = [],
  masteredThreshold = 1.0,
  topN,
} = {}) {

  let candidates = subtopics ?? Object.keys(masteryStore);

  if (candidates.length === 0) {
    return {};
  }

  candidates = candidates.filter((s) => {
    if (exclude.includes(s)) return false;
    if (!masteryStore[s]) return false;
    if (masteryStore[s].score >= masteredThreshold) return false;
    return true;
  });

  if (candidates.length === 0) {
    return { mastered: true };
  }

  const EPSILON = 0.05;

  let weighted = candidates.map((s) => {
    const mastery = masteryStore[s].score;
    const focusWeight = (1 - mastery) ** 3 + EPSILON;

    return {
      subtopic: s,
      mastery,
      weaknesses: masteryStore[s].weaknesses ?? [],
      focusWeight,
    };
  });

  weighted.sort((a, b) => a.mastery - b.mastery);

  if (topN && topN > 0) {
    weighted = weighted.slice(0, topN);
  }

  const totalWeight = weighted.reduce((sum, w) => sum + w.focusWeight, 0);

  const result = {};

  for (const { subtopic, focusWeight, mastery, weaknesses } of weighted) {
    result[subtopic] = {
      focus: `${Math.round((focusWeight / totalWeight) * 100)}%`,
      mastery: round(mastery),
      weaknesses: [...weaknesses], // shallow copy for safety
    };
  }

  return result;
}

// ─── Utils ────────────────────────────────────────────────────────────────────
function round(n, dp = 3) {
  return Math.round(n * 10 ** dp) / 10 ** dp;
}

/** Get the raw mastery store (for persistence) */
function exportStore() {
  return { ...masteryStore };
}

/** Load a previously saved mastery store */
function importStore(data) {
  masteryStore = { ...data };
}

/** Get mastery score for a single subtopic (0–1) */
function getMastery(subtopic) {
  return round(getScore(subtopic));
}

// ─── Exports ──────────────────────────────────────────────────────────────────
module.exports = { updateMastery, getRecommendations, exportStore, importStore, getMastery };


// ══════════════════════════════════════════════════════════════════════════════
// USAGE EXAMPLES (remove in production)
// ══════════════════════════════════════════════════════════════════════════════



// const { updateMastery, getRecommendations, exportStore, importStore } = require('./mastery');

// ── Example 1: Student answers a question about balanced + full binary trees
//    They got it partially right (60% correct)
console.log(updateMastery({
  subtopics: {"binary_trees.balanced": {improved: [], weak: ["doesn't understand which ranges are balanced"]}, "binary_trees.full": {improved: [], weak: ["incorrectly understanding of if tree was full or not"]}},
  outcome: "partial",
  correctness: 0.6
}));



// ── Example 2: Student nails a question covering complete + perfect trees
updateMastery({ subtopics: {"binary_trees.complete": {improved: ["fjak"], weak:[]}, "binary_trees.perfect": {improved: [], weak:[]}}, outcome: "correct" });
updateMastery({ subtopics: {"binary_trees.complete": {improved: [], weak:[]}, "binary_trees.perfect": {improved: [], weak:[]}}, outcome: "correct" });


console.log(masteryStore)



// ── Example 3: Get recommendations for a specific topic
console.log(getRecommendations({
  subtopics: [
    "binary_trees.balanced",
    "binary_trees.full",
    "binary_trees.complete",
    "binary_trees.perfect",
  ]
}));
// → {
//     "binary_trees.balanced": "38%",
//     "binary_trees.full":     "33%",
//     "binary_trees.complete": "16%",
//     "binary_trees.perfect":  "13%"
//   }




// ── Example 5: Only return top 3 weakest subtopics
console.log(getRecommendations({ topN: 3 }));

/*
// ── Example 6: Persist and restore
const saved = exportStore();
// ... save to DB/file/localStorage ...
importStore(saved); // restore later

*/