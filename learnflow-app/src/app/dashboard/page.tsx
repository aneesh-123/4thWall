"use client";

import { useState } from "react";
import { getRecommendations } from "@/lib/api";
import type { RecommendationResponse } from "@/lib/types";

export default function DashboardPage() {
  const [learnerId, setLearnerId] = useState("");
  const [recs, setRecs] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleLoad() {
    if (!learnerId.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await getRecommendations(learnerId);
      setRecs(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load recommendations");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>

      <div className="flex gap-2 mb-8">
        <input
          type="text"
          value={learnerId}
          onChange={(e) => setLearnerId(e.target.value)}
          placeholder="Enter your learner ID"
          className="flex-1 rounded-md border border-border bg-background px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
        />
        <button
          onClick={handleLoad}
          disabled={loading || !learnerId.trim()}
          className="rounded-md bg-primary px-5 py-2 text-sm font-medium text-white hover:bg-primary-light disabled:opacity-40 transition-colors"
        >
          {loading ? "Loading..." : "Load"}
        </button>
      </div>

      {error && <p className="text-sm text-error mb-4">{error}</p>}

      {recs && (
        <>
          {recs.all_mastered ? (
            <div className="rounded-xl border border-success/30 bg-success/5 p-6 text-center">
              <div className="text-2xl mb-2">All concepts mastered!</div>
              <p className="text-sm text-muted">Great work. Consider reviewing with extension-level questions.</p>
            </div>
          ) : recs.recommendations.length === 0 ? (
            <div className="rounded-xl border border-border p-6 text-center text-muted">
              No recommendations yet. Start a learning session to build your profile.
            </div>
          ) : (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">Study Recommendations</h2>
              {recs.recommendations.map((rec) => {
                const masteryPct = Math.round(rec.mastery * 100);
                const barColor =
                  masteryPct >= 60 ? "bg-success" : masteryPct >= 30 ? "bg-warning" : "bg-error";

                return (
                  <div key={rec.concept_id} className="rounded-xl border border-border p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-sm">{rec.concept_id.replace(/_/g, " ")}</span>
                      <span className="text-xs px-2 py-0.5 rounded bg-card border border-border">
                        {rec.difficulty}
                      </span>
                    </div>

                    {/* Mastery bar */}
                    <div className="h-2 rounded-full bg-border mb-2">
                      <div className={`h-2 rounded-full ${barColor} transition-all`} style={{ width: `${masteryPct}%` }} />
                    </div>
                    <div className="flex justify-between text-xs text-muted mb-2">
                      <span>Mastery: {masteryPct}%</span>
                      <span>Focus: {rec.focus_pct}%</span>
                    </div>

                    <p className="text-xs text-muted">{rec.reason}</p>

                    {rec.misconception_hints.length > 0 && (
                      <div className="mt-2 text-xs text-warning">
                        {rec.misconception_hints.map((h, i) => (
                          <div key={i}>{h}</div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
