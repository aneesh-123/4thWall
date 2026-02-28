"use client";

import { useState } from "react";
import { generateTest, submitTest } from "@/lib/api";
import type { TestData, TestQuestion, TestResponse, TestResult } from "@/lib/types";

export default function TestPage() {
  const [learnerId] = useState(() => crypto.randomUUID());
  const [topic, setTopic] = useState("");
  const [numQuestions, setNumQuestions] = useState(5);
  const [test, setTest] = useState<TestData | null>(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [result, setResult] = useState<TestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleGenerate() {
    if (!topic.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const t = await generateTest(topic, numQuestions, learnerId);
      setTest(t);
      setCurrentIdx(0);
      setAnswers({});
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate test");
    } finally {
      setLoading(false);
    }
  }

  async function handleSubmit() {
    if (!test) return;
    setLoading(true);
    try {
      const responses: TestResponse[] = test.questions.map((q) => ({
        question_id: q.id,
        answer: answers[q.id] || "",
      }));
      const res = await submitTest(test, responses, learnerId);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit test");
    } finally {
      setLoading(false);
    }
  }

  // ── Setup screen ──────────────────────────────────────────────────────────
  if (!test && !result) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[70vh] px-4">
        <h1 className="text-2xl font-bold mb-6">Adaptive Test</h1>
        <div className="w-full max-w-md space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Topic</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. sorting algorithms"
              className="w-full rounded-md border border-border bg-background px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Number of Questions</label>
            <input
              type="number"
              min={1}
              max={10}
              value={numQuestions}
              onChange={(e) => setNumQuestions(Math.max(1, Math.min(10, +e.target.value)))}
              className="w-24 rounded-md border border-border bg-background px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          {error && <p className="text-sm text-error">{error}</p>}
          <button
            onClick={handleGenerate}
            disabled={loading || !topic.trim()}
            className="w-full rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-white hover:bg-primary-light disabled:opacity-40 transition-colors"
          >
            {loading ? "Generating..." : "Generate Test"}
          </button>
        </div>
      </div>
    );
  }

  // ── Result screen ─────────────────────────────────────────────────────────
  if (result) {
    const outcomeColor = {
      correct: "text-success",
      partial: "text-warning",
      confused: "text-warning",
      incorrect: "text-error",
    }[result.outcome];

    return (
      <div className="max-w-2xl mx-auto p-6">
        <h1 className="text-2xl font-bold mb-4">Test Results</h1>
        <div className="rounded-xl border border-border p-6 mb-6">
          <div className="text-3xl font-bold mb-1">{result.score_pct}%</div>
          <div className={`text-sm font-medium ${outcomeColor}`}>
            {result.outcome.charAt(0).toUpperCase() + result.outcome.slice(1)} — {result.score}/{result.total} points
          </div>
        </div>

        <div className="space-y-3">
          {result.question_scores.map((qs, i) => (
            <div key={qs.question_id} className={`rounded-lg border p-4 text-sm ${qs.correct ? "border-success/30 bg-success/5" : "border-error/30 bg-error/5"}`}>
              <div className="font-medium mb-1">
                Q{i + 1}: {qs.correct ? "Correct" : "Incorrect"} ({qs.score}/{1})
              </div>
              {qs.feedback && <div className="text-muted">{qs.feedback}</div>}
            </div>
          ))}
        </div>

        <button
          onClick={() => { setTest(null); setResult(null); setAnswers({}); }}
          className="mt-6 rounded-md bg-primary px-5 py-2 text-sm font-medium text-white hover:bg-primary-light transition-colors"
        >
          Take Another Test
        </button>
      </div>
    );
  }

  // ── Question screen ───────────────────────────────────────────────────────
  const question = test!.questions[currentIdx];

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-lg font-bold">
          Question {currentIdx + 1} of {test!.num_questions}
        </h1>
        <span className="text-xs px-2 py-1 rounded bg-card border border-border">
          {test!.difficulty} — {question.type.replace("_", " ")}
        </span>
      </div>

      <div className="rounded-xl border border-border p-6 mb-6">
        <p className="text-sm leading-relaxed mb-4">{question.text}</p>
        <QuestionInput question={question} value={answers[question.id] || ""} onChange={(v) => setAnswers({ ...answers, [question.id]: v })} />
      </div>

      <div className="flex justify-between">
        <button
          onClick={() => setCurrentIdx(Math.max(0, currentIdx - 1))}
          disabled={currentIdx === 0}
          className="rounded-md border border-border px-4 py-2 text-sm hover:bg-card disabled:opacity-40 transition-colors"
        >
          Previous
        </button>
        {currentIdx < test!.num_questions - 1 ? (
          <button
            onClick={() => setCurrentIdx(currentIdx + 1)}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-light transition-colors"
          >
            Next
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="rounded-md bg-success px-5 py-2 text-sm font-medium text-white hover:bg-success/90 disabled:opacity-40 transition-colors"
          >
            {loading ? "Submitting..." : "Submit Test"}
          </button>
        )}
      </div>
    </div>
  );
}

function QuestionInput({ question, value, onChange }: { question: TestQuestion; value: string; onChange: (v: string) => void }) {
  if (question.type === "multiple_choice" && question.options) {
    return (
      <div className="space-y-2">
        {question.options.map((opt, i) => (
          <label key={i} className={`flex items-center gap-3 rounded-lg border p-3 cursor-pointer transition-colors ${value === String(i) ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"}`}>
            <input type="radio" name="mcq" value={i} checked={value === String(i)} onChange={() => onChange(String(i))} className="accent-primary" />
            <span className="text-sm">{opt}</span>
          </label>
        ))}
      </div>
    );
  }

  if (question.type === "true_false") {
    return (
      <div className="flex gap-3">
        {["true", "false"].map((v) => (
          <button
            key={v}
            onClick={() => onChange(v)}
            className={`flex-1 rounded-lg border p-3 text-sm font-medium capitalize transition-colors ${value === v ? "border-primary bg-primary/5 text-primary" : "border-border hover:border-primary/50"}`}
          >
            {v}
          </button>
        ))}
      </div>
    );
  }

  // fill_blank, short_answer, explain, code_write → textarea
  const isCode = question.type === "code_write";
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={question.type === "fill_blank" ? "Fill in the blank..." : "Write your answer..."}
      rows={isCode ? 8 : 4}
      className={`w-full rounded-md border border-border bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 resize-y ${isCode ? "font-mono" : ""}`}
    />
  );
}
