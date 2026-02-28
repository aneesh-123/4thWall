"use client";

import { useState, useRef, useEffect } from "react";
import { chatTurn, getHint } from "@/lib/api";
import type { HintResponse } from "@/lib/types";

interface Message {
  role: "user" | "tutor";
  text: string;
}

export default function LearnPage() {
  const [learnerId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [turnNumber, setTurnNumber] = useState(1);
  const [loading, setLoading] = useState(false);
  const [topic, setTopic] = useState("");
  const [hint, setHint] = useState<HintResponse | null>(null);
  const [hintLevel, setHintLevel] = useState(1);
  const [previousHints, setPreviousHints] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", text }]);
    setLoading(true);
    setHint(null);

    try {
      const res = await chatTurn({
        learner_id: learnerId,
        turn_number: turnNumber,
        user_input: text,
        topic: topic || undefined,
      });
      setMessages((prev) => [...prev, { role: "tutor", text: res.reply }]);
      setTurnNumber(turnNumber + 1);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "tutor", text: `Error: ${err instanceof Error ? err.message : "Something went wrong"}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function handleHint() {
    if (loading) return;
    setLoading(true);
    try {
      const lastUserMsg = [...messages].reverse().find((m) => m.role === "user")?.text || "";
      const res = await getHint(lastUserMsg || "general help", hintLevel, topic, "", previousHints);
      setHint(res);
      setPreviousHints((prev) => [...prev, res.hint_text]);
      if (res.can_request_next) {
        setHintLevel(hintLevel + 1);
      }
    } catch {
      setHint({ level: hintLevel, hint_text: "Could not generate a hint right now.", can_request_next: false, topic });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-3.5rem)]">
      {/* Topic bar */}
      <div className="border-b border-border p-3 flex gap-2 items-center">
        <label className="text-sm text-muted">Topic:</label>
        <input
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g. binary search trees"
          className="flex-1 max-w-xs rounded-md border border-border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
        />
        <button
          onClick={handleHint}
          disabled={loading || messages.length === 0}
          className="ml-auto rounded-md bg-warning/20 px-3 py-1.5 text-sm font-medium text-warning hover:bg-warning/30 disabled:opacity-40 transition-colors"
        >
          Hint (Level {hintLevel})
        </button>
      </div>

      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-muted mt-20">
            <p className="text-lg font-medium mb-2">Start a conversation</p>
            <p className="text-sm">Set a topic above, then ask a question to begin learning.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[75%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-primary text-white"
                  : "bg-card border border-border"
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-card border border-border rounded-xl px-4 py-2.5 text-sm text-muted animate-pulse">
              Thinking...
            </div>
          </div>
        )}

        {hint && (
          <div className="mx-auto max-w-md rounded-xl border border-warning/30 bg-warning/5 p-4 text-sm">
            <div className="font-medium text-warning mb-1">Hint — Level {hint.level}</div>
            <div>{hint.hint_text}</div>
            {hint.can_request_next && (
              <button
                onClick={handleHint}
                className="mt-2 text-xs text-warning underline hover:no-underline"
              >
                Need more help?
              </button>
            )}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border p-3">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex gap-2"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your answer or question..."
            className="flex-1 rounded-md border border-border bg-background px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-md bg-primary px-5 py-2 text-sm font-medium text-white hover:bg-primary-light disabled:opacity-40 transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
