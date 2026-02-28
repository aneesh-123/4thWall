import type {
  ChatTurnRequest,
  ChatTurnResponse,
  GradeRequest,
  GradeResult,
  HintResponse,
  RecommendationResponse,
  TestData,
  TestResponse,
  TestResult,
  KnowledgeGraph,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5002";

async function apiFetch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

// ── Chat ────────────────────────────────────────────────────────────────────

export async function chatTurn(req: ChatTurnRequest): Promise<ChatTurnResponse> {
  return apiFetch<ChatTurnResponse>("/chat/turn", req);
}

// ── Grading ─────────────────────────────────────────────────────────────────

export async function gradeAnswer(req: GradeRequest): Promise<GradeResult> {
  return apiFetch<GradeResult>("/api/grade", req);
}

// ── Tests ───────────────────────────────────────────────────────────────────

export async function generateTest(
  topic: string,
  numQuestions: number = 5,
  learnerId?: string,
): Promise<TestData> {
  return apiFetch<TestData>("/api/test/generate", {
    topic,
    num_questions: numQuestions,
    learner_id: learnerId,
  });
}

export async function submitTest(
  test: TestData,
  responses: TestResponse[],
  learnerId?: string,
): Promise<TestResult> {
  return apiFetch<TestResult>("/api/test/submit", {
    test,
    responses,
    learner_id: learnerId,
  });
}

// ── Hints ───────────────────────────────────────────────────────────────────

export async function getHint(
  questionText: string,
  level: number = 1,
  topic?: string,
  studentAnswer?: string,
  previousHints?: string[],
): Promise<HintResponse> {
  return apiFetch<HintResponse>("/api/hint", {
    question_text: questionText,
    level,
    topic,
    student_answer: studentAnswer,
    previous_hints: previousHints,
  });
}

export async function getAllHints(
  questionText: string,
  topic?: string,
  studentAnswer?: string,
): Promise<{ hints: HintResponse[] }> {
  return apiFetch<{ hints: HintResponse[] }>("/api/hint/all", {
    question_text: questionText,
    topic,
    student_answer: studentAnswer,
  });
}

// ── Recommendations ─────────────────────────────────────────────────────────

export async function getRecommendations(
  learnerId: string,
  topN: number = 5,
): Promise<RecommendationResponse> {
  return apiFetch<RecommendationResponse>("/api/recommend", {
    learner_id: learnerId,
    top_n: topN,
  });
}

// ── Knowledge Graph ─────────────────────────────────────────────────────────

export async function getKnowledgeGraph(learnerId: string): Promise<KnowledgeGraph> {
  const res = await fetch(`${BASE_URL}/api/knowledge-graph/${learnerId}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch knowledge graph: ${res.status}`);
  }
  return res.json();
}
