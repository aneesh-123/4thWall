// ── Learner ─────────────────────────────────────────────────────────────────

export interface Learner {
  learner_id: string;
  name?: string;
}

// ── Chat ────────────────────────────────────────────────────────────────────

export interface ChatTurnRequest {
  learner_id: string;
  turn_number: number;
  user_input: string;
  doc_id?: string;
  topic?: string;
}

export interface ChatTurnResponse {
  reply: string;
  turn_number: number;
  learner_id: string;
  mastery_deltas?: Record<string, { before: number; after: number }>;
  recommendations?: {
    next_concepts: RecommendationItem[];
    next_template: string;
  };
}

// ── Test ────────────────────────────────────────────────────────────────────

export interface TestQuestion {
  id: string;
  type: "multiple_choice" | "true_false" | "fill_blank" | "short_answer" | "explain" | "code_write";
  difficulty: string;
  text: string;
  concept_id: string;
  options?: string[];
  correct_index?: number;
  correct_answer?: boolean | string;
  accept_variations?: string[];
  rubric_hint?: string;
  explanation?: string;
  language?: string;
}

export interface TestData {
  test_id: string;
  topic: string;
  difficulty: string;
  mastery_at_generation: number;
  questions: TestQuestion[];
  num_questions: number;
  num_objective: number;
  num_subjective: number;
  instructions: string;
  learner_id?: string;
  learner_mastery?: number;
}

export interface TestResponse {
  question_id: string;
  answer: string;
}

export interface QuestionScore {
  question_id: string;
  type: string;
  score: number;
  correct: boolean;
  feedback?: string;
  correct_answer?: number | boolean | string;
  strength?: string;
  outcome?: string;
}

export interface TestResult {
  test_id: string;
  score: number;
  total: number;
  score_pct: number;
  outcome: "correct" | "partial" | "confused" | "incorrect";
  question_scores: QuestionScore[];
  mastery_update?: Record<string, { before: number; after: number }>;
  learner_id?: string;
}

// ── Grading ─────────────────────────────────────────────────────────────────

export interface GradeRequest {
  question_text: string;
  topic: string;
  submission: string;
  mode?: "informal" | "formal";
  learner_id?: string;
  concept_context?: string;
}

export interface CriterionScore {
  name: string;
  score: number;
  max_score: number;
  met: boolean;
  feedback: string;
}

export interface GradeResult {
  criteria_scores: CriterionScore[];
  overall_score: number;
  total_points: number;
  outcome: "correct" | "partial" | "confused" | "incorrect";
  strength: string;
  improvement: string;
  misconceptions_detected: string[];
  sources_used: string[];
  rubric?: {
    criteria: { name: string; weight: number; formal_requirement: string; informal_requirement: string }[];
    sources: { type: string; reference: string }[];
    total_points: number;
  };
  mode?: string;
  mastery_update?: Record<string, { before: number; after: number }>;
}

// ── Hints ───────────────────────────────────────────────────────────────────

export interface HintResponse {
  level: number;
  hint_text: string;
  can_request_next: boolean;
  topic: string;
}

// ── Recommendations ─────────────────────────────────────────────────────────

export interface RecommendationItem {
  concept_id: string;
  mastery: number;
  difficulty: string;
  focus_pct: number;
  reason: string;
  misconception_hints: string[];
  weaknesses: string[];
}

export interface RecommendationResponse {
  recommendations: RecommendationItem[];
  all_mastered: boolean;
  learner_id?: string;
}

// ── Knowledge Graph ─────────────────────────────────────────────────────────

export interface GraphNode {
  id: string;
  mastery: number;
  label?: string;
}

export interface GraphLink {
  source: string;
  target: string;
}

export interface KnowledgeGraph {
  nodes: GraphNode[];
  links: GraphLink[];
}
