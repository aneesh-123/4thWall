import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] px-4">
      <h1 className="text-4xl font-bold tracking-tight sm:text-5xl mb-4">
        Welcome to <span className="text-primary">LearnFlow</span>
      </h1>
      <p className="text-muted text-lg max-w-xl text-center mb-10">
        An AI-powered adaptive tutor that tracks your mastery, generates personalized tests,
        and gives you transparent feedback with progressive hints.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full max-w-2xl">
        <Link
          href="/learn"
          className="flex flex-col items-center gap-3 rounded-xl border border-border p-6 hover:border-primary hover:bg-primary/5 transition-colors"
        >
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
            <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>
          <span className="font-semibold">Start Learning</span>
          <span className="text-sm text-muted text-center">Chat with your AI tutor</span>
        </Link>

        <Link
          href="/test"
          className="flex flex-col items-center gap-3 rounded-xl border border-border p-6 hover:border-primary hover:bg-primary/5 transition-colors"
        >
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
          </svg>
          <span className="font-semibold">Take a Test</span>
          <span className="text-sm text-muted text-center">Adaptive difficulty based on mastery</span>
        </Link>

        <Link
          href="/dashboard"
          className="flex flex-col items-center gap-3 rounded-xl border border-border p-6 hover:border-primary hover:bg-primary/5 transition-colors"
        >
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
            <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <span className="font-semibold">Dashboard</span>
          <span className="text-sm text-muted text-center">Track your progress</span>
        </Link>
      </div>
    </div>
  );
}
