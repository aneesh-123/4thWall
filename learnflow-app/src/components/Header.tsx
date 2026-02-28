"use client";

import Link from "next/link";

export default function Header() {
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold text-lg">
          <span className="text-primary">Learn</span>
          <span>Flow</span>
        </Link>

        <nav className="hidden md:flex items-center gap-6 text-sm">
          <Link href="/learn" className="hover:text-primary transition-colors">
            Learn
          </Link>
          <Link href="/test" className="hover:text-primary transition-colors">
            Test
          </Link>
          <Link href="/dashboard" className="hover:text-primary transition-colors">
            Dashboard
          </Link>
        </nav>

        {/* Mobile menu button */}
        <button className="md:hidden p-2 rounded hover:bg-card" aria-label="Menu">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
            <path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" strokeWidth="1.5" fill="none" />
          </svg>
        </button>
      </div>
    </header>
  );
}
