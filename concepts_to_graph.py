#!/usr/bin/env python3
"""
concepts_to_graph.py — Convert concepts.json into a prerequisite knowledge graph.

Usage:
    python concepts_to_graph.py --input concepts.json --outdir out

Outputs:
    out/graph.json    Nodes + edges as JSON
    out/graph.md      Human-readable dependency listing
    out/graph.png     NetworkX / matplotlib visualisation

No API calls — fully heuristic and deterministic.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

try:
    import networkx as nx
except ImportError:
    sys.exit("ERROR: Install networkx:\n  pip install networkx")

try:
    import matplotlib
    matplotlib.use("Agg")       # headless — works without a display
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    sys.exit("ERROR: Install matplotlib:\n  pip install matplotlib")


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    id: str
    label: str
    description: str
    keywords: list[str]
    topic: str
    difficulty: str          # "easy" | "medium" | "hard"
    pages: list[int]


@dataclass
class Edge:
    from_id: str
    to_id: str
    type: str = "prerequisite"


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty assignment  (keyword heuristic + positional fallback)
# ─────────────────────────────────────────────────────────────────────────────

_EASY_RE = re.compile(
    r"\b(introduc|overview|basic|fundamental|what is|definition|"
    r"concept|background|motivation|histor|review|recall|simple|primer)\b",
    re.I,
)
_HARD_RE = re.compile(
    r"\b(advanced|complex|optim|proof|prove|theorem|lemma|derive|"
    r"formal|analysis|asymptot|worst.case|trade.?off|NP|convergence|"
    r"correctness|generaliz)\b",
    re.I,
)


def assign_difficulty(name: str, notes: str, position: float) -> str:
    """
    position: 0.0 = first subtopic in the PDF, 1.0 = last.
    Keyword match takes priority; ties fall back to thirds by position.
    """
    text = f"{name} {notes}"
    if _EASY_RE.search(text):
        return "easy"
    if _HARD_RE.search(text):
        return "hard"
    if position < 0.33:
        return "easy"
    if position < 0.67:
        return "medium"
    return "hard"


# ─────────────────────────────────────────────────────────────────────────────
# Keyword extraction
# ─────────────────────────────────────────────────────────────────────────────

_STOP = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "could should may might shall can need of in on at by for with about against "
    "between through during before after from up down out if as it its they them "
    "their what which who this that these those we our you your i my and or but "
    "not no so then when where why how all each most other some such too very "
    "just also used use using given well shown often called known section".split()
)


def _extract_keywords(text: str, n: int = 6) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'\-]{2,}", text.lower())
    tokens = [t for t in tokens if t not in _STOP and len(t) > 2]
    return [w for w, _ in Counter(tokens).most_common(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Build nodes from concepts.json
# ─────────────────────────────────────────────────────────────────────────────

def build_nodes(data: dict) -> list[Node]:
    """
    Accepts both output formats from pdf_to_concepts.py:
      • New format: topics[].subtopics[] with {name, notes, pages}
      • Legacy format: topics[].subtopics[].concepts[] with {name, ...}
    """
    all_subtopics: list[tuple[str, dict]] = []   # (topic_name, subtopic_dict)

    for t in data.get("topics", []):
        topic_name = t.get("topic", "Unknown")
        for st in t.get("subtopics", []):
            # New format: subtopic is the leaf node
            if "concepts" not in st:
                all_subtopics.append((topic_name, st))
            else:
                # Legacy format: each concept is a leaf node
                for c in st.get("concepts", []):
                    all_subtopics.append((topic_name, c))

    total = len(all_subtopics)
    nodes: list[Node] = []
    for idx, (topic_name, item) in enumerate(all_subtopics):
        name  = item.get("name", "").strip()
        notes = item.get("notes", item.get("description", "")).strip()
        pages = item.get("pages", [])
        if not name:
            continue

        node_id = f"n{idx:03d}"
        kws     = _extract_keywords(f"{name} {notes}")
        pos     = idx / max(total - 1, 1)
        diff    = assign_difficulty(name, notes, pos)

        nodes.append(Node(
            id=node_id,
            label=name,
            description=notes,
            keywords=kws,
            topic=topic_name,
            difficulty=diff,
            pages=pages,
        ))

    return nodes


# ─────────────────────────────────────────────────────────────────────────────
# Edge inference  (no LLM, deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_overlap(a: Node, b: Node) -> float:
    sa = set(a.keywords) | {w for w in a.label.lower().split() if len(w) > 3}
    sb = set(b.keywords) | {w for w in b.label.lower().split() if len(w) > 3}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _top_k_similar(target: Node, candidates: list[Node], k: int = 2) -> list[Node]:
    scored = [(c, _keyword_overlap(target, c)) for c in candidates if c.id != target.id]
    scored.sort(key=lambda x: -x[1])
    return [c for c, score in scored[:k] if score > 0]


def infer_edges(nodes: list[Node]) -> list[Edge]:
    """
    Prerequisite inference strategy:
    1. Group nodes into easy / medium / hard tiers.
    2. Cross-tier:
         • Each medium node links the top-2 keyword-similar easy nodes as prereqs.
         • Each hard  node links the top-2 keyword-similar medium nodes as prereqs.
         • If a hard node has no medium matches, fall back to top-2 easy.
    3. Within-hard tier: sequential chain by order of appearance.
    4. Transitive reduction removes redundant edges.

    If a tier is empty the next-lower non-empty tier fills in.
    """
    tiers: dict[str, list[Node]] = {"easy": [], "medium": [], "hard": []}
    for n in nodes:
        tiers[n.difficulty].append(n)

    raw_edges: set[tuple[str, str]] = set()

    # Cross-tier: medium ← easy
    for med in tiers["medium"]:
        for prereq in _top_k_similar(med, tiers["easy"]):
            raw_edges.add((prereq.id, med.id))

    # Cross-tier: hard ← medium (or ← easy if no medium)
    for hard in tiers["hard"]:
        sources = _top_k_similar(hard, tiers["medium"])
        if not sources:
            sources = _top_k_similar(hard, tiers["easy"])
        for prereq in sources:
            raw_edges.add((prereq.id, hard.id))

    # Within-hard sequential chain
    for i in range(len(tiers["hard"]) - 1):
        raw_edges.add((tiers["hard"][i].id, tiers["hard"][i + 1].id))

    if not raw_edges:
        return []

    # Build DiGraph and run transitive reduction
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n.id)
    G.add_edges_from(raw_edges)
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    try:
        G = nx.transitive_reduction(G)
    except nx.NetworkXError:
        pass   # cycle-safe fallback: keep raw edges

    return [Edge(from_id=u, to_id=v) for u, v in G.edges()]


# ─────────────────────────────────────────────────────────────────────────────
# Output: graph.json
# ─────────────────────────────────────────────────────────────────────────────

def write_json(nodes: list[Node], edges: list[Edge], path: str) -> None:
    payload = {
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Output: graph.md
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown(nodes: list[Node], edges: list[Edge], path: str) -> None:
    # Map each node to its prerequisite node IDs
    prereqs: dict[str, list[str]] = {n.id: [] for n in nodes}
    for e in edges:
        prereqs[e.to_id].append(e.from_id)

    id_to_node = {n.id: n for n in nodes}
    diff_order = {"easy": 0, "medium": 1, "hard": 2}
    sorted_nodes = sorted(nodes, key=lambda n: (diff_order[n.difficulty], n.topic, n.label))

    out: list[str] = ["# Prerequisite Knowledge Graph\n"]

    current_diff: Optional[str] = None
    for n in sorted_nodes:
        if n.difficulty != current_diff:
            current_diff = n.difficulty
            out.append(f"\n## {n.difficulty.capitalize()} concepts\n")

        out.append(f"### {n.label}")
        if n.description:
            out.append(f"> {n.description}")
        out.append(f"- **Topic:** {n.topic}")
        if n.pages:
            out.append(f"- **Pages:** {', '.join(str(p) for p in n.pages)}")
        if n.keywords:
            out.append(f"- **Keywords:** {', '.join(n.keywords)}")
        if prereqs[n.id]:
            labels = [f"`{id_to_node[pid].label}`" for pid in prereqs[n.id] if pid in id_to_node]
            out.append(f"- **Prerequisites:** {', '.join(labels)}")
        out.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


# ─────────────────────────────────────────────────────────────────────────────
# Visualization: graph.png
# ─────────────────────────────────────────────────────────────────────────────

_DIFF_COLORS = {"easy": "#6fcf97", "medium": "#f2c94c", "hard": "#eb5757"}
_DIFF_LABEL  = {"easy": "Easy",    "medium": "Medium",   "hard": "Hard"}


def _hierarchical_pos(
    G: nx.DiGraph,
    layer_gap: float = 1.8,
    node_gap:  float = 2.2,
) -> Optional[dict]:
    """Top-down layout: each topological generation is a horizontal row."""
    try:
        layers = list(nx.topological_generations(G))
    except nx.NetworkXUnfeasible:
        return None   # cycle — caller falls back to spring layout

    pos: dict = {}
    for depth, layer in enumerate(layers):
        layer = sorted(layer)   # deterministic left-to-right order
        n = len(layer)
        x_start = -(n - 1) * node_gap / 2
        for i, node in enumerate(layer):
            pos[node] = (x_start + i * node_gap, -depth * layer_gap)
    return pos


def visualize(nodes: list[Node], edges: list[Edge], path: str) -> None:
    id_to_node = {n.id: n for n in nodes}

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n.id)
    for e in edges:
        G.add_edge(e.from_id, e.to_id)

    pos = _hierarchical_pos(G) or nx.spring_layout(G, seed=42)

    # Truncate labels to fit inside nodes
    labels = {
        nid: (lab[:22] + "…" if len(lab := id_to_node[nid].label) > 25 else lab)
        for nid in G.nodes()
    }
    node_colors = [_DIFF_COLORS.get(id_to_node[nid].difficulty, "#ccc") for nid in G.nodes()]

    n_nodes = G.number_of_nodes()
    fig_w = max(14, n_nodes * 1.4)
    fig_h = max(9,  fig_w * 0.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("#f8f8f8")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=1800, alpha=0.92,
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=7, font_weight="bold",
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#555", arrows=True,
        arrowstyle="-|>", arrowsize=20,
        connectionstyle="arc3,rad=0.05",
        width=1.4,
        min_source_margin=22, min_target_margin=22,
    )

    legend_patches = [
        mpatches.Patch(color=_DIFF_COLORS[d], label=_DIFF_LABEL[d])
        for d in ("easy", "medium", "hard")
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
              title="Difficulty", title_fontsize=9, framealpha=0.85)

    ax.set_title("Prerequisite Knowledge Graph", fontsize=13, fontweight="bold", pad=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph image saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert concepts.json into a prerequisite knowledge graph."
    )
    parser.add_argument("--input",  required=True,
                        help="Path to concepts.json (output of pdf_to_concepts.py).")
    parser.add_argument("--outdir", default="out",
                        help="Directory for output files (default: out).")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: File not found: {args.input}")

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    print("[1/4] Building nodes…")
    nodes = build_nodes(data)
    if not nodes:
        sys.exit("ERROR: No nodes extracted. Check that concepts.json is non-empty.")
    print(f"      {len(nodes)} node(s) across "
          f"{len({n.topic for n in nodes})} topic(s).")

    print("[2/4] Inferring prerequisite edges…")
    edges = infer_edges(nodes)
    print(f"      {len(edges)} edge(s) after transitive reduction.")

    json_path = os.path.join(args.outdir, "graph.json")
    md_path   = os.path.join(args.outdir, "graph.md")
    png_path  = os.path.join(args.outdir, "graph.png")

    print("[3/4] Writing graph.json and graph.md…")
    write_json(nodes, edges, json_path)
    write_markdown(nodes, edges, md_path)

    print("[4/4] Rendering graph.png…")
    visualize(nodes, edges, png_path)

    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Nodes  : {len(nodes)}")
    print(f"  Edges  : {len(edges)}")
    difficulty_counts = Counter(n.difficulty for n in nodes)
    for d in ("easy", "medium", "hard"):
        print(f"  {d.capitalize():6s} : {difficulty_counts.get(d, 0)}")
    print(sep)
    print(f"  {os.path.relpath(json_path)}")
    print(f"  {os.path.relpath(md_path)}")
    print(f"  {os.path.relpath(png_path)}")
    print(sep)


if __name__ == "__main__":
    main()
