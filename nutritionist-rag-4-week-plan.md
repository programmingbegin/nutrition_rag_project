# Nutritionist RAG Project — 4-Week Plan

**Stack:** Python + LangChain, Chroma (vector store), OpenAI/Claude (LLM), hybrid retrieval (BM25 + vector)

**Use case:** A RAG-powered assistant that gives personalized nutrition/sleep guidance based on a user profile (age, height, weight, diet, sleep hours), grounded in real dietary guideline documents.

**Time commitment assumed:** ~15–20 hrs/week

---

## Architecture Overview

```
User profile (age, height, weight, diet, sleep)
        │
        ├──────────────► Metrics tool (BMI, calorie need — LangChain Tool, deterministic math)
        │
        ├──────────────► Nutrient lookup tool (food → calories/protein/fat/carbs, exact table lookup)
        │
        └──────────────► Retriever (hybrid: BM25 + vector search)
                                  │
                          Vector store (Chroma — chunked guideline docs)
                                  │
                                  ▼
                     LLM generation (personalized, cites sources)
                                  │
                                  ▼
                  Response (advice + sources + disclaimer)
```

Key design decision: user profile data is **not retrieved**, it's structured input. It flows through two deterministic tools — a metrics calculator (BMI, Mifflin-St Jeor calorie formula) and a nutrient lookup table (exact food → macro/calorie values) — rather than being left to the LLM to compute or recall from memory. It's also used to construct the retrieval query so the guidelines pulled back are relevant to that specific profile.

**Why a third tool for nutrient lookup:** food composition data (e.g. "1 large egg has ~70 kcal, 6g protein") is exact tabular data, not something to embed and semantically retrieve — and not something an LLM should be trusted to recall precisely. It belongs in a structured table (SQLite/pandas), queried directly, the same way the metrics tool avoids LLM arithmetic. This is the difference between *retrieval* (fuzzy, semantic, for guideline prose) and *lookup* (exact, tabular, for nutrient facts) — both are "grounding," but they need different mechanisms.

---

## Week 1 — Concepts + Working Basic Pipeline

Learn just enough theory to start, then build in parallel.

- **Days 1–2:** Core concepts — embeddings, chunking strategies, vector search (ANN/HNSW), the retrieve → augment → generate loop, and common failure modes (retrieval misses, context limits, hallucination despite retrieval)
- **Days 2–4:** Collect & clean nutrition guideline documents (Dietary Guidelines for Americans 2025–2030, WHO nutrition/sleep guidelines). Chunk with `RecursiveCharacterTextSplitter` (~500 tokens, 50 overlap). Embed with `text-embedding-3-small` and store in Chroma.
- **Day 4:** Set up the nutrient lookup table — start with a lightweight Kaggle CSV of common foods for quick iteration, with a plan to swap in USDA FNDDS/SR Legacy data once the pipeline works end-to-end.
- **Days 5–7:** Basic retrieval + generation working end-to-end.

**Goal by end of week:** Ask "how much protein does an adult need daily?" and get a grounded, cited answer.

---

## Week 2 — Personalization Layer

The differentiator of this project — give it the most room while momentum is fresh.

- Build the profile schema (age, height, weight, diet, sleep hours)
- Build the metrics tool (BMI calculator, Mifflin-St Jeor calorie equation) as a **LangChain Tool**, not LLM-computed math
- Build the nutrient lookup tool (food name + serving size → calories/protein/fat/carbs from the SQLite/pandas table), also as a **LangChain Tool**, not LLM-recalled data
- Wire up a LangChain agent combining the metrics tool + nutrient lookup tool + retriever
- Tune prompt construction so retrieved guideline chunks + calculated metrics + looked-up nutrient facts merge into one coherent, cited response
- In parallel: start drafting a ~15–20 profile evaluation test set (with expected guideline citations) so it's ready for Week 3

**Known risk:** this week tends to run long because getting the agent to blend calculated metrics with retrieved text coherently takes more prompt iteration than expected. Budget slack here if possible.

---

## Week 3 — Retrieval Quality + Evaluation

Run together since eval work naturally surfaces what needs improving.

- Add hybrid search (BM25 + vector) — highest-impact upgrade for guideline-style text (exact terms like "vitamin D" combined with semantic search)
- Run the eval set through RAGAS (or manual scoring): retrieval precision/recall, faithfulness, answer relevance
- Fix the 2–3 biggest issues found (commonly: chunk size, missing metadata filters, poor handling of edge-case profiles like very low sleep or high BMI)
- Optional if time allows: reranking (Cohere rerank or cross-encoder) — nice-to-have, not essential to the portfolio story

---

## Week 4 — Interface, Memory, Deploy, Document

- **Days 1–2:** Streamlit frontend (fastest path to a demo-able UI) + conversation memory (`ConversationBufferMemory` or summary memory) for follow-up questions
- **Day 3:** Deploy (Render, Railway, or Hugging Face Spaces)
- **Days 4–5:** README — architecture diagram, design decisions and tradeoffs, actual eval numbers, screenshots/short demo video
- **Days 6–7 (buffer):** Absorb whatever slipped from Week 2/3, or write an optional short blog post walking through the design decisions

---

## What Was Cut to Fit 4 Weeks

- Reranking (optional add-back if ahead of schedule)
- Extensive query rewriting / multi-hop query decomposition
- Custom frontend beyond Streamlit

None of these are what makes the project stand out for a portfolio — the personalization logic and the evaluation numbers are. Add them back only if time allows, in that order.

---

## Design Notes Worth Highlighting in the README

1. **Why a tool for metrics instead of LLM math** — determinism and correctness for calculations like BMI and calorie needs.
2. **Why hybrid search** — nutrition guidelines mix exact terminology with conceptual questions; vector search alone misses exact-term queries.
3. **Why an explicit evaluation step** — most portfolio RAG projects skip this; showing retrieval and generation metrics (not just "it works") signals real understanding.
4. **Scope disclaimer** — the system should present itself as educational/informational, not a replacement for a doctor or registered dietitian. Worth describing how it handles out-of-scope or concerning profile inputs (e.g., extreme values) as a genuine RAG design consideration, not just a legal note.
