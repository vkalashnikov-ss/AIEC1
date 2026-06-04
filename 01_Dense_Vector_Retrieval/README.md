<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

<h1 align="center" id="heading">Session 1: Dense Vector Retrieval</h1>

### [Quicklinks]()

| 📰 Module Sheet                                                                 | ⏺️ Recording | 🖼️ Slides | 👨‍💻 Repo       | 📝 Homework | 📁 Feedback |
| :------------------------------------------------------------------------------- | :----------- | :-------- | :------------ | :---------- | :---------- |
| [Dense Vector Retrieval](../00_Docs/Modules/01_Dense_Vector_Retrieval/README.md) |[Recording!](https://us02web.zoom.us/rec/share/sHWvo0Nd1aI0SEhKecOLEX9kFGVJJAdYfsKiuTmm8t85W48Z2lnjpnzTy8jAd8R5.PwuqibGwAZhvDd8c) <br> passcode: `C62n^@Q!`| [Session 1 Slides](https://canva.link/htfqf8i39yejyhn) | You are here! | [Session 1 Assignment](https://forms.gle/Z9qskfVaAvPjn6gz8) | [Feedback 6/2](https://forms.gle/21a2uoL9DVZPwgJP6) |


## 🏗️ How AIM Does Assignments

> 📅 **Assignments will always be released to students as live class begins.** We will never release assignments early.

Each assignment will have a few of the following categories of exercises:

- ❓ **Questions** - these will be questions that you will be expected to gather the answer to. These can appear as general questions, or questions meant to spark a discussion in your breakout rooms.

- 🏗️ **Activities** - these will be work or coding activities meant to reinforce specific concepts or theory components.

- 🚧 **Advanced Builds (optional)** - Take on a challenge. These builds require you to create something with minimal guidance outside of the documentation.

## Main Assignment

In this assignment, you will build a vector RAG application using LangChain v1, OpenAI embeddings, and Qdrant.

The main notebook is:

```text
01_Cat_Health_Vector_RAG_LangChain_Qdrant.ipynb
```

The notebook uses the bundled cat health guideline PDF in `data/cat_health_guidelines.pdf`.

### Setup

From this folder, install the environment with uv:

```bash
uv sync
```

Then open the notebook in Cursor or VS Code and select the Python/Jupyter environment created by uv.

You will also need an OpenAI API key available when running the notebook.

---

## 🏗️ Activity #1: Embedding Similarity

Run the embedding similarity primer in the notebook.

You will compare embeddings for terms like:

- `king`
- `queen`
- `banana`
- `cat`
- `veterinarian`
- `cat health guidelines`

#### ❓Question #1

Why is cosine similarity useful for dense vector retrieval?

##### ✅ Answer:

Cosine similarity measures the **angle** between two embedding vectors rather than the distance between them, which makes it a natural fit for the geometry that modern embedding models produce. A few concrete reasons it is useful for dense retrieval:

- **It captures semantic direction, not magnitude.** Embedding models encode meaning into the *direction* a vector points in high-dimensional space. Two texts that are about the same topic point in similar directions even if one is much longer than the other. Cosine ignores vector length, so "cat" and "feline health and wellness guidelines for adult cats" can still score as highly related despite very different magnitudes.
- **It produces a bounded, comparable score in `[-1, 1]`.** That gives us a clean way to rank candidates (1.0 = same direction, 0 = orthogonal/unrelated, -1 = opposite). Ranking is exactly what a retriever needs.
- **It is cheap and parallelizable.** It is just a dot product plus two norms, so it scales to large indexes. For normalized embeddings (which most providers return or can be normalized to), cosine reduces to a pure dot product — the operation that vector databases like Qdrant are optimized for via ANN indexes.
- **It matches how the embedding model was trained.** OpenAI embedding models (and most contrastive embedding models) are trained with objectives that pull related texts to point in similar directions, so cosine is the metric the embedding space was actually shaped for.

In the primer in the notebook we saw this directly: `king` ↔ `queen` scored ~0.59, `king` ↔ `banana` scored ~0.31, and `cat` ↔ `cat health guidelines` scored ~0.50 — the rank order matches our intuition even though the absolute numbers are model-specific.

---

## 🏗️ Activity #2: Build the Vector RAG Pipeline

Run the notebook sections that:

1. Load the PDF into LangChain `Document` objects
2. Split the document into chunks
3. Embed the chunks
4. Store the chunk embeddings in in-memory Qdrant
5. Retrieve relevant chunks with similarity scores
6. Generate an answer grounded in retrieved context

#### ❓Question #2

Why is metadata important for a RAG application?

##### ✅ Answer:

Metadata is what turns a retrieved chunk from "a blob of text" into a **traceable, filterable, and trustworthy** piece of evidence:

- **Citations and trust.** Fields like `source`, `page`, and `start_index` let us tell the user *where* an answer came from (e.g., "page 8 of `cat_health_guidelines.pdf`"). Without this, users have no way to verify the LLM's output and we cannot detect hallucinations.
- **Filtering and routing.** Metadata such as `document_type: "cat_health_guideline"` lets the vector store pre-filter the search space (e.g., only feline documents, only recent versions, only a given customer's documents). This improves both relevance and security/multi-tenancy.
- **Debugging retrieval.** When an answer looks wrong, metadata tells us which page or section was retrieved so we can inspect chunking, overlap, or the source PDF itself.
- **Re-ranking and weighting.** Metadata like recency, author, or document quality can be used as features in a re-ranker on top of vector similarity.
- **Lifecycle management.** We need metadata to update or delete documents in the index when the source PDF changes — otherwise the store silently drifts out of date.
- **Domain safety.** For a health assistant, knowing the source document and page is essential so the assistant (and the user) can defer to the authoritative guideline and to a veterinarian.

#### ❓Question #3

What tradeoff do we make when choosing chunk size and chunk overlap?

##### ✅ Answer:

Chunking is a **context vs. precision** tradeoff, and overlap is a **continuity vs. cost** tradeoff.

**Chunk size:**

- **Larger chunks** preserve more surrounding context inside a single vector, so the LLM gets richer, self-contained passages and is less likely to be missing a definition or a qualifying sentence. The downside is that each vector represents a *mixture* of topics, which dilutes the embedding — the chunk can match many queries weakly instead of one query strongly, hurting ranking precision. Larger chunks also burn more prompt tokens per retrieved result.
- **Smaller chunks** produce sharper, more topically focused embeddings and better top-`k` precision, but each chunk may be too fragmentary to actually answer the question on its own (e.g., a sentence that says "this is contraindicated" without saying *what* "this" is). You then typically need a larger `k` to reassemble the context, which adds noise and cost.

**Chunk overlap:**

- **More overlap** reduces the risk that an important sentence gets split exactly at a chunk boundary and loses its surrounding context. It improves recall on boundary-spanning facts.
- **More overlap** also means more chunks → more embeddings to compute and store, more duplicated text in retrieved results, and a higher chance the top-`k` is filled with near-duplicates of the same passage.

**Rule of thumb for this notebook:** 1,000 chars with 200-char overlap is a reasonable starting point for prose PDFs. For dense reference material (tables, definitions, dosing) smaller chunks (~300–500 chars) with proportionally smaller overlap usually rank better; for narrative or explanatory text, larger chunks tend to give the LLM enough context to answer in one shot.

#### ❓Question #4

What does a similarity score help you understand, and what does it not prove by itself?

##### ✅ Answer:

**What it helps with:**

- **Relative ranking.** It tells us which chunk is *closer* to the query than another, in the geometry of this specific embedding model. That ordering is the whole basis of vector retrieval.
- **A sanity signal for tuning.** Watching scores rise or fall as we change the query wording, chunk size, or embedding model gives quick feedback on whether retrieval is getting better.
- **A rough confidence floor.** If the top score is very low (or the gap between top-1 and top-`k` is tiny), retrieval is probably weak and the answer is more likely to be ungrounded.

**What it does *not* prove:**

- **It is not absolute truth or factual correctness.** A high cosine score only means the two vectors point in similar directions in *this* embedding space — it does not mean the chunk actually answers the question, nor that the chunk's content is correct.
- **It is not comparable across models.** A 0.58 from `text-embedding-3-small` is not the same as a 0.58 from another model. Thresholds must be re-calibrated per model.
- **It is not calibrated probability.** Cosine similarity is not a probability of relevance. There is no universal "good > 0.8" rule, especially with OpenAI embeddings, which often produce scores in a narrower band.
- **It does not distinguish topical match from answer match.** A chunk can be highly similar because it shares vocabulary with the query (e.g., "cat" and "veterinarian") while not containing the actual answer. That is why we still inspect content, use re-ranking, or evaluate with task-level metrics.

---

## 🏗️ Activity #3: Vibe Check Retrieval Quality

Run the notebook's vibe check queries and inspect both:

- The retrieved context
- The generated answer

#### ❓Question #5

For the vibe check queries, did the retrieved context seem relevant before generation? Why or why not?

##### ✅ Answer:

Mostly yes, with one expected miss:

- **"What preventive care is recommended for cats?"** — Retrieved chunks covered annual/biannual exams, individualized risk assessment, and parasite prevention. Highly relevant; the question maps cleanly onto vocabulary that appears throughout the AAHA/AAFP guideline.
- **"What symptoms should make me call a veterinarian?"** — Retrieved chunks on vomiting, diarrhea, appetite changes, polyuria/polydipsia, grooming changes, and behavior changes were on-topic. Relevant. Some of the chunks were oriented toward *clinicians* asking the *owner* questions rather than owner-facing advice, but the underlying signs are the same.
- **"What should I know about feeding a healthy adult cat?"** — Retrieved chunks on body condition score, RER/DER calculations, AAFCO labeling, and gradual diet transitions are directly relevant nutrition content.
- **"Can my cat help me file my taxes?"** — Retrieval still returned the four nearest chunks (the vector store always returns `k` results), but their similarity scores were low and the content was unrelated. This is the expected behavior: **vector search is always best-effort and never returns "nothing"**, so it is the *prompt* and the LLM that correctly refused to answer. This is exactly why we keep the "I don't have enough information…" instruction in the system prompt and inspect scores rather than relying on retrieval alone to filter junk queries.

**Why relevance was generally good:** the queries use plain-English health/care vocabulary that overlaps strongly with the guideline's terminology, the chunks are large enough (1,000 chars) to carry self-contained context, and `k=4` is enough variety to cover multi-faceted questions like "what symptoms" without drowning the prompt in noise.

**Where it could fail:** queries that use synonyms the document does not use, queries that require combining facts across many pages, or queries that ask about something the PDF simply does not cover (like the taxes question). For those, we would need query rewriting, a larger `k` with a re-ranker, or a similarity-score threshold to abstain.

---

## 🏗️ Activity #4: Tune Retrieval

Improve retrieval quality by changing one or more of:

- Chunk size
- Chunk overlap
- Retrieval `k`
- Query wording

Document what changed and whether retrieval improved.

##### Settings Changed:

- **Test question:** "How often should senior cats be examined by a veterinarian?"
- **Chunk size:** `1000` → `500`
- **Chunk overlap:** `200` → `100`
- **Retrieval `k`:** `4` → `6`
- **Query wording:** unchanged (kept the same question so the chunking/`k` effect is isolated).

##### Results:

1. **Top-1 became sharper and more on-target.** With 1,000-char chunks the top hit was a broad "discussion items for all life stages" passage (score ≈ 0.71) that *mentioned* exam frequency only in passing. With 500-char chunks the top hit became the chunk that literally states `"Senior cats should be seen at least every 6 months and more frequently for those with chronic conditions"` (score ≈ 0.73). Smaller chunks → more topically pure embedding → better alignment with a specific, fact-shaped query.
2. **Scores compressed less and ranked better.** Because each chunk is now about one thing, the truly relevant chunk pulled away from the others instead of being averaged together with neighboring topics. Bumping `k` to 6 added supporting context (annual exam baseline, senior-care framing) without introducing noise.
3. **The final generated answer improved.** Baseline answered correctly but tersely (only the 6-month figure). The tuned answer added the supporting fact that annual exams are the minimum for all cats and that senior cats with chronic conditions need to be seen more often — i.e., a more complete, still-grounded answer drawn from multiple sources.

**Takeaway:** chunk size should be tuned **to the shape of the questions you expect**. Smaller chunks reward specific fact-shaped queries; larger chunks help broad explanatory queries. A production system would typically combine moderate-sized chunks with a re-ranker over a larger `k` to get the best of both worlds.

---

## Optional Deep Dive: RAG From Scratch

If you want to look underneath the library abstractions, run the optional reference notebook:

```text
02_Cat_Health_Vector_RAG_From_Scratch.ipynb
```

It builds the same retrieval pipeline again with only:

- `pypdf` for extracting text from the PDF
- Python standard-library HTTP requests for calling OpenAI
- Handcrafted document, chunking, embedding, similarity-search, vector-store, and generation primitives

This notebook is a reference walkthrough, not an additional assignment. Its purpose is to make the responsibilities hidden by LangChain, Qdrant, and provider SDKs visible.

---

## Submitting Your Homework

### Main Assignment

Follow these steps to prepare and submit your homework:

1. Pull the latest updates from upstream into the main branch of your AIE9 repo:

```bash
git checkout main
git pull upstream main
git push origin main
```

2. Start Cursor from the `01_Dense_Vector_Retrieval` folder.
3. Complete the notebook.
4. Answer the questions in this `README.md`.
5. Add, commit, and push your modified work to your origin repository.

When submitting your homework, provide the GitHub URL to your AIE9 repo.
