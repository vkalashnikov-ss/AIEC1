# Session 10: LLM Servers - Lessons Learned

## 3 Lessons Learned

### 1. Serverless and Dedicated Endpoints Represent Different Design Philosophies

**Full Explanation:**
Serverless endpoints prioritize developer velocity and cost transparency through per-token billing on shared infrastructure, while dedicated endpoints prioritize performance consistency and capacity guarantees through fixed hourly costs on isolated infrastructure. This isn't just a pricing difference—it's a fundamental architecture choice that determines latency profiles, throughput guarantees, and operational complexity. Choosing the wrong model for your use case can lead to poor user experience (serverless during peak hours) or unnecessary cost (dedicated for variable workloads). Production applications should profile their actual traffic patterns before committing to either model.

**One-Sentence Version:**
Serverless maximizes flexibility and cost transparency; dedicated maximizes performance consistency and capacity guarantees.

---

### 2. Token Throughput Directly Determines Concurrent User Capacity and Cost Efficiency

**Full Explanation:**
Token throughput (tokens per second) is the true bottleneck for scaling LLM applications. An endpoint with 50 tokens/second can handle roughly 50 concurrent users assuming 1-second average token generation time. But throughput also affects cost-per-user: a high-throughput endpoint with higher hourly cost might be cheaper per token when serving many users. Understanding this relationship prevents costly mistakes like over-provisioning serverless for predictable high traffic (use dedicated instead) or under-provisioning dedicated and hitting queuing bottlenecks. Tools like LangSmith can visualize this relationship empirically.

**One-Sentence Version:**
Throughput limits concurrent users; optimizing for it requires matching endpoint choice to expected traffic patterns.

---

### 3. RAG Constrains LLM Outputs to Factual, Source-Based Information

**Full Explanation:**
RAG is a pragmatic solution to hallucination in LLMs: by retrieving relevant context before generation, we ground the model's response to actual sources. This doesn't require model retraining or fine-tuning—just retrieval + generation. However, RAG quality depends critically on embedding model quality, chunk size, retrieval strategy, and prompt design. Embedding model choice is especially important: Qwen3 embeddings (4B or 8B) provide excellent quality for open-source RAG, while larger models like OpenAI's offer even better semantic understanding at higher cost. The architecture forces a tradeoff: larger context windows increase accuracy but also latency and cost.

**One-Sentence Version:**
RAG converts hallucination from a fundamental problem into a retrieval problem, making LLM outputs verifiable and sourceable.

---

## 3 Lessons Yet to Learn

### 1. How to Optimize Chunk Size and Overlap for Different Document Types and Query Patterns

**Full Explanation:**
This session used fixed chunk sizes (750 tokens) and no overlap, but optimal chunking depends on document structure (PDFs vs. long-form text vs. structured data), query specificity (keyword-based vs. semantic), and downstream task (QA vs. summarization vs. synthesis). Fine-grained chunks enable precise retrieval but create noise; coarse chunks provide context but may miss detailed answers. Overlap helps continuity but increases embedding cost. Learning when to tune these requires empirical evaluation across real workloads.

**One-Sentence Version:**
Optimal chunking is task-specific and requires empirical tuning against your actual queries and documents.

---

### 2. Advanced Retrieval Strategies Beyond Vector Similarity (Hybrid Search, Semantic Routing, Re-ranking)

**Full Explanation:**
This session used basic vector similarity search. Production RAG systems often employ:
- **Hybrid search**: Combining vector similarity with BM25 keyword search to capture both semantic and lexical relevance
- **Semantic routing**: Using an LLM to classify queries and route them to specialized retrievers
- **Re-ranking**: Using a cross-encoder to score and reorder retrieval results by relevance
- **Multi-hop retrieval**: Iteratively refining queries based on partial answers
Each strategy has different latency/accuracy/cost tradeoffs that require empirical comparison.

**One-Sentence Version:**
Vector similarity is a starting point; production systems require hybrid search, routing, and re-ranking for quality and efficiency.

---

### 3. Cost-Quality Tradeoffs Across the Full Stack (Embedding Models, LLMs, Inference Endpoints, Re-ranking)

**Full Explanation:**
Every component in a RAG pipeline has cost and quality dimensions:
- Embedding models: Qwen3 4B (cheaper, faster) vs. Qwen3 8B vs. OpenAI (expensive, highest quality)
- LLMs: Open-source models (cheap) vs. proprietary (expensive, higher quality)
- Inference: Serverless (per-token) vs. dedicated (fixed cost)
- Re-ranking: Lightweight models vs. expensive cross-encoders
Optimizing requires understanding how each component affects end-to-end accuracy and cost, then profiling against real workloads. A cheaper embedding model with better re-ranking might beat an expensive embedding model with no re-ranking.

**One-Sentence Version:**
Cost-quality optimization requires empirical profiling across all components, not just choosing the "best" model for each.

---

## Summary

**3 Lessons Learned:**
1. Serverless vs. dedicated is a latency-for-flexibility tradeoff
2. Throughput determines concurrent capacity; cost optimization requires matching to traffic patterns
3. RAG makes LLMs factual; quality depends on embedding, retrieval, and prompt design

**3 Lessons Yet to Learn:**
1. Task-specific chunk size and overlap optimization
2. Advanced retrieval strategies (hybrid, routing, re-ranking)
3. Full-stack cost-quality optimization across embedding, LLM, and inference choices
