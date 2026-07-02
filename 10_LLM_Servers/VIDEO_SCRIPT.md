# Session 10: LLM Servers - Video Script

## Video Title
"Building RAG Applications with Fireworks AI: Serverless vs. Dedicated LLM Endpoints"

## Video Duration
~12-15 minutes

---

## Section 1: Introduction (0:00 - 1:00)

**Narrative:**
"In this session, we explore LLM serving—how to deploy large language models in production and choose between serverless and dedicated endpoints. We'll build a RAG application using Fireworks AI's open-source models and understand why token throughput and latency matter for real-world applications."

**Screen to Show:**
- Project directory structure (10_LLM_Servers/)
- Brief overview of Fireworks AI dashboard

**Key Points:**
- LLM serving is about infrastructure and deployment choices
- Two main models: serverless (flexible, per-token) vs. dedicated (consistent, fixed cost)
- RAG combines retrieval and generation for factual, sourceable responses

---

## Section 2: Fireworks AI Setup (1:00 - 3:00)

**Narrative:**
"First, let's set up our Fireworks AI environment. We'll show both options: using the serverless endpoint (instant, no setup) or creating a dedicated deployment (guaranteed capacity, 15-20 minutes)."

**Screen to Show:**
- `.env` file with `FIREWORKS_API_KEY`
- Fireworks web UI showing available models
- Model selection: `gpt-oss-20b` for chat, `qwen3-embedding-4b` for embeddings

**Key Points:**
- API key setup from fireworks.ai/api-keys
- Serverless endpoint: `accounts/fireworks/models/gpt-oss-20b` (instant)
- Dedicated deployment option: higher setup cost but consistent performance

---

## Section 3: Testing Endpoints with Notebook (3:00 - 6:00)

**Narrative:**
"Let's test our endpoint with the notebook. We'll send a single request to verify connectivity, then slam the endpoint with 24 concurrent requests to understand throughput and latency."

**Screen to Show:**
- Jupyter notebook: endpoint_slammer.ipynb
- Cell 1: API key loading
- Cell 2: Single request (show latency)
- Cell 3: 24 concurrent requests (show response times and any failures)
- Output showing response diversity and timing

**Key Points:**
- Single request verifies endpoint is alive
- Concurrent requests reveal throughput: 24 requests completing ~5-10 seconds = ~2-5 tokens/second per request
- Variable latency on serverless (shared infrastructure)
- Dedicated endpoints would show more consistent latency

---

## Section 4: Model Configuration and Tool Calling (6:00 - 8:30)

**Narrative:**
"Now let's look at our code architecture. We centralize model configuration so switching between serverless and dedicated is just an environment variable change. We also handle a quirk of open-source models: they sometimes append extra tokens like `<|call|>` to tool calls, which we fix before execution."

**Screen to Show:**
- `app/models.py`: `get_chat_model()` function
- Highlight FIREWORKS_BASE_URL and environment variable usage
- Show `fix_tool_calls()` function handling invalid JSON

**Key Points:**
- Environment variables control endpoint choice (no code changes needed)
- Open-source models need error handling (tool call cleanup)
- Centralized configuration enables easy model swaps

---

## Section 5: RAG Pipeline Architecture (8:30 - 11:00)

**Narrative:**
"The heart of this session is our RAG pipeline. It loads PDFs, chunks them intelligently, embeds them with Fireworks embeddings, stores vectors in Qdrant, and then retrieves relevant context for LLM generation."

**Screen to Show:**
- `app/rag.py` file structure
- Diagram or walkthrough of RAG flow:
  1. Load PDFs from `data/`
  2. Split into 750-token chunks
  3. Embed with Qwen3 embedding model
  4. Store in Qdrant (in-memory)
  5. Retrieve and generate
- Show the `retrieve_information` tool

**Key Points:**
- Token-aware chunking ensures consistent semantic boundaries
- Qdrant in-memory storage (great for demos, but persist to disk in production)
- Embedding quality directly affects retrieval quality
- Retrieved context constrains LLM responses to factual information

**Code to Highlight:**
```python
# Token-aware chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750, 
    chunk_overlap=0, 
    length_function=_tiktoken_len
)

# Embedding and vector store
embedding_model = OpenAIEmbeddings(
    model="accounts/fireworks/models/qwen3-embedding-4b",
    ...
)
```

---

## Section 6: Agent Graph and Tool Usage (11:00 - 13:00)

**Narrative:**
"The agent graph is a simple but powerful pattern. The model decides whether to call tools or end. If it calls tools, the tool node executes them and returns results. The model then makes a new decision. This loop continues until the model produces a final response without tool calls."

**Screen to Show:**
- `app/graphs/simple_agent.py`
- Graph visualization (StateGraph with agent → action → agent flow)
- Demo run: `python main.py`
- Show input query and agent's tool call, then final response

**Key Points:**
- Tool-using agents enable complex, multi-step reasoning
- LangGraph provides the orchestration layer
- Each iteration is stateless (messages accumulate, not state)
- Works with both proprietary and open-source models

---

## Section 7: Running the Full Application (13:00 - 14:30)

**Narrative:**
"Let's run the complete application. We'll see the agent take a question about kittens, call the retrieve_information tool to search our knowledge base, and then generate a grounded, factual response."

**Screen to Show:**
- Terminal running `python main.py`
- Output showing:
  - [human] query about kitten vaccinations
  - [ai] calling tools: retrieve_information
  - [tool] retrieve_information: retrieved context (truncated)
  - [ai] final response based on retrieved context

**Key Points:**
- Single command orchestrates embedding, retrieval, and generation
- Tool calling is transparent to the user
- Retrieved context appears in the response

---

## Section 8: Key Takeaways and Production Insights (14:30 - 15:00)

**Narrative:**
"In production, consider these factors when choosing and optimizing LLM endpoints:

1. **Serverless for unpredictable workloads**, dedicated for consistent traffic
2. **Monitor token throughput** to understand concurrent user capacity
3. **RAG quality depends on embedding model choice and retrieval strategy**
4. **Cost and quality are not opposites**—cheaper embeddings with better re-ranking can beat expensive embeddings"

**Screen to Show:**
- Quick summary graphic or bullet points
- Fireworks dashboard showing cost estimates

**Key Points:**
- Latency and throughput are your primary metrics
- Trade-offs are always present: cost, quality, latency
- Empirical evaluation (LangSmith, RAGAS) is essential in production

---

## Section 9: Optional Advanced Topics (Bonus - 15:00 - 16:30)

**Narrative (if time permits):**
"For advanced use cases, consider:
- **Hybrid search**: Combine vector + keyword search for better recall
- **Local models**: Replace Fireworks with Ollama for offline deployments
- **Cost analysis**: Track token usage with LangSmith to compare providers"

**Screen to Show:**
- Brief mention of Activity 1 (RAGAS evaluation) and Advanced Activity (Ollama)
- Links to next resources

---

## Production Checklist (End Card)

**What to Deploy:**
- ✅ Serverless or dedicated Fireworks endpoint
- ✅ RAG pipeline with persistent vector store (not in-memory)
- ✅ LangSmith tracing for cost and latency monitoring
- ✅ Graceful error handling for model quirks
- ✅ Rate limiting and queue management

**What to Monitor:**
- Time-to-first-token (latency)
- Tokens per second (throughput)
- Token cost per request
- Retrieval quality (RAGAS scores)
- User satisfaction metrics

---

## Script Notes

- **Tone**: Technical but accessible; explain tradeoffs, not just features
- **Pacing**: Move quickly through setup; spend time on architecture and tradeoffs
- **Live Demos**: Show the endpoint slamming notebook and agent execution
- **Visual Aids**: Diagram the RAG pipeline; show code with comments
- **Call to Action**: Encourage trying both serverless and dedicated to feel the difference
