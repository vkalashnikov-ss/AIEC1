# Session 10: LLM Servers - Session Notes

## Overview

Session 10 focuses on understanding and deploying Language Model servers using Fireworks AI, a managed inference platform. We explore the differences between serverless and dedicated endpoints, implement endpoint testing, and build a production-ready RAG (Retrieval-Augmented Generation) application leveraging open-source models.

---

## Core Concepts

### 1. LLM Serving Infrastructure
LLM serving is the practice of running large language models on managed infrastructure with optimized hardware and networking. Different deployment models offer different tradeoffs:

- **Serverless**: On-demand, shared infrastructure; ideal for variable workloads
- **Dedicated**: Reserved infrastructure with guaranteed capacity; ideal for consistent production workloads

### 2. Fireworks AI Platform
Fireworks AI is an inference platform specializing in:
- Fast, cost-effective open-source model hosting
- Both serverless and dedicated deployment options
- Models like `gpt-oss-20b` (chat) and `qwen3-embedding-4b` (embeddings)
- Simple CLI (`firectl`) and web UI for deployment management

### 3. Performance Metrics
When evaluating LLM endpoints, consider:
- **Latency**: Time to first token (TTFT) and time per token (TPT)
- **Throughput**: Tokens processed per second (TPS)
- **Cost**: Per-token pricing vs. fixed hourly rates
- **Availability**: Guaranteed uptime and reliability

### 4. RAG Architecture
Retrieval-Augmented Generation (RAG) combines:
1. **Retrieval**: Using vector embeddings to fetch relevant documents from a knowledge base
2. **Generation**: Using the retrieved context to produce grounded, accurate responses

This addresses hallucination in LLMs by constraining outputs to factual, source-based information.

---

## Detailed Explanation

### Serverless vs. Dedicated Endpoints

**Serverless Endpoints:**
- Use Fireworks's shared infrastructure pool
- Automatically scaled based on demand
- No setup or configuration required
- Pay only for what you use (per-token pricing)
- Variable latency depending on current load
- Example: `accounts/fireworks/models/gpt-oss-20b`

**Dedicated Endpoints:**
- Custom deployment on isolated GPU/compute resources
- Minimum 15-20 minutes setup time via web UI or `firectl`
- Fixed hourly cost regardless of usage
- Consistent, predictable performance
- Auto-shutdown after 60 minutes of inactivity (configurable)
- Better for production workloads with stable traffic patterns

### Token Throughput and Latency

**Why Token Throughput Matters:**
- Throughput (tokens/second) determines how many concurrent requests an endpoint can handle
- Example: An endpoint with 50 TPS can theoretically serve 50 concurrent users with 1-second average token generation time
- Low throughput = longer wait times for users = poor UX and potential timeouts
- High throughput = more expensive per unit time, but handles more concurrent users efficiently

**Why Latency Matters:**
- Users expect responses within 1-3 seconds for interactive applications
- Latency includes: model inference time, network round-trips, and queuing delays
- High latency degrades user experience and can cause timeouts
- Serverless endpoints experience variable latency; dedicated endpoints provide consistent baseline

### RAG Pipeline

The implemented RAG pipeline (`app/rag.py`) follows this flow:

1. **Document Loading**: PDFs from `data/` directory are loaded using `PyMuPDFLoader`
2. **Chunking**: Documents are split into overlapping chunks using token-aware splitting (750 tokens per chunk)
3. **Embedding**: Chunks are embedded using Fireworks's Qwen3 embedding model
4. **Vector Store**: Embeddings are stored in an in-memory Qdrant vector database
5. **Retrieval**: User queries are embedded and matched against stored chunks
6. **Generation**: Retrieved context is passed to the LLM with the user query for grounded answer generation

---

## Implementation Walkthrough

### Project Structure

```
10_LLM_Servers/
├── app/
│   ├── models.py          # Chat model configuration (Fireworks ChatOpenAI)
│   ├── rag.py             # RAG pipeline and retrieve_information tool
│   ├── state.py           # LangGraph message state schema
│   ├── tools.py           # Tool definitions
│   ├── graphs/
│   │   ├── simple_agent.py      # Basic tool-using agent
│   │   └── agent_with_helpfulness.py  # Enhanced agent variant
├── data/                  # PDF documents for RAG
├── endpoint_slammer.ipynb # Endpoint testing notebook
├── main.py               # Agent entry point
├── pyproject.toml        # Project dependencies
└── ENDPOINT_SETUP.md     # Detailed endpoint setup guide
```

### Key Components

#### 1. Model Configuration (`app/models.py`)
```python
def get_chat_model(model_name: str | None = None, *, temperature: float = 0):
    """Return a configured LangChain ChatOpenAI client pointed at Fireworks."""
    name = model_name or os.environ.get(
        "FIREWORKS_CHAT_MODEL", "accounts/fireworks/models/gpt-oss-20b"
    )
    return ChatOpenAI(
        model=name,
        temperature=temperature,
        openai_api_key=os.environ["FIREWORKS_API_KEY"],
        openai_api_base=FIREWORKS_BASE_URL,
    )
```

This centralizes model configuration, allowing easy switching between serverless and dedicated endpoints.

#### 2. RAG Pipeline (`app/rag.py`)
The `retrieve_information` tool implements a two-step RAG graph:
- **Step 1 (retrieve)**: Vector similarity search to fetch relevant documents
- **Step 2 (generate)**: LLM generation constrained to retrieved context

The tool uses in-memory Qdrant to avoid external dependencies.

#### 3. Agent Graph (`app/graphs/simple_agent.py`)
A LangGraph-based agent that:
- Invokes the chat model with tool bindings
- Routes to `ToolNode` when tool calls are needed
- Iterates until no more tool calls are requested

---

## Run/Test Instructions

### 1. Environment Setup

```bash
# Copy example environment
cp .env.example .env

# Add your Fireworks API key
echo "FIREWORKS_API_KEY=your_api_key_here" >> .env
```

### 2. Option A: Use Serverless Endpoint (Recommended for Testing)

No setup required; use the default model identifiers:
- Chat: `accounts/fireworks/models/gpt-oss-20b`
- Embeddings: `accounts/fireworks/models/qwen3-embedding-4b`

### 3. Option B: Create a Dedicated Endpoint

```bash
# Using firectl CLI
firectl create deployment my-gpt-oss \
  --model accounts/fireworks/models/gpt-oss-20b \
  --min-replica 1 \
  --max-replica 1

# Then update environment
echo "FIREWORKS_CHAT_MODEL=accounts/fireworks/deployments/my-gpt-oss" >> .env
```

Or use the Fireworks web UI (see `ENDPOINT_SETUP.md`).

### 4. Test Endpoint with Notebook

```bash
jupyter notebook endpoint_slammer.ipynb
```

The notebook tests:
1. Single request verification
2. Concurrent request handling (24 simultaneous requests)

### 5. Run Agent

```bash
# Requires PDF documents in data/ directory
python main.py
```

This invokes the agent with a sample query about kitten vaccinations.

### 6. Test RAG Directly

```python
from app.rag import retrieve_information

result = retrieve_information("What are the recommended vaccinations for kittens?")
print(result)
```

---

## Troubleshooting

### Issue: "FIREWORKS_API_KEY not set"
**Solution**: Ensure your API key is in `.env` and `load_dotenv()` is called before imports.

### Issue: "No documents found" in RAG
**Solution**: Ensure PDF files are in the `data/` directory. The RAG pipeline silently continues with empty documents.

### Issue: Endpoint deployment hangs
**Solution**: Check deployment status in Fireworks UI. Deployments can take 15-20 minutes. Adjust resource allocation if needed.

### Issue: Tool calls include invalid JSON (e.g., `<|call|>` suffix)
**Solution**: The `fix_tool_calls()` function in `models.py` handles this automatically.

### Issue: High latency on serverless
**Solution**: This is expected during peak hours. Switch to a dedicated endpoint for consistent latency requirements.

---

## Completion Summary

✅ **Completed:**
- Environment setup for Fireworks AI integration
- Implemented RAG pipeline with document loading, chunking, embedding, and retrieval
- Created agent graph with tool-using capability
- Endpoint testing infrastructure (`endpoint_slammer.ipynb`)
- Model configuration abstraction supporting serverless and dedicated endpoints
- Tool call error handling for open-source models
- In-memory vector store (Qdrant) for retrieval

✅ **Key Learnings:**
1. Serverless vs. dedicated is a latency/throughput vs. cost tradeoff
2. Token throughput directly impacts concurrent user capacity and cost efficiency
3. RAG constrains LLM outputs to factual, source-based information
4. Vector embeddings enable semantic search beyond keyword matching

✅ **Production Considerations:**
- Use dedicated endpoints for customer-facing applications
- Monitor latency and throughput with LangSmith or equivalent
- Consider local models (Ollama) for offline/edge deployments
- Plan auto-shutdown for cost control on dedicated endpoints

---

## Next Steps

1. **Activity 1**: Implement RAGAS evaluation comparing Fireworks AI vs. OpenAI providers with LangSmith cost analysis
2. **Advanced Activity**: Swap Fireworks endpoints for locally-running Ollama models; compare quality and latency
3. **Optimization**: Implement semantic routing to minimize unnecessary LLM calls
4. **Monitoring**: Integrate LangSmith tracing to track token usage and costs at scale
