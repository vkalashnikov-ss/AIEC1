# Session 10 Optional Activities Guide

This guide covers two optional activities that extend the core Session 10 assignment with advanced evaluation and local model experimentation.

---

## Activity 1: RAGAS Evaluation with Cost Analysis

### Overview

Compare your Fireworks AI RAG application against OpenAI's gpt-4o-mini using:
- **RAGAS metrics**: Context precision, recall, faithfulness, answer relevancy
- **LangSmith tracing**: Token usage and cost analysis
- **Empirical testing**: Real queries from the cat health guide PDF

### Why This Matters

In production, you need data to support model choice decisions. This activity teaches you to:
1. Measure RAG quality programmatically (not just subjective testing)
2. Track costs with LangSmith
3. Make cost-quality tradeoffs with evidence

### Metrics Explained

| Metric | What It Measures | Why It Matters |
|--------|------------------|---|
| **Context Precision** | % of retrieved docs that are relevant | Measures if retrieval is noisy |
| **Context Recall** | % of all relevant docs that were retrieved | Measures if retrieval is complete |
| **Faithfulness** | % of answer claims grounded in context | Detects hallucinations |
| **Answer Relevancy** | % of answer that directly answers the question | Measures focus and completeness |

### Prerequisites

```bash
# API keys required
export FIREWORKS_API_KEY="your_fireworks_key"
export OPENAI_API_KEY="your_openai_key"
export LANGSMITH_API_KEY="your_langsmith_key"

# Dependencies already installed via pyproject.toml
# - ragas==0.4.3
# - langsmith>=0.9.0
# - datasets
```

### Running the Activity

```bash
# Open and run the notebook
jupyter notebook activity1_ragas_evaluation.ipynb
```

**Expected Output:**
- RAGAS evaluation scores for both providers
- Cost breakdown (input tokens, output tokens, total cost)
- Comparison table showing quality vs. cost tradeoffs
- Recommendations for your use case

### Cost Expectations

For 5 test queries:

**Fireworks (gpt-oss-20b):**
- Input tokens: ~3,000 (retrieval + generation)
- Output tokens: ~500 (avg 100 per response)
- Cost: ~$0.0005 total (~$0.00001 per query)

**OpenAI (gpt-4o-mini):**
- Input tokens: ~3,000
- Output tokens: ~500
- Cost: ~$0.0009 total (~$0.00018 per query)

**At Scale (1000 queries/day):**
- Fireworks: ~$0.15/day = ~$45/month
- OpenAI: ~$0.18/day = ~$54/month (but higher quality)

### Key Findings

Typically, you'll observe:

1. **Quality**: OpenAI slightly edges out Fireworks on complex questions, but both handle straightforward queries well
2. **Cost**: Fireworks is 3-5x cheaper, especially at scale
3. **Retrieval**: Better retrieval engineering (hybrid search, re-ranking) narrows the quality gap more than model choice
4. **Sweet Spot**: Fireworks + optimized retrieval delivers 90% of OpenAI quality at 20% of the cost

### Next Steps

After running this activity:
1. Add LangSmith integration to your production RAG pipeline
2. Monitor metrics weekly to catch quality regressions
3. Experiment with retrieval optimization to improve scores
4. Consider semantic routing: simple queries → Fireworks, complex queries → OpenAI

---

## Advanced Activity: Local Models with Ollama

### Overview

Build a RAG application using locally-running open-source models with Ollama. This demonstrates:
- **No external API dependency**: Run models on your hardware
- **Privacy**: Data never leaves your infrastructure
- **Cost**: One-time setup, ~$1-5/month electricity
- **Customization**: Fine-tune models on your domain

### Why This Matters

Local models are increasingly viable for production use cases. This activity teaches you to:
1. Set up and manage local LLM infrastructure
2. Compare local vs. cloud-hosted performance
3. Design hybrid architectures (local + cloud fallback)

### Model Selection

We use:
- **Chat**: Mistral 7B (4.1 GB, good speed/quality balance)
- **Embeddings**: Nomic Embed Text (274 MB, fast and effective)

**Why Mistral?**
- Open-source, commercially licensed
- Good quality for a 7B model
- Fast inference on CPU/GPU
- 8K context window (suitable for RAG)

**Alternatives:**
- `llama2`: Smaller, faster, lower quality
- `neural-chat`: Optimized for chat, smaller
- `dolphin-mixtral`: Larger, slower, higher quality (~30B)

### Prerequisites

#### Step 1: Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

#### Step 2: Pull Models

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

**Note**: First pull takes 5-15 minutes depending on internet speed.

#### Step 3: Start Ollama Server

```bash
ollama serve
```

This starts a local API server on `http://localhost:11434`.

#### Step 4: Verify Installation

```bash
# Check running models
curl http://localhost:11434/api/tags

# Test a simple query
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

### Running the Activity

```bash
# Ensure Ollama is running in another terminal
ollama serve

# In a new terminal, run the notebook
jupyter notebook advanced_activity_ollama.ipynb
```

**Expected Output:**
- Latency measurements for local RAG (retrieval + generation)
- Sample responses from Mistral
- Quality assessment and recommendations
- Resource usage analysis
- Trade-offs comparison table

### Performance Expectations

**Latency (on Apple Silicon M1/M2):**
- Retrieval: 0.2-0.5s
- Generation: 1-3s
- Total: 1-3.5s per query

**Latency (on CPU):**
- Retrieval: 0.2-0.5s
- Generation: 5-15s
- Total: 5-15s per query

**Quality:**
- Simple FAQs: 90% as good as Fireworks
- Complex reasoning: 70-80% as good as Fireworks
- Medical/legal accuracy: May need additional validation

### Trade-offs: Local vs. Cloud

| Factor | Local (Mistral) | Fireworks | OpenAI |
|--------|---|---|---|
| Initial setup | 30 min + 5GB disk | 5 min | 1 min |
| Monthly cost (1k queries) | $1-5 | $15 | $45 |
| Latency (per query) | 1-10s | 1-3s | 1-2s |
| Quality | Good | Good+ | Excellent |
| Scalability | Hard | Easy | Very easy |
| Privacy | Excellent | Good | Fair |
| Maintenance | High | Low | None |

### Recommendation Matrix

**Use Local Models When:**
- ✅ Privacy is critical (healthcare, legal, finance)
- ✅ Offline/air-gapped deployment required
- ✅ High query volume ($500+/month in API costs)
- ✅ Experimenting with prompts and fine-tuning
- ✅ Need to customize model behavior

**Don't Use Local Models When:**
- ❌ Need quick time-to-market
- ❌ Accuracy is critical and budget permits OpenAI
- ❌ Limited infrastructure/DevOps resources
- ❌ Unpredictable query volume
- ❌ Mobile or serverless deployment

### Advanced Optimizations

#### 1. Improve Quality

```bash
# Use a larger, better model
ollama pull neural-chat  # Better quality than mistral
ollama pull dolphin-mixtral  # Even better, but slower

# Fine-tune on your data (advanced)
ollama create my-custom-model --base mistral --from Modelfile
```

#### 2. Improve Latency

```bash
# Quantize to 4-bit for speed (slight quality loss)
ollama pull mistral:q4
ollama run mistral:q4

# Use GPU acceleration if available
# (Ollama auto-detects NVIDIA/AMD GPUs)
```

#### 3. Improve Throughput

```bash
# Run multiple Ollama instances
OLLAMA_NUM_PARALLEL=4 ollama serve

# Or use a production deployment framework
# - vLLM (https://github.com/vllm-project/vllm)
# - Text Generation WebUI
# - LocalAI
```

#### 4. Production Deployment

For production, consider:
- **vLLM**: Fast inference server with batching
- **LocalAI**: Drop-in OpenAI API replacement
- **Replicate**: Hosted open-source models
- **HuggingFace Inference**: Managed deployment

### Hybrid Architecture Example

```python
from app.models import get_chat_model
from langchain_ollama import ChatOllama

def intelligent_routing(question: str):
    """Route to local or cloud based on complexity."""
    
    # Quick heuristic: if question is short and matches FAQ patterns,
    # use local; otherwise use cloud
    if len(question) < 100 and any(
        keyword in question.lower() 
        for keyword in ["what is", "how do", "when"]
    ):
        return ChatOllama(model="mistral")
    else:
        return get_chat_model()  # Uses Fireworks/OpenAI

# Use in RAG pipeline
llm = intelligent_routing(user_question)
answer = rag_chain.invoke({"question": user_question})
```

---

## Completing Both Activities

### Suggested Schedule

1. **Week 1**: Complete Activity 1 (RAGAS evaluation)
   - Understand metrics and cost implications
   - Make informed decision about cloud model choice

2. **Week 2**: Complete Advanced Activity (Local models)
   - Set up and experiment with Ollama
   - Compare performance to cloud options
   - Design hybrid strategy if applicable

### Deliverables

For each activity, prepare a Loom video covering:

**Activity 1:**
- Key RAGAS metrics and what they mean
- Cost breakdown for both providers
- One specific insight you gained (e.g., "retrieval quality matters more than model choice")
- Your recommendation for this specific RAG application

**Advanced Activity:**
- Ollama setup process (can speed-record this)
- Sample queries and latency measurements
- Comparison table: local vs. cloud
- Your thoughts on when you'd use each approach

### Grading Rubric

| Category | Excellent | Good | Acceptable |
|----------|-----------|------|-----------|
| **Technical** | Both activities complete and functional | One activity complete, other attempted | Basic setup but incomplete runs |
| **Analysis** | Metrics explained well, insights drawn | Metrics shown, basic analysis | Metrics shown, no analysis |
| **Recommendation** | Clear use-case recommendation with evidence | Recommendation with partial evidence | Generic recommendation |
| **Presentation** | Clear Loom video, well-organized | Loom video present, some confusion | Loom exists but hard to follow |

---

## Troubleshooting

### Activity 1: RAGAS Issues

**Problem**: `ModuleNotFoundError: No module named 'ragas'`
```bash
uv sync  # Reinstall dependencies
```

**Problem**: LangSmith not tracing
```bash
# Verify env vars
echo $LANGSMITH_API_KEY
echo $LANGSMITH_PROJECT

# Check LangSmith UI for project
# https://smith.langchain.com
```

**Problem**: Low RAGAS scores
- Check if Fireworks API key is correct
- Ensure PDF is loaded (should show "Loaded 1 documents")
- Try with fewer test queries first

### Advanced Activity: Ollama Issues

**Problem**: `Connection refused: localhost:11434`
```bash
# Start Ollama in a new terminal
ollama serve

# Or check if it's already running
lsof -i :11434
```

**Problem**: Out of memory errors
```bash
# Check available memory
free -h  # Linux
vm_stat  # macOS

# Use smaller models
ollama pull mistral:q4  # Quantized version
```

**Problem**: Slow inference
- Model is running on CPU. Consider GPU options:
  ```bash
  # Check if GPU is used
  ollama list
  # NVIDIA: nvidia-smi to check GPU usage
  # Apple: Activity Monitor → GPU
  ```
- Try quantized version: `ollama pull mistral:q4`

---

## Resources

### Papers & Reading

- [RAGAS Paper](https://arxiv.org/abs/2309.15217) - Understanding evaluation metrics
- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288) - Behind open-source models
- [Mistral 7B Paper](https://arxiv.org/abs/2310.06825) - Model architecture

### Tools

- [LangSmith](https://smith.langchain.com) - Tracing and monitoring
- [Ollama](https://ollama.com) - Local model serving
- [RAGAS](https://ragas.io) - RAG evaluation framework
- [vLLM](https://github.com/vllm-project/vllm) - Production inference

### Community

- [LangChain Discord](https://discord.gg/6adMQxSpJS)
- [Ollama GitHub Discussions](https://github.com/ollama/ollama/discussions)
- [AI Maker Space Community](https://discord.gg/V7MHEG39EH)

---

## Next Steps After Activities

1. **Integrate LangSmith into production pipeline**
   - Track real user queries
   - Monitor metrics over time
   - Set up alerts for quality degradation

2. **Optimize retrieval**
   - Implement hybrid search
   - Try different embedding models
   - Experiment with chunk sizes

3. **Deploy with confidence**
   - Choose provider based on Activity 1 data
   - Set up monitoring and alerting
   - Plan for fallback/failover strategies

4. **Plan for scale**
   - Estimate daily query volume
   - Calculate monthly costs
   - Consider dedicated vs. serverless
   - Design auto-scaling strategy
