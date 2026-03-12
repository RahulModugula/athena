# Athena Roadmap: RAG Pipeline → MCP-Native Multi-Agent Research Platform

## Vision

> "ATHENA: MCP-native research agent with LangGraph orchestration. A multi-agent research team (researcher, analyst, fact-checker, writer) that streams results through FastAPI + Server-Sent Events, builds a growing Graph RAG knowledge base across research sessions, and deploys in Docker on K8s."

---

## Current State (as of Phase 1)

A production-grade single-path RAG system + LangGraph agent pipeline:

- FastAPI backend with async PostgreSQL (pgvector)
- Hybrid search: dense (HNSW) + BM25 + RRF fusion + cross-encoder reranking
- LangGraph 4-agent pipeline: supervisor → researcher → analyst → fact-checker → writer
- Dual LLM provider support: Anthropic Claude + ZhipuAI GLM-4 (configurable)
- Streaming via SSE: `/api/query/stream` and `/api/research/stream`
- Document ingestion: PDF, TXT, Markdown, HTML, DOCX
- RAGAS evaluation framework with async API integration
- Streamlit UI (upload, search, evaluate pages)
- Docker Compose stack, GitHub Actions CI

---

## Phase 0 — Foundation (COMPLETE)

**Goal**: Migrate LLM providers, wire eval API, expand document support.

### 0A. Dual LLM Provider Support ✅
- `backend/app/config.py` — `llm_provider` field (`anthropic` | `zhipuai`)
- `backend/app/generation/chain.py` — Dynamic provider selection in `get_llm()`
- `backend/app/agents/llm.py` — Same pattern for agent LLM factory
- New dep: `langchain-anthropic>=0.3.0`
- Set `ATHENA_LLM_PROVIDER=anthropic` and `ATHENA_ANTHROPIC_API_KEY` in `.env`

### 0B. Eval API Wired ✅
- `backend/app/api/routes.py` — `POST /api/eval/run` launches `run_evaluation()` as `asyncio.create_task`
- `backend/app/models/orm.py` — `status` column on `EvalRun` (`pending` → `running` → `completed` | `failed`)
- `backend/alembic/versions/002_eval_status.py` — Migration

### 0C. Document Loaders Expanded ✅
- `backend/app/ingestion/loader.py` — `load_html()` (BeautifulSoup + lxml), `load_docx()` (python-docx)
- `backend/app/api/routes.py` — MIME type allowlist updated
- `backend/tests/test_loader.py` — 5 unit tests
- New deps: `beautifulsoup4`, `lxml`, `python-docx`

---

## Phase 1 — LangGraph Multi-Agent Orchestration (COMPLETE)

**Goal**: 4-agent research team with supervisor routing. New `/api/research` endpoints, `/api/query` untouched.

### Agents Package: `backend/app/agents/`

| File | Role |
|------|------|
| `state.py` | `ResearchState` TypedDict — shared state across all nodes |
| `llm.py` | `get_agent_llm()` — provider-agnostic LLM factory |
| `supervisor.py` | Plans research strategy, structured output via `ResearchPlan` |
| `researcher.py` | Decomposes question into 2-3 sub-queries, retrieves chunks for each |
| `analyst.py` | Identifies patterns/contradictions, produces cited analysis |
| `fact_checker.py` | Verifies claims against sources, outputs `FactCheckItem` with confidence |
| `writer.py` | Synthesizes final answer with `[Source N]` citations |
| `graph.py` | `StateGraph` wiring: START → supervisor → researcher → analyst → fact_checker → writer → END |

### API Endpoints (in `backend/app/api/routes.py`)
- `POST /api/research` — Synchronous, returns full `ResearchResponse` (answer, analysis, fact_check, sources, agent_trace, latency_ms)
- `POST /api/research/stream` — SSE stream of agent progress events: `agent_start`, `retrieval`, `analysis`, `fact_check`, `answer`, `done`

### RetrievalService (`backend/app/services/retrieval_service.py`)
- Extracted from `_retrieve_chunks` in routes.py
- `RetrievalService.retrieve(question, strategy, top_k, document_ids) → list[dict]`
- Injected into agents via `state["_retrieval_service"]`

### New Schemas (in `backend/app/models/schemas.py`)
- `ResearchRequest` — question, max_iterations, strategy
- `ResearchResponse` — answer, analysis, fact_check, sources, agent_trace, latency_ms
- `AgentStep` — agent, action, duration_ms
- `FactCheckResult` — claim, supported, confidence, evidence

### New Dep: `langgraph>=0.3.0`

---

## Phase 2 — Graph RAG Knowledge Base

**Goal**: Persistent Neo4j knowledge graph that grows across research sessions.

### Infrastructure
- `docker-compose.yml` — Add `neo4j:5-community` service (ports 7474:7474, 7687:7687, APOC plugin)
- `backend/app/config.py` — Add `neo4j_uri`, `neo4j_user`, `neo4j_password` (optional — falls back to vector-only if unset)
- New deps: `neo4j>=5.0.0`, `langchain-neo4j>=0.3.0`

### Graph Package: `backend/app/graph/`

| File | Purpose |
|------|---------|
| `extractor.py` | `LLMGraphTransformer` (Claude) extracts entities + relationships from chunks. Incremental only. |
| `store.py` | Neo4j wrapper: `add_entities_and_relationships()`, `query_subgraph()`, `search_entities()`, `get_entity_context()` |
| `models.py` | Pydantic: `Entity`, `Relationship`, `Subgraph` |

### Integration Points
- `backend/app/services/document_service.py` — After embedding, queue graph extraction as background task
- `backend/app/main.py` — Init Neo4j in lifespan → `app.state.graph_store`
- New: `backend/app/retrieval/graph_search.py` — Query Neo4j for entity context, return as virtual chunks
- `backend/app/retrieval/hybrid.py` — Add `include_graph: bool` param
- `backend/app/agents/researcher.py` — Second tool: `query_knowledge_graph`

### Cross-Session Memory
- `persist_to_graph` node after writer — writes session entities/relationships to Neo4j
- New migration `003_research_sessions.py` — `research_sessions` table
- New migration `004_graph_metadata.py` — `graph_extracted: bool` on `chunks`

### Testing
- `backend/tests/test_graph_extractor.py` — Mock LLM extraction
- `backend/tests/test_graph_store.py` — Neo4j operations

---

## Phase 3 — MCP Server Integration

**Goal**: Expose Athena as an MCP server + let agents call external MCP tools.

### Athena as MCP Server: `backend/app/mcp/`

| File | Purpose |
|------|---------|
| `server.py` | MCP server (using `mcp` SDK). Tools: `athena_search_documents`, `athena_research_question`, `athena_query_knowledge_graph`, `athena_ingest_document` |
| `tools.py` | JSON Schema tool definitions |
| `backend/mcp_server.py` | Standalone entry point (stdio + SSE transport) |

### External MCP Client
- `backend/app/mcp/client.py` — MCP client for connecting to external servers
- `backend/app/agents/tools.py` — LangChain wrappers: `web_search`, `verify_claim`, `fetch_url`
- `researcher.py` — Gets `web_search` + `fetch_url`
- `fact_checker.py` — Gets `verify_claim` + `web_search`
- `backend/app/config.py` — `mcp_servers: dict[str, str] = {}` mapping

### New Dep: `mcp>=1.0.0`

---

## Phase 4 — Observability, Caching & Hardening

**Goal**: Production readiness.

### OpenTelemetry Tracing
- New: `backend/app/observability/tracing.py` — OTLP exporter, FastAPI + SQLAlchemy + httpx instrumentation
- `backend/app/observability/logging.py` — Add trace IDs to structlog entries
- New deps: `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-instrumentation-sqlalchemy`, `opentelemetry-exporter-otlp`

### Prometheus Metrics
- New: `backend/app/observability/metrics.py` — `athena_query_latency_seconds`, `athena_agent_step_duration_seconds`, `athena_llm_tokens_total`, `athena_documents_ingested_total`
- `backend/app/main.py` — `/metrics` endpoint
- New dep: `prometheus-fastapi-instrumentator>=7.0.0`

### Redis Caching & Rate Limiting
- `docker-compose.yml` — Add Redis service
- New: `backend/app/services/cache.py` — Cache embeddings (content hash), retrieval results (query hash, 5min TTL)
- New: `backend/app/api/middleware.py` — Redis rate limiting
- New dep: `redis[hiredis]`

### Streamlit UI Updates
- `streamlit_app/pages/2_search.py` — "Deep Research" mode toggle → `/api/research/stream` with agent step display
- New: `streamlit_app/pages/4_knowledge_graph.py` — Neo4j graph visualization (pyvis)

---

## Phase 5 — Kubernetes Deployment

**Goal**: Full K8s deployment with monitoring.

### New Directory: `k8s/`

```
k8s/
├── namespace.yaml
├── configmap.yaml
├── secrets.yaml              # template, gitignored
├── ingress.yaml
├── kustomization.yaml
├── backend/
│   ├── deployment.yaml       # resource limits, HPA, health probes → /api/health
│   └── service.yaml
├── postgres/
│   ├── statefulset.yaml      # PVC-backed
│   └── service.yaml
├── neo4j/
│   ├── statefulset.yaml      # PVC-backed
│   └── service.yaml
├── redis/
│   ├── deployment.yaml
│   └── service.yaml
└── monitoring/
    ├── prometheus.yaml
    └── grafana.yaml          # pre-configured dashboards
```

### CI/CD Extension
- `.github/workflows/ci.yml` — Add Docker image build + push (GHCR), `kubectl --dry-run` manifest validation, optional staging deploy

---

## Dependency Additions by Phase

| Phase | New Dependencies |
|-------|-----------------|
| 0 | `langchain-anthropic`, `beautifulsoup4`, `lxml`, `python-docx` |
| 1 | `langgraph>=0.3.0` |
| 2 | `neo4j>=5.0.0`, `langchain-neo4j>=0.3.0` |
| 3 | `mcp>=1.0.0` |
| 4 | `opentelemetry-*`, `prometheus-fastapi-instrumentator`, `redis[hiredis]` |
| 5 | (K8s manifests only) |

## Alembic Migrations

| File | Phase | Change |
|------|-------|--------|
| `001_initial_schema.py` | 0 (existing) | Base schema |
| `002_eval_status.py` | 0 | `status` column on `eval_runs` |
| `003_research_sessions.py` | 2 | `research_sessions` table |
| `004_graph_metadata.py` | 2 | `graph_extracted` column on `chunks` |

## Key Design Decisions

1. **Claude Sonnet for agents** — cost/speed balance for multi-call orchestration; ZhipuAI kept as fallback
2. **Neo4j optional** — graceful fallback to vector-only if `neo4j_uri` unset
3. **Additive API** — `/api/research` is new; `/api/query` never breaks
4. **Async graph extraction** — background tasks prevent ingestion slowdown
5. **`max_iterations` guard** — prevents runaway agent loops (default 3, max 5)
6. **No breaking changes across phases** — each phase is independently deployable
