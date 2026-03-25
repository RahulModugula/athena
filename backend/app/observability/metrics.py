from prometheus_client import Counter, Histogram

query_latency = Histogram("athena_query_latency_seconds", "Query latency", ["strategy"])
agent_step_duration = Histogram(
    "athena_agent_step_duration_seconds", "Agent step duration", ["agent"]
)
documents_ingested = Counter("athena_documents_ingested_total", "Documents ingested")
llm_requests = Counter("athena_llm_requests_total", "LLM API requests", ["provider"])
