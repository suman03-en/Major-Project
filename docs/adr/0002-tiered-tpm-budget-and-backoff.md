# Tiered TPM Budget, Context Condensation, and Rate-Limit Backoff

Status: accepted

Free-tier Groq endpoints enforce an 8,000 tokens-per-minute (TPM) cap, while Devanagari legal **Provisions** tokenize densely and unbounded **Section** sibling expansion plus multi-loop helper calls can exhaust that budget within a single query. We chose a configurable script-aware context token budget (`MAX_CONTEXT_TOKENS = 3500`, `MAX_SIBLINGS_PER_SECTION = 2`) that packs primary **Provisions** and **Procedures** before optional siblings, header-aware jittered exponential backoff in `StandardLLMClient` (bypassed immediately when healthy `LLM_FALLBACKS` exist), and an optional `LLM_HELPER_MODEL` split for translation and completeness checks over raw character truncation or disabling iterative retrieval. All caps and retry thresholds live in `Settings` so switching to a paid tier only requires bumping configuration values.
