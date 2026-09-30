"""Aplica o patch local no main.py oficial do Mem0 server (roda no build).
Falha de proposito se o texto esperado nao for encontrado exatamente 1 vez."""
import sys
from pathlib import Path

MAIN = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/main.py")
src = MAIN.read_text()

PARAMS_LINE = '    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}\n'
SKIP_BLOCK = (
    '    if params.get("infer", True) and params.get("agent_id") in _SKIP_INFER_AGENTS:  # patch local\n'
    '        logging.info("skip_infer: agent_id=%s (auto-sync bloqueado pelo patch local)", params.get("agent_id"))\n'
    '        return JSONResponse(content={"results": [], "skipped": "infer_blocked_for_agent"})\n'
)

replacements = [
    (
        'BUNDLED_LLM_PROVIDERS = ("openai", "anthropic", "gemini")',
        'BUNDLED_LLM_PROVIDERS = ("openai", "anthropic", "gemini", "ollama")',
    ),
    (
        'BUNDLED_EMBEDDER_PROVIDERS = ("openai", "gemini")',
        'BUNDLED_EMBEDDER_PROVIDERS = ("openai", "gemini", "ollama")',
    ),
    (
        "set_session_factory(SessionLocal)\ninitialize_state(DEFAULT_CONFIG)",
        "from local_config import LOCAL_CONFIG as DEFAULT_CONFIG, SKIP_INFER_AGENTS as _SKIP_INFER_AGENTS  # patch local\n\n"
        "set_session_factory(SessionLocal)\ninitialize_state(DEFAULT_CONFIG)",
    ),
    (PARAMS_LINE, PARAMS_LINE + SKIP_BLOCK),
]

for old, new in replacements:
    count = src.count(old)
    if count != 1:
        raise SystemExit(f"patch_main: esperado 1 ocorrencia, achei {count}: {old[:60]!r}")
    src = src.replace(old, new)

MAIN.write_text(src)
print(f"patch_main: {MAIN} ajustado com sucesso")
