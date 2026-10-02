"""
mem0_tools.py - tools de memoria das Crews via API REST do Mem0 server.

Le do .env desta pasta: MEM0_URL, MEM0_API_KEY, MEM0_USER_ID, MEM0_INFER.
Uso numa Crew:
    from mem0_tools import memory_tools
    Agent(..., tools=memory_tools(agent_id="crew-gcn", crew_role="pesquisador"))
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).with_name(".env"))

MEM0_URL = os.environ.get("MEM0_URL", "http://127.0.0.1:8888")
MEM0_USER_ID = os.environ.get("MEM0_USER_ID", "rizzo")
MEM0_INFER = os.environ.get("MEM0_INFER", "false").strip().lower() in {"1", "true", "yes", "on"}


def _api(method: str, path: str, body: dict):
    key = os.environ.get("MEM0_API_KEY")
    if not key:
        raise RuntimeError("MEM0_API_KEY ausente no .env da Crew")
    req = urllib.request.Request(
        MEM0_URL + path,
        data=json.dumps(body).encode(),
        method=method,
        headers={"Content-Type": "application/json", "X-API-Key": key},
    )
    # timeout alto: com infer=true o servidor chama o LLM de extracao
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read() or b"null")


def _results(resp):
    if isinstance(resp, dict):
        return resp.get("results", [])
    return resp or []


class BuscarInput(BaseModel):
    query: str = Field(..., description="O que procurar, em poucas palavras.")


class SalvarInput(BaseModel):
    fato: str = Field(..., description="Uma frase curta e autocontida, ex.: 'GCN: nicho inicial e roofing em Calgary.'")


class BuscarMemoriaTool(BaseTool):
    name: str = "buscar_memoria"
    description: str = (
        "Busca na memoria de longo prazo compartilhada (Hermes e Crews) fatos, "
        "decisoes e preferencias relevantes para a tarefa. Use no inicio da tarefa."
    )
    args_schema: Type[BaseModel] = BuscarInput

    def _run(self, query: str) -> str:
        try:
            resp = _api("POST", "/search", {"query": query, "filters": {"user_id": MEM0_USER_ID}, "top_k": 5})
        except (urllib.error.URLError, RuntimeError) as e:
            return f"Erro ao buscar memoria: {e}"
        items = _results(resp)
        if not items:
            return "Nenhuma memoria relevante encontrada."
        return "\n".join(f"- [{m.get('agent_id') or '?'}] {m.get('memory')}" for m in items)


class SalvarMemoriaTool(BaseTool):
    name: str = "salvar_memoria"
    description: str = (
        "Grava UM fato duravel na memoria de longo prazo (decisao, descoberta, preferencia, "
        "fato do projeto). Escreva uma frase curta e autocontida comecando pelo projeto, "
        "ex.: 'GCN: ...'. NAO grave rascunhos, textos longos nem o resultado inteiro da "
        "tarefa - isso vai para os arquivos de saida."
    )
    args_schema: Type[BaseModel] = SalvarInput
    agent_id: str
    crew_role: str = ""

    def _run(self, fato: str) -> str:
        body = {
            "messages": [{"role": "user", "content": fato}],
            "user_id": MEM0_USER_ID,
            "agent_id": self.agent_id,
            "metadata": {"source": "crewai", "crew_role": self.crew_role},
            "infer": MEM0_INFER,
        }
        try:
            resp = _api("POST", "/memories", body)
        except (urllib.error.URLError, RuntimeError) as e:
            return f"Erro ao salvar memoria: {e}"
        saved = [m.get("memory") for m in _results(resp) if m.get("memory")]
        if not saved:
            return "Nada novo foi gravado."
        return "Saved to memory: " + " | ".join(saved)


def memory_tools(agent_id: str, crew_role: str = ""):
    return [BuscarMemoriaTool(), SalvarMemoriaTool(agent_id=agent_id, crew_role=crew_role)]
