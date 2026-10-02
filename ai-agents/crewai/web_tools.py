"""
web_tools.py - busca na web via DuckDuckGo (biblioteca ddgs), sem chave de API.
Uso numa Crew:
    from web_tools import BuscarWebTool
    Agent(..., tools=[BuscarWebTool()])
"""
from typing import Type

from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import BaseModel, Field


class BuscarWebInput(BaseModel):
    query: str = Field(..., description="Termos de busca, em poucas palavras.")


class BuscarWebTool(BaseTool):
    name: str = "buscar_web"
    description: str = (
        "Busca na web (DuckDuckGo) e devolve ate 5 resultados com titulo, link e trecho. "
        "Use termos curtos; para assuntos locais ou recentes, inclua cidade e ano."
    )
    args_schema: Type[BaseModel] = BuscarWebInput

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(query, max_results=5)
        except Exception as e:
            return f"Erro na busca web: {e}"
        if not results:
            return "Nenhum resultado encontrado."
        return "\n\n".join(f"{r.get('title')}\n{r.get('href')}\n{r.get('body')}" for r in results)
