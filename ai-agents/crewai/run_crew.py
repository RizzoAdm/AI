#!/usr/bin/env python3
"""
run_crew.py - executa uma Crew definida em crews/<nome>/agents.yaml + tasks.yaml.

Uso (venv do CrewAI ativo, ou chamando venv/bin/python pelo caminho absoluto):
  python run_crew.py pessoal "seu pedido aqui"
  python run_crew.py pessoal "seu pedido aqui" --verbose

Saidas: outputs/<crew>/<AAAA-MM-DD_HHMM>_<tema>/  (pedido.md + um .md por task)
A ultima linha impressa e OUTPUT_DIR=<pasta>.
"""
import os

# Telemetria do CrewAI desligada (antes de importar o crewai)
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import argparse
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import yaml
from crewai import LLM, Agent, Crew, Process, Task

from mem0_tools import BuscarMemoriaTool, SalvarMemoriaTool
from web_tools import BuscarWebTool

BASE = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434"


def slug(text, n=40):
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-zA-Z0-9]+", "-", t).strip("-").lower()
    return t[:n].strip("-") or "pedido"


def build_tools(names, crew_name, agent_name):
    factory = {
        "buscar_memoria": lambda: BuscarMemoriaTool(),
        "salvar_memoria": lambda: SalvarMemoriaTool(agent_id=f"crew-{crew_name}", crew_role=agent_name),
        "buscar_web": lambda: BuscarWebTool(),
    }
    unknown = [n for n in names if n not in factory]
    if unknown:
        sys.exit(f"Tool desconhecida em '{agent_name}': {unknown}. Disponiveis: {list(factory)}")
    return [factory[n]() for n in names]


def main():
    ap = argparse.ArgumentParser(description="Executa uma Crew.")
    ap.add_argument("crew", help="nome da pasta em crews/ (ex.: pessoal)")
    ap.add_argument("pedido", help="o pedido, entre aspas")
    ap.add_argument("--verbose", action="store_true", help="mostra o raciocinio dos agentes")
    a = ap.parse_args()

    cdir = BASE / "crews" / a.crew
    if not cdir.is_dir():
        sys.exit(f"Crew '{a.crew}' nao encontrada em {cdir}")
    agents_cfg = yaml.safe_load((cdir / "agents.yaml").read_text())
    tasks_cfg = yaml.safe_load((cdir / "tasks.yaml").read_text())

    out = BASE / "outputs" / a.crew / f"{datetime.now():%Y-%m-%d_%H%M}_{slug(a.pedido)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pedido.md").write_text(a.pedido + "\n")

    agents = {}
    for name, c in agents_cfg.items():
        agents[name] = Agent(
            role=c["role"],
            goal=c["goal"],
            backstory=c["backstory"],
            llm=LLM(model=f"ollama/{c['model']}", base_url=OLLAMA_URL, temperature=c.get("temperature", 0.3)),
            tools=build_tools(c.get("tools", []), a.crew, name),
            max_iter=c.get("max_iter", 5),
            allow_delegation=False,
            verbose=a.verbose,
        )

    tasks = {}
    for i, (name, c) in enumerate(tasks_cfg.items(), 1):
        if c["agent"] not in agents:
            sys.exit(f"Task '{name}': agente '{c['agent']}' nao existe no agents.yaml")
        missing = [x for x in c.get("context", []) if x not in tasks]
        if missing:
            sys.exit(f"Task '{name}': context {missing} precisa vir ANTES no tasks.yaml")
        tasks[name] = Task(
            description=f"Today's date: {datetime.now():%Y-%m-%d}.\n\n" + c["description"],
            expected_output=c["expected_output"],
            agent=agents[c["agent"]],
            context=[tasks[x] for x in c.get("context", [])],
        )

    crew = Crew(
        agents=list(agents.values()),
        tasks=list(tasks.values()),
        process=Process.sequential,
        memory=False,
        verbose=a.verbose,
    )
    result = crew.kickoff(inputs={"pedido": a.pedido})
    # Grava cada task aqui (o output_file do CrewAI tira a "/" inicial de caminhos absolutos)
    for i, (name, t_out) in enumerate(zip(tasks, result.tasks_output), 1):
        (out / f"{i}_{name}.md").write_text((t_out.raw or "") + "\n")
    print(result.raw)
    print(f"OUTPUT_DIR={out}")


if __name__ == "__main__":
    main()
