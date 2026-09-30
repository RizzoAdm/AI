#!/usr/bin/env python3
"""Relatorio semanal da memoria compartilhada (Mem0 server). So biblioteca padrao.
Uso:
  python3 mem0_report.py                # gera reports/AAAA-MM-DD.md (ultimos 7 dias)
  python3 mem0_report.py --days 14      # outra janela
  python3 mem0_report.py --delete <id>  # apaga uma memoria pelo id
"""
import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "http://127.0.0.1:8888"
QDRANT = "http://127.0.0.1:6333"
COLLECTION = "mem0_memories"
USER_ID = "rizzo"
DUP_THRESHOLD = 0.80


def load_key():
    for line in (HERE / ".env").read_text().splitlines():
        if line.startswith("ADMIN_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("ADMIN_API_KEY nao encontrado no .env")


def http(method, url, key=None, body=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def parse_dt(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def get_memories(key):
    d = http("GET", f"{API}/memories?user_id={USER_ID}&top_k=1000", key)
    return d.get("results", []) if isinstance(d, dict) else d


def neighbors(point_id):
    body = {
        "query": point_id,
        "limit": 4,
        "with_payload": True,
        "filter": {"must": [{"key": "user_id", "match": {"value": USER_ID}}]},
    }
    d = http("POST", f"{QDRANT}/collections/{COLLECTION}/points/query", body=body)
    return [p for p in d.get("result", {}).get("points", []) if p["id"] != point_id]


def skip_count(days):
    try:
        out = subprocess.run(
            ["docker", "logs", "--since", f"{days * 24}h", "mem0-server"],
            capture_output=True, text=True, timeout=60,
        )
        return (out.stdout + out.stderr).count("skip_infer")
    except Exception as e:
        return f"erro ao ler logs: {e}"


def report(days):
    key = load_key()
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    mems = get_memories(key)
    agent = lambda m: m.get("agent_id") or "(sem agent_id)"
    recent = [m for m in mems if (parse_dt(m.get("created_at")) or now) >= since]
    recent.sort(key=lambda m: m.get("created_at") or "")
    total = Counter(agent(m) for m in mems)

    L = [f"# Relatorio Mem0 - {now.astimezone():%Y-%m-%d %H:%M}", "",
         f"Janela: ultimos {days} dias | user_id: {USER_ID}", "",
         "## 1. Totais por agente", ""]
    for a, n in total.most_common():
        L.append(f"- {a}: {n} no total, {sum(1 for m in recent if agent(m) == a)} novas na janela")
    L += ["", f"Total geral: {len(mems)}", "",
          "## 2. Memorias novas na janela (e verdade? e duravel?)", ""]
    for m in recent:
        L.append(f"- [{(m.get('created_at') or '')[:10]}] {agent(m)} | {m.get('memory')} | id: {m.get('id')}")
    if not recent:
        L.append("- (nenhuma)")

    L += ["", f"## 3. Candidatas a duplicata, voce decide (similaridade >= {DUP_THRESHOLD})", ""]
    seen = set()
    try:
        for m in recent:
            for p in neighbors(m["id"]):
                pair = tuple(sorted([str(m["id"]), str(p["id"])]))
                if p.get("score", 0) >= DUP_THRESHOLD and pair not in seen:
                    seen.add(pair)
                    L.append(f"- {p['score']:.3f} | {m.get('memory')}  <->  "
                             f"{p.get('payload', {}).get('data')} | ids: {m['id']} / {p['id']}")
        if not seen:
            L.append("- (nenhuma)")
    except (urllib.error.URLError, KeyError) as e:
        L.append(f"- nao foi possivel verificar: {e}")

    L += ["", "## 4. Auto-sync do Hermes bloqueado (skip_infer) na janela", "",
          f"- {skip_count(days)} descartes (conta so desde a ultima recriacao do container)", "",
          "## 5. Rotina (5 min)", "",
          "- [ ] Secao 2: cada item e verdadeiro e duravel? Lixo -> python3 mem0_report.py --delete <id>",
          "- [ ] Secao 3: apague a versao mais velha ou menos completa de cada par",
          "- [ ] O canario da semana aparece na secao 2? (decisao real dita ao Hermes sem pedir pra anotar)",
          "- [ ] Algo importante que voce disse e NAO aparece? Anote aqui (falso negativo): ___",
          "- [ ] Secao 4 alta e secao 2 quase vazia = Hermes nao esta gravando por conta propria; revisar SOUL.md",
          ""]

    out_dir = HERE / "reports"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{now.astimezone():%Y-%m-%d}.md"
    path.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nRelatorio salvo em: {path}")


def main():
    ap = argparse.ArgumentParser(description="Relatorio e manutencao da memoria Mem0")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--delete", metavar="ID")
    args = ap.parse_args()
    if args.delete:
        print(http("DELETE", f"{API}/memories/{args.delete}", load_key()))
    else:
        report(args.days)


if __name__ == "__main__":
    main()
