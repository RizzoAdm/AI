#!/usr/bin/env python3
"""
mem0_admin.py - administracao do Mem0 server (so stdlib, sem venv).

Uso (na pasta ~/Projects/AI/ai-agents/mem0-server):
  python3 mem0_admin.py register
  python3 mem0_admin.py create-key <label> <arquivo.env> [NOME_VAR]
  python3 mem0_admin.py list-keys
  python3 mem0_admin.py revoke <key_id>

Segredos nunca vao para a linha de comando nem para a tela:
a senha e pedida com getpass; a chave criada e gravada direto no .env.
"""
import getpass, json, os, sys, urllib.request, urllib.error
from pathlib import Path

BASE = os.environ.get("MEM0_URL", "http://127.0.0.1:8888")


def _safe_detail(detail):
    # Nunca ecoa o campo "input" dos erros de validacao do FastAPI
    if isinstance(detail, dict) and "detail" in detail:
        detail = detail["detail"]
    if isinstance(detail, list):
        return "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in detail)
    return str(detail)


def call(method, path, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read())
        except Exception:
            detail = None
        return e.code, detail


def login():
    email = input("E-mail do admin: ").strip()
    pw = getpass.getpass("Senha: ")
    st, resp = call("POST", "/auth/login", {"email": email, "password": pw})
    if st != 200:
        sys.exit(f"Login falhou (HTTP {st}): {_safe_detail(resp)}")
    return resp["access_token"]


def register():
    name = input("Nome: ").strip()
    email = input("E-mail: ").strip()
    pw = getpass.getpass("Senha (min. 8 caracteres): ")
    if pw != getpass.getpass("Repita a senha: "):
        sys.exit("Senhas diferentes. Nada foi feito.")
    st, resp = call("POST", "/auth/register", {"name": name, "email": email, "password": pw})
    if st != 200:
        sys.exit(f"Cadastro falhou (HTTP {st}): {_safe_detail(resp)}")
    print("Admin criado com sucesso.")


def upsert_env(path, var, value):
    p = Path(path).expanduser()
    lines = p.read_text().splitlines() if p.exists() else []
    lines = [l for l in lines if not l.startswith(var + "=")]
    lines.append(f"{var}={value}")
    p.write_text("\n".join(lines) + "\n")
    os.chmod(p, 0o600)


def create_key(label, env_path, var="MEM0_API_KEY"):
    token = login()
    st, resp = call("POST", "/api-keys", {"label": label}, token)
    if st != 201:
        sys.exit(f"Criacao falhou (HTTP {st}): {_safe_detail(resp)}")
    upsert_env(env_path, var, resp["key"])
    print(f"Chave '{label}' criada (prefixo {resp['key_prefix']}), gravada em {env_path} como {var}.")


def list_keys():
    token = login()
    st, resp = call("GET", "/api-keys", token=token)
    if st != 200:
        sys.exit(f"Listagem falhou (HTTP {st}): {_safe_detail(resp)}")
    for k in resp:
        print(f"{k['id']}  {k['label']:<12} prefixo={k['key_prefix']}  ultimo uso={k['last_used_at']}")


def revoke(key_id):
    token = login()
    st, resp = call("DELETE", f"/api-keys/{key_id}", token=token)
    print(f"HTTP {st}: {_safe_detail(resp)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["register"]:
        register()
    elif args[:1] == ["create-key"] and len(args) in (3, 4):
        create_key(*args[1:])
    elif args[:1] == ["list-keys"]:
        list_keys()
    elif args[:1] == ["revoke"] and len(args) == 2:
        revoke(args[1])
    else:
        print(__doc__)
