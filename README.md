# AI Services Reinstall Guide

Step-by-step guide to reinstall all AI-related services on a fresh Ubuntu setup.

## Table of Contents

- [Hardware Reference](#hardware-reference) — machine specs, disk layout
- [0. Before You Start](#0-before-you-start) — git setup, cloning this repo, what is *not* in the repo (data, secrets)
- [1. NVIDIA Driver & CUDA Setup](#1-nvidia-driver--cuda-setup) — driver install, `nvidia-smi`, optional CUDA Toolkit
- [2. Ollama Installation](#2-ollama-installation) — install, model storage path, model list, global setting `OLLAMA_MAX_LOADED_MODELS=2`, 64K context variants (Modelfiles)
- [3. Open WebUI (Docker)](#3-open-webui-docker) — Docker install, container, restart policy, LAN/remote access, users
- [4. Post-Install Checklist](#4-post-install-checklist) — verification for the base stack and the agents stack
- [5. Mem0 — Agent Memory Layer](#5-mem0--agent-memory-layer-venv-graph-memory) — venv, Neo4j + APOC, embeddings model, `.env`, `config.py`, test
- [6. Git — `~/Projects/AI` repo](#6-git--projectsai-repo) — `.gitignore`, nested repos
- [7. CrewAI ↔ Mem0 Integration](#7-crewai--mem0-integration-venv-manual-tool) — Python 3.12 venv, dependencies and pins, integration script, troubleshooting
- [8. Paperclip Agent Manager](#8-paperclip-agent-manager-docker) — clone, `.env` secret, Docker Compose, restart policy, first login, verify, updating
- [9. Hermes Agent](#9-hermes-agent-orchestrator-for-the-crewai-team) — official installer, model choice (`gpt-oss:20b`), 64K variants, wizard choices, config fixes, validation tests, troubleshooting, first-install history (9.12)
- [10. Shared Memory — Mem0 Server + Qdrant Server](#10-shared-memory--mem0-server-docker--qdrant-server) — Qdrant in Docker, patched Mem0 server (Ollama + Qdrant), Hermes memory provider, auto-sync block, `SOUL.md` memory rules, weekly validation report + timer
- [Pending / To Investigate](#pending--to-investigate) — open items
- [Notes](#notes) — cross-cutting reminders

## Hardware Reference

| Component   | Spec                                                                   |
| ----------- | ---------------------------------------------------------------------- |
| CPU         | AMD Ryzen 9 5950X                                                      |
| RAM         | 32GB 3600MHz                                                           |
| GPU         | GeForce RTX 4080 SUPER 16GB                                            |
| Motherboard | Aorus X570 Elite                                                       |
| OS Drive    | Samsung SSD 990 PRO 2TB (500GB Ubuntu / 1.3TB Windows / 200GB Bazzite) |

> This guide targets the **Ubuntu** partition (500GB). Ubuntu 26.04 LTS ("resolute") ships only Python 3.14 by default — see Section 7.1 for why a second Python version is needed for CrewAI.

---

## 0. Before You Start

### 0.1 Clone this repo

The scripts and config files referenced in this guide (e.g. `ai-agents/mem0/config.py`, `ai-agents/crewai/crewai_mem0_example.py`) live in this repo. On a fresh install, clone it first:

```
sudo apt install git
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
mkdir -p ~/Projects
git clone https://github.com/rizzo9555/AI.git ~/Projects/AI
```

### 0.2 What is *not* in this repo

By design (see `.gitignore`, Section 6), the repo holds no data and no secrets. These must be backed up **before** wiping the Ubuntu partition, and restored afterwards:

- Open WebUI data — Docker volume `open-webui` (users, chats, settings)
- Shared memories (current, Section 10) — `ai-agents/qdrant/storage/` (Qdrant server data) and `ai-agents/mem0-server/postgres-data/` + `ai-agents/mem0-server/history/` (Mem0 server auth DB and audit trail)
- Legacy Mem0 data (Section 5, being retired) — `ai-agents/mem0/neo4j/data/` (empty) and `ai-agents/mem0/qdrant_data/` (test data only)
- Paperclip data — `ai-agents/paperclip/data/`
- Hermes Agent config/data — `~/.hermes/` (`config.yaml`, `.env`, `mem0.json`, `SOUL.md`, sessions, memories); outside the repo by design (Section 9.1). The memory-related pieces are also written out in Section 10.
- Every `.env` file (Neo4j password, Paperclip's `BETTER_AUTH_SECRET`, Mem0 server's `POSTGRES_PASSWORD`/`JWT_SECRET`/`ADMIN_API_KEY`) — also keep these values in a password manager
- Optional: Ollama models (`/usr/share/ollama/.ollama/models`, ~110GB) — re-downloadable, just slow

> The backup/restore procedure itself isn't written yet — see Pending.

---

## 1. NVIDIA Driver & CUDA Setup

Install the latest recommended NVIDIA driver:

```
sudo apt update
sudo ubuntu-drivers install
sudo reboot
```
> `ubuntu-drivers install` is the current form of the older `ubuntu-drivers autoinstall` (deprecated alias, same result).

Verify the driver installation:

```
nvidia-smi
```

### Optional: CUDA Toolkit

**Not needed by anything in this guide** — Ollama ships its own CUDA runtime libraries, and nothing here compiles CUDA code. Only install it if a future tool needs `nvcc`. If so, follow NVIDIA's instructions (https://developer.nvidia.com/cuda-downloads) but install only the **`cuda-toolkit`** package — not the `cuda` meta-package, which also installs NVIDIA's own driver and can conflict with the one installed by `ubuntu-drivers` above.

Verify:

```
nvcc --version
```

---

## 2. Ollama Installation

Install Ollama:

```
curl -fsSL https://ollama.com/install.sh | sh
```

Verify the service is running:

```
systemctl status ollama
```

### 2.1 Model Storage Location

Models downloaded via `ollama pull` are stored at:

```
/usr/share/ollama/.ollama/models
```

### 2.2 Reinstalling Models

Pull each model back down:

```
ollama pull qwen3-coder:30b
ollama pull qwen3.8:27b
ollama pull qwen2.5-coder:14b
ollama pull gpt-oss:20b
ollama pull deepseek-r1:14b
ollama pull gemma4:26b
ollama pull gemma4:12b
ollama pull qwen3:14b
ollama pull nomic-embed-text   # embeddings model required by Mem0 (Section 5.3)
```

Plus one model pulled via Hugging Face GGUF instead of the Ollama library. **Currently unused** — it was the orchestrator of the first (reverted) Hermes Agent install, and it **can't be used with Hermes Agent through Ollama**: its Qwen3 base caps context at 40,960 tokens, below Hermes's 64K minimum (Section 9.2). **Kept on purpose** (decided 2026-09-28) for a future test: extending it to 64K via YaRN, probably with llama.cpp directly and a `q8_0` KV cache, to compare against `gpt-oss:20b-64k` (see Pending). On a reinstall, pull it only when that test is on the table:

```
ollama pull hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M
```

Verify installed models:

```
ollama list
```

Expected: **10 models** (9 from the Ollama library + the Hermes GGUF), plus the **2 `-64k` variants** once Section 2.4 is done — 12 entries total.

### 2.3 Global Setting: `OLLAMA_MAX_LOADED_MODELS=2`

Caps how many models Ollama keeps resident at the same time. Ollama only loads a second model if its memory estimate says it fits next to the first; otherwise it unloads the first one. So `2` is a ceiling, not an obligation, and the 16GB GPU is never overcommitted.

**Why 2 and not 1 (changed 2026-09-28):** with `1`, every Mem0 memory search or add (Section 10) loads the small embedder `nomic-embed-text` (~0.4GB), which **evicts the 12GB orchestrator** `gpt-oss:20b-64k` from VRAM — so Hermes would pay a full model reload on every turn. With `2`, the embedder stays loaded next to the orchestrator. Measured with `ollama ps`: `gpt-oss:20b-64k` 12GB **100% GPU** + `nomic-embed-text` 397MB at **74%/26% CPU/GPU** (it doesn't fully fit in the VRAM left over, but a tiny embedder on the Ryzen CPU is fast enough — milliseconds per search vs. a 12GB reload). Two *big* models still never coexist: `gpt-oss` (12GB) + `gemma4:12b-64k` (8.5GB) don't fit, so Ollama keeps swapping them (Section 9.9).

History: first set to `1` during the first Hermes Agent install; kept through the uninstall/reinstall; raised to `2` when the Mem0 server was introduced.

```
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null << 'EOF'
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=2"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Verify it took effect (should print `OLLAMA_MAX_LOADED_MODELS=2`):

```
systemctl show ollama -p Environment --value | tr ' ' '\n' | grep OLLAMA
```

(Prints one variable per line, so nothing gets cut at the screen edge.)

Verify the coexistence in practice (expect two lines in `ollama ps`, `gpt-oss:20b-64k` at `100% GPU`):

```
ollama run gpt-oss:20b-64k "responda apenas: ok"
curl -s http://localhost:11434/api/embed -d '{"model":"nomic-embed-text","input":"teste"}' > /dev/null
ollama ps
```
> `tee` **overwrites** the whole `override.conf`. If more Ollama variables are added later, put them all in this same file, one `Environment=` line each.
> `systemctl restart ollama` unloads every model — an answer in progress in Hermes/Open WebUI gets interrupted.

### 2.4 64K Context Variants (Modelfiles)

Some tools (Hermes Agent, Section 9) talk to Ollama through its OpenAI-compatible `/v1` endpoint, which **ignores per-request context size** (`num_ctx`) — the model silently loads at Ollama's default (4096), even if the tool displays a bigger number. The fix is a model *variant* with the context baked in via a Modelfile. Variants reuse the base model's files (no extra disk space) and don't affect other apps (Open WebUI, Mem0), which keep using the base models.

The Modelfiles are versioned in this repo at `~/Projects/AI/modelfiles/`. Recreate the variants after Section 2.2 (terminal, no venv needed):

```
ollama create gpt-oss:20b-64k -f ~/Projects/AI/modelfiles/gpt-oss-20b-64k.Modelfile
ollama create gemma4:12b-64k -f ~/Projects/AI/modelfiles/gemma4-12b-64k.Modelfile
```

Each Modelfile is just two lines, e.g.:

```
FROM gpt-oss:20b
PARAMETER num_ctx 65536
```

Verify the real loaded context (column `CONTEXT` should read `65536`):

```
ollama run gpt-oss:20b-64k "responda só: ok"
ollama ps
```

> Ollama **caps** `num_ctx` at the model's trained maximum — a variant can't push a model past its native context. E.g. `qwen3:14b` stays at 40,960 even with `num_ctx 65536` (verified with `ollama ps`). Check a model's native maximum with `ollama show <model>` (line `context length`).

---

## 3. Open WebUI (Docker)

> **Why Ollama stays native and isn't dockerized:** with a dedicated NVIDIA GPU, native Ollama uses the system driver directly with zero extra config. Dockerizing it would require installing and maintaining the NVIDIA Container Toolkit just for GPU passthrough, with no real benefit on a single-machine setup. Open WebUI itself doesn't touch the GPU (it's just the web interface), so only it needs to be containerized — the NVIDIA Container Toolkit step is skipped entirely.

### 3.1 Install Docker

Using Docker's official install script (simpler than the manual apt-repository method, works across distros):

```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

Allow running Docker without `sudo` (optional but recommended):

```
sudo usermod -aG docker $USER
```
> Log out and back in (or restart the terminal/session) for the group change to take effect.

Confirm the Docker service is running and enabled to start on boot:

```
sudo systemctl status docker
sudo systemctl is-enabled docker   # should say "enabled"; if not: sudo systemctl enable docker
```

Verify Docker:

```
docker run hello-world
```

### 3.2 Run Open WebUI

Since Ollama runs natively (not in Docker), Open WebUI is run with `--network=host` so the container shares the host's network stack and can reach Ollama directly at `127.0.0.1:11434` — no port mapping or `host.docker.internal` needed on Linux.

```
docker run -d \
  --name open-webui \
  --network=host \
  -v open-webui:/app/backend/data \
  -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

Access the interface at:

```
http://localhost:8080
```
> Note: `--network=host` is Linux-specific and is why the port is 8080 directly (no `-p` mapping) rather than remapped to 3000 as on Mac/Windows setups.
> **Restart policy — standard for every container in this guide: `unless-stopped`.** The container comes back after a crash or a reboot, but stays down after a manual `docker stop`. (`always` would bring it back on the next Docker/machine restart even after a manual stop.) Standardized on 2026-09-27; an existing container can be switched without recreating it: `docker update --restart unless-stopped open-webui`.

### 3.3 Verify

```
docker ps
docker logs open-webui
```

### 3.4 LAN Access (other devices at home)

Find the machine's local IP:

```
ip addr show | grep "inet " | grep -v 127.0.0.1
```

From another device on the same network, open `http://<LOCAL_IP>:8080`. If it doesn't connect, check the firewall:

```
sudo ufw status
sudo ufw allow 8080/tcp   # if ufw is active and the port isn't allowed
```
> Consider setting a DHCP reservation for this machine on the router so the local IP doesn't change on reboot.
> **ufw only governs Open WebUI here because it uses the host network.** Containers that publish ports with `-p` (Neo4j, Paperclip) bypass ufw entirely — Docker writes its own iptables rules — which is why Neo4j is bound to `127.0.0.1` in Section 5.2.

### 3.5 Remote Access (internet)

Not yet configured. Preferred approach when needed: a VPN (e.g. Tailscale) rather than a public reverse proxy — no ports exposed to the internet, simplest to set up for personal use.

### 3.6 First Login & Users

The first account created at `http://localhost:8080` becomes the admin automatically. Additional users (e.g. family members) are added manually via *Admin Panel → Users → Add User*. The "email" field is required as a login identifier but doesn't need to be a real, working address (e.g. `name@family.local` works fine — nothing is sent to it).

---

## 4. Post-Install Checklist

**Base stack (Sections 1–3):**

- [ ] `nvidia-smi` shows the RTX 4080 SUPER correctly
- [ ] `ollama list` shows all 10 models (Section 2.2) plus the 2 `-64k` variants (Section 2.4)
- [ ] `systemctl show ollama -p Environment --value | tr ' ' '\n' | grep OLLAMA` prints `OLLAMA_MAX_LOADED_MODELS=2` (Section 2.3)
- [ ] `sudo systemctl is-enabled docker` returns `enabled`
- [ ] Open WebUI loads at `http://localhost:8080`
- [ ] Open WebUI can see and query the Ollama models
- [ ] GPU usage confirmed during inference (`nvidia-smi` while running a prompt)
- [ ] Open WebUI reachable from another device via `http://<LOCAL_IP>:8080`
- [ ] Admin account created; additional family user accounts added

**Agents stack (Sections 5–10)** — check after finishing those sections:

- [ ] `crewai_mem0_example.py` completes both rounds (Section 7.3, CrewAI `venv` active) — until the Crew tool is migrated to the Mem0 server API (Pending)
- [ ] Paperclip loads at `http://localhost:3100`, and the checks in Section 8.6 print `ENV OK` and `SECRET MATCHES`
- [ ] Hermes Agent: `hermes doctor` shows no `✗` lines, and the 4 tests in Section 9.10 pass (basic chat, `-c 65536` in the Ollama logs, tool calling, vision)
- [ ] Qdrant answers at `curl -s http://localhost:6333/` (Section 10.2)
- [ ] Mem0 server: `/docs` returns HTTP 200, a request without key returns 401, and add/search work (Section 10.6)
- [ ] `hermes memory status` shows `Provider: mem0` + `available ✓`, and the canary test passes (Section 10.8)
- [ ] `systemctl --user list-timers mem0-report.timer --no-pager` shows the next Monday run (Section 10.9)
- [ ] Every running container uses the `unless-stopped` restart policy (`neo4j-mem0` is intentionally stopped with `no` — Section 5.2):

```
docker inspect -f '{{.Name}} {{.HostConfig.RestartPolicy.Name}}' open-webui docker-paperclip-1 qdrant mem0-server mem0-postgres neo4j-mem0
```

---

## 5. Mem0 — Agent Memory Layer (venv, Graph Memory)

> **⚠️ Legacy — superseded by Section 10 (2026-09-29).** The shared memory now runs as a **Mem0 server in Docker** backed by a **Qdrant server**, used by both Hermes and (soon) the Crew. Two findings retired this section:
> 1. **Mem0 2.0.0 (2026-04-14) removed external graph stores (Neo4j, Memgraph, Kuzu, Apache AGE) from the open-source SDK.** Graph memory became built-in *entity linking*: entities are extracted with spaCy and stored in a parallel `{collection}_entities` collection inside the vector store. The `graph_store` block below is **silently ignored** by every `mem0ai` 2.x — confirmed on 2026-09-28: the Neo4j database had 0 nodes and 0 relationships despite memories being written.
> 2. Embedded Qdrant can only be opened by one process at a time, so it can't be shared between Hermes and the Crew.
>
> Kept below as a record of how the library setup worked. Its data (`qdrant_data/`, 3 test memories) and the `venv-mem0` venv are scheduled for removal once the Crew is migrated (see Pending). **Neo4j** (`neo4j-mem0`) is **stopped** with restart policy `no` — kept (empty) only while Graphiti, a temporal knowledge-graph memory that runs on Neo4j, is evaluated (see Pending).

Gives agents (starting with CrewAI) persistent memory: a vector store (Qdrant, embedded/local) for semantic recall plus a graph store (Neo4j) for entity/relationship memory. Runs as a Python library inside a dedicated venv — not the official Docker server bundle, since that bundle only supports OpenAI/Anthropic/Gemini out of the box (solved later by patching it — Section 10). Fully local via Ollama.

### 5.1 Create the venv and install Mem0

```
mkdir -p ~/Projects/AI/ai-agents/mem0 && cd ~/Projects/AI/ai-agents/mem0
python3 -m venv venv-mem0
source venv-mem0/bin/activate
pip install mem0ai ollama neo4j langchain-neo4j python-dotenv
```
> Note: as of `mem0ai` 2.2.0, the `[graph]` install extra was dropped — the `neo4j`/`langchain-neo4j` packages must be installed manually, as above. The `ollama` package (official Python client) is also required separately for the Ollama embedder to work.
> **Watch out — `mem0ai[extras]` is not for fastembed/BM25.** It's a bundle for cloud vector-store integrations (AWS Bedrock, OpenSearch, Elasticsearch) and pulls in `boto3`, `elasticsearch`, `opensearch-py`, plus older `langchain`/`langchain-community` packages. If `langgraph`/`langchain-neo4j` are installed in the same venv, this downgrades `langchain-core` and breaks them. For BM25 keyword search, install `fastembed` directly instead — see Section 7.2.1.
> **Version drift (pending):** this venv installs `mem0ai` unpinned (2.2.0 as of 2026-09-27), while the CrewAI venv pins `2.0.14` (Section 7.2) — and both read/write the same Qdrant and Neo4j data. Works so far; aligning them is listed under Pending.

### 5.2 Run Neo4j locally (Docker, with APOC plugin)

Graph Memory requires the **APOC** plugin enabled:

```
mkdir -p ~/Projects/AI/ai-agents/mem0/neo4j/data ~/Projects/AI/ai-agents/mem0/neo4j/plugins

docker run -d \
  --name neo4j-mem0 \
  -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
  -v ~/Projects/AI/ai-agents/mem0/neo4j/data:/data \
  -v ~/Projects/AI/ai-agents/mem0/neo4j/plugins:/plugins \
  -e NEO4J_AUTH=neo4j/CHANGE_ME_ON_FIRST_BOOT \
  -e NEO4J_apoc_export_file_enabled=true \
  -e NEO4J_apoc_import_file_enabled=true \
  -e NEO4J_apoc_import_file_use__neo4j__config=true \
  -e NEO4J_PLUGINS='["apoc"]' \
  --restart unless-stopped \
  neo4j:2026.09.0
```

- Port `7474`: Neo4j Browser (`http://localhost:7474`)
- Port `7687`: Bolt protocol (used by Mem0 to connect)
- `NEO4J_AUTH` only sets the password on **first boot of an empty data volume**. To change the password later without losing data, log into the Browser and run: `ALTER CURRENT USER SET PASSWORD FROM 'old' TO 'new';`
- `127.0.0.1:` in front of each port — only this machine can reach Neo4j. Mem0 connects via `localhost`, so nothing changes for it. Without the prefix, Docker opens the ports to the whole LAN regardless of ufw (see 3.4).
- `NEO4J_PLUGINS` — current variable name. The older `NEO4JLABS_PLUGINS` (Neo4j 4.x) still works but logs a rename warning on every start.
- `neo4j:2026.09.0` — pinned to the version in use as of 2026-09-27 (`docker exec neo4j-mem0 neo4j --version`) instead of `latest`, so a reinstall doesn't silently jump to a newer release. Upgrade deliberately.

> The container currently running was created with the earlier flags (`NEO4JLABS_PLUGINS`, ports on all interfaces, `neo4j:latest`; restart policy later switched to `unless-stopped` via `docker update`). The data is safe either way, since it lives in the bind-mounted `neo4j/data` folder.

**Current state (2026-09-28): stopped.** Mem0 2.x no longer writes to Neo4j (see the note at the top of Section 5), and the old container had port 7687 open to the whole LAN. It was stopped without deleting anything:

```
docker update --restart no neo4j-mem0
docker stop neo4j-mem0
```

To bring it back (e.g. for the Graphiti evaluation): `docker start neo4j-mem0`, then recreate it with the command above if it's going to be kept.

### 5.3 Ollama models required

```
ollama pull nomic-embed-text
```

(Reuses an existing chat/reasoning model, e.g. `qwen3:14b`, for entity/fact extraction — no separate pull needed.)

### 5.4 Secrets — `.env`

Create `~/Projects/AI/ai-agents/mem0/.env` (gitignored — see Section 6):

```
NEO4J_PASSWORD=your_real_password_here
```

Keep a `.env.example` (no real value, safe to commit) alongside it:

```
NEO4J_PASSWORD=
```

### 5.5 `config.py`

```
import os
from dotenv import load_dotenv
from mem0 import Memory

load_dotenv()

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen3:14b",
            "ollama_base_url": "http://localhost:11434"
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0_memories",
            "embedding_model_dims": 768,
            "path": "/home/guilherme/Projects/AI/ai-agents/mem0/qdrant_data"
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": os.getenv("NEO4J_PASSWORD"),
            "database": "neo4j"
        }
    },
    "version": "v1.1"
}

memory = Memory.from_config(config_dict=config)
```
> `embedding_model_dims: 768` matches `nomic-embed-text`'s output size — Mem0's Qdrant default (1536) is sized for OpenAI embeddings and must be overridden or `add`/`search` will fail with a dimension mismatch.
> **Important — Qdrant is running in local/embedded mode, not as a server.** Because `vector_store.config` uses `path` (not `host`/`port`), Qdrant has no separate process — it's a set of files on disk that Python opens directly. This means **only one process can have it open at a time**. Don't run a Mem0 standalone script and the CrewAI integration (Section 7) at the same time — one will fail to open the locked files.

### 5.6 Test

```
# test_mem0.py
from config import memory

conversation = [
    {"role": "user", "content": "Uso Ollama local com uma RTX 4080 Super de 16GB."},
    {"role": "assistant", "content": "Entendido, vou lembrar disso."}
]

memory.add(conversation, user_id="rizzo")

results = memory.search(
    "Qual GPU eu uso?",
    filters={"user_id": "rizzo"},
    limit=3
)
for hit in results["results"]:
    print(hit["memory"])
```

```
python test_mem0.py
```

Verify the graph side by opening `http://localhost:7474` and running `MATCH (n) RETURN n LIMIT 25;` — connected nodes confirm Graph Memory is writing correctly.
> ⚠️ Obsolete with `mem0ai` 2.x: this query returns nothing, because the SDK no longer writes to Neo4j (see the note at the top of Section 5). Entities now live in the `mem0_memories_entities` collection in Qdrant.
> Note: `search()`/`get_all()` require `user_id` inside `filters={}` — passing it as a top-level kwarg (as `add()`/`delete_all()` still accept) raises a `ValueError` on current versions. This applies across `mem0ai` 2.x releases, at least down to `2.0.14`.

---

## 6. Git — `~/Projects/AI` repo

`~/Projects/AI` is a git repository (cloned in Section 0.1). Its root `.gitignore` contains:

```
# Python virtual environments
**/venv*/
__pycache__/
*.pyc

# Local databases (data, not source)
ai-agents/mem0/neo4j/data/
ai-agents/mem0/neo4j/plugins/
ai-agents/mem0/qdrant_data/
ai-agents/qdrant/storage/

# Nested git repos (Paperclip and Hermes Agent are each cloned from their own
# upstream repo — see Sections 8 and 9)
ai-agents/paperclip/
ai-agents/hermes-agent/

# Mem0 server (Section 10): upstream source clone, secrets, runtime data, reports
ai-agents/mem0-server-src/
ai-agents/mem0-server/.env
ai-agents/mem0-server/postgres-data/
ai-agents/mem0-server/history/
ai-agents/mem0-server/reports/

# Secrets
.env
```
> **Nested repo note:** `ai-agents/paperclip/`, `ai-agents/hermes-agent/` and `ai-agents/mem0-server-src/` are each a git clone of their own upstream repo, with their own `.git`. The lines above stop the main `~/Projects/AI` repo from tracking any of them. Paperclip's own upstream `.gitignore` also already ignores its `.env` and `data/`, so neither can be committed by accident from inside that repo either. Hermes Agent doesn't have this issue — its config/secrets live entirely outside the repo, at `~/.hermes` (Section 9.1). Our own Mem0 server files (`ai-agents/mem0-server/`: Dockerfile, compose, patch, report script, systemd units) **are** tracked; only its `.env`, data and reports are not.
> The bare `.env` line already matches `.env` files in any folder; `ai-agents/mem0-server/.env` is listed explicitly anyway, as documentation.
> Reports (`reports/`) are ignored because they contain the text of the memories themselves.
> Fixed 2026-09-27: the committed `.gitignore` used to contain the literal `cat > … << 'EOF'` and `EOF` lines of the heredoc command that was meant to *create* it. The heredoc is a terminal command — only the lines between those two markers belong in the file.

---

## 7. CrewAI ↔ Mem0 Integration (venv, manual tool)

> **Migration pending (next stage):** the tool below still opens the **legacy embedded Qdrant** (Section 5) through the `mem0` library. It will be rewritten to call the **Mem0 server REST API** (Section 10) with `user_id: rizzo` and a Crew-specific `agent_id`, so the Crew and Hermes share one memory. After that, `mem0ai`/`spaCy`/`fastembed` can be removed from this venv (they live inside the Mem0 server image now).

Gives a CrewAI agent access to the Mem0 memory set up in Section 5, so it can recall facts/preferences across runs. **CrewAI has no built-in native support for a `"provider": "mem0"` memory backend** — confirmed by grepping the entire installed `crewai`/`crewai-core` source (version 1.15.22) for the string `"mem0"`: zero matches outside the `mem0` package itself. A `memory_config={"provider": "mem0", ...}` is silently ignored by this CrewAI version; it falls back to its own default (ChromaDB-based) memory, which then fails due to the `posthog`/`chromadb` version conflict noted in 7.4. Integration here is done manually instead, via a custom Tool.

### 7.1 Create the venv (separate from Mem0's) with Python 3.12

Ubuntu 26.04 ships only Python 3.14 by default, which lacks pre-built wheels for several CrewAI dependencies (e.g. `tiktoken`, which requires a Rust compiler to build from source on 3.14). Python 3.12 is used instead, installed via the Deadsnakes PPA:

```
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv
```

Then create the venv:

```
mkdir -p ~/Projects/AI/ai-agents/crewai && cd ~/Projects/AI/ai-agents/crewai
python3.12 -m venv venv
source venv/bin/activate
```

### 7.2 Install dependencies

Install in separate steps (installing everything in one `pip install` line can trigger a `resolution-too-deep` error — CrewAI's dependency tree, combined with `langchain-neo4j`, is too complex for pip to solve in one pass):

```
pip install --upgrade pip setuptools wheel
pip install crewai
pip install mem0ai==2.0.14
pip install python-dotenv
pip install neo4j
pip install langchain-neo4j
```
> `mem0ai` is pinned to `2.0.14` here for stability, though this pin turned out not to be the fix for the integration issue (see 7.4) — the `mem0` library itself works fine with any recent 2.x version, as long as `search()`/`get_all()` calls use `filters={"user_id": ...}` (see Section 5.6's note).

`mem0ai` also requires its own `.env` with `NEO4J_PASSWORD` — same value as Section 5.4, copy `~/Projects/AI/ai-agents/mem0/.env` or create a fresh one in the `crewai` folder. A `.env.example` (no real value) is committed in the `crewai` folder as a template.

`crewai` installs `chromadb`, which pins `posthog<6.0.0`; `mem0ai` has no upper bound on `posthog` and pulls the latest (7.x) by default. Pin it down explicitly after installing both:

```
pip install "posthog<6.0.0"
```

This leaves a cosmetic `pip check` warning (`mem0ai requires posthog>=7.14.0, but you have posthog 5.4.0`) — see 7.4 for why this is safe to ignore.

#### 7.2.1 Optional extras: spaCy (entity extraction) + fastembed (BM25 keyword search)

```
pip install "mem0ai[nlp]==2.0.14"
pip install fastembed
```
> Same `mem0ai` pin as in 7.2 — keeps pip from switching the version while adding the extra.

Both models download automatically on first use — no manual `spacy download` step needed: the `en_core_web_sm` spaCy model downloads the first time a tool call triggers entity extraction, and the fastembed BM25 model downloads on the first search. Confirmed working end-to-end (Section 7.3's script).
> Do **not** run `pip install "mem0ai[extras]"` for this — see the warning in Section 5.1. If it's already been run by mistake, recover with:
>
> ```
> pip uninstall -y langchain langchain-community elasticsearch elastic-transport opensearch-py opensearch-protobufs boto3 botocore s3transfer
> pip install --upgrade "langchain-core>=1.4.7,<2" "langchain-text-splitters>=1.1.2,<2" "posthog<6.0.0"
> ```

### 7.3 The integration script

`~/Projects/AI/ai-agents/crewai/crewai_mem0_example.py` — a minimal working example. Key points:

- `memory=False` on the `Crew` — disables CrewAI's own (broken) default memory system.
- A custom `BaseTool` (`buscar_memoria`) that the agent calls explicitly to search Mem0 (`mem0_client.search(query, filters={"user_id": USER_ID}, limit=5)`).
- A `salvar_memoria()` helper function, called manually after each `crew.kickoff()`, that writes the turn to Mem0 (`mem0_client.add(..., user_id=USER_ID)`).
- The agent's LLM must be set **explicitly** to Ollama — `memory_config`/Mem0 setup has no effect on which LLM the *agent* itself uses to reason. Without this, CrewAI defaults to OpenAI and fails with `OPENAI_API_KEY is required`:

```
from crewai import LLM
ollama_llm = LLM(model="ollama/qwen3:14b", base_url="http://localhost:11434")
# ...
agent = Agent(..., llm=ollama_llm, tools=[BuscarMemoriaTool()])
```

The full script is committed in this repo at `ai-agents/crewai/crewai_mem0_example.py`.

### 7.4 Troubleshooting notes (from setting this up)

- **`resolution-too-deep` on `pip install`**: install packages one at a time (Section 7.2), not all in one command.
- **`tiktoken` build fails needing a Rust compiler**: symptom of running on Python 3.14; switch to 3.12 (Section 7.1) rather than installing Rust.
- **Duplicated `(venv)` in the shell prompt** (e.g. `((venv) )`, `(venv) (venv)`): a known quirk when a venv is activated on top of another already-active one, or after a broken attempt to customize `PS1`. It's purely cosmetic (confirmed via `$VIRTUAL_ENV` and `which python3` — the correct interpreter is always used), but if it's distracting, the reliable fix is closing the terminal application entirely and opening a new one, then activating the venv once.
- **`OPENAI_API_KEY is required`**: the agent's `llm=` wasn't set explicitly — see Section 7.3.
- **`chromadb` requires `posthog<6.0.0`, but `mem0ai` requires `posthog>=7.14.0`**: a real conflict between CrewAI's `chromadb` dependency and `mem0ai`'s declared metadata — the two ranges don't overlap, so no single `posthog` version satisfies both `pip check`. Resolved by pinning `posthog<6.0.0` (Section 7.2), which leaves a residual `pip check` warning from `mem0ai`'s side. This is safe in practice: both packages only use `posthog` for anonymous telemetry (simple `capture()` calls), an API that's been stable across major versions, and this was confirmed by running the integration script (Section 7.3) end-to-end with `posthog` 5.4.0 — `add()` and `search()` both worked with no exceptions. The theoretical risk is a future `mem0ai` release calling a `posthog` 7.x-only feature outside the paths already tested here; if that ever surfaces, the more robust fix is disabling Mem0's telemetry entirely (`MEM0_TELEMETRY=false` in `.env`), which removes the dependency on `posthog`'s version for `mem0ai`'s side — not applied yet, listed under Pending.
- **`mem0ai[extras]` breaks `langchain-core`/`langgraph`**: see the warning in Section 5.1 and the recovery command in 7.2.1. Installing `mem0ai[extras]` for BM25 support is the wrong flag — it's meant for AWS/OpenSearch/Elasticsearch integrations — and pulls an old `langchain`/`langchain-community` that downgrades `langchain-core`, breaking anything in the venv that needs `langchain-core>=1.x` (`langgraph`, `langchain-neo4j`, `langchain-classic`, `langgraph-sdk`, `langgraph-prebuilt`).
- **`memory_save_failed` warning with "empty scope stack"**: misleading — this came from CrewAI's default (ChromaDB) memory failing silently in the background (see `posthog` conflict above), not from Mem0. It disappeared once `memory=False` + the manual tool approach (Section 7.3) replaced the native `memory_config`.
- **Qdrant appears to not be running (`docker ps` doesn't show it, nothing on port 6333)**: expected — it's running in local/embedded mode (Section 5.6), not as a server.

---

## 8. Paperclip Agent Manager (Docker)

![Paperclip](paperclip.png)

Open-source orchestration platform ([paperclipai/paperclip](https://github.com/paperclipai/paperclip)) that manages a team of AI agents (CrewAI, Claude Code, Codex, etc.) like employees in a company — org chart, tickets, budgets, governance. Will also be used in the future "Get Contractors Now" project.
> **Why Docker and not native (Node/pnpm):** Paperclip doesn't touch the GPU, so the reason Ollama stays native doesn't apply here. Docker was chosen for isolation and portability, matching the Open WebUI approach — the official install path builds the image locally from source (no pre-built image to just pull), so the repo still needs to be cloned either way.

### 8.1 Clone the repo

```
cd ~/Projects/AI/ai-agents
git clone https://github.com/paperclipai/paperclip.git
cd paperclip
```

This creates a **nested git repo** inside `~/Projects/AI` — see the `.gitignore` note in Section 6.

### 8.2 Create `.env` with the auth secret

`BETTER_AUTH_SECRET` (session/auth signing key) is required — the quickstart compose file refuses to start without it. Generate it once and keep it in `.env` at the repo root:

```
cd ~/Projects/AI/ai-agents/paperclip
echo "BETTER_AUTH_SECRET=$(openssl rand -hex 32)" > .env
```

- On a reinstall, restore the **old** value from backup instead of generating a new one — a new secret at least invalidates every existing login session.
- `.env` is already ignored by Paperclip's own upstream `.gitignore`, and the outer repo ignores the whole folder (Section 6).

> History: the current install (2026-09-25) passed the secret inline on the first `docker compose up` and saved it to `.env` afterwards, reading it back with `docker exec docker-paperclip-1 env | grep BETTER_AUTH_SECRET`. Same end result.

### 8.3 Run via Docker Compose (official quickstart)

```
cd ~/Projects/AI/ai-agents/paperclip
docker compose --env-file .env -f docker/docker-compose.quickstart.yml up -d
```

- **`--env-file .env` is required.** With `-f docker/...`, Compose looks for `.env` in the compose file's own folder (`docker/`), not in the current folder — without the flag it fails with `required variable BETTER_AUTH_SECRET is missing a value` (confirmed 2026-09-27).
- First run builds the image from the repo's `Dockerfile` (slower); later runs reuse the built image (see 8.7 for updates).
- Persistent data (embedded PostgreSQL, uploads, secrets key, agent workspace data) lives under `data/docker-paperclip/` at the repo root — the compose file's default `../data/docker-paperclip`, resolved from `docker/`. Ignored by upstream's `.gitignore` (`data/`) and by the outer repo (Section 6).
- Access at `http://localhost:3100`.
- Port 3100 is published on all interfaces, so it's open to the LAN regardless of ufw (see 3.4). Paperclip itself still rejects hostnames other than `localhost` until LAN access is configured (see Pending).

> **Container name:** Compose names it `docker-paperclip-1` (derived from the compose file's folder, `docker`, + the service name), **not** `paperclip`. Use the real name for `docker exec`/`docker update` below — check with `docker ps` if unsure.

### 8.4 Keep it always running (survive reboots)

```
docker update --restart unless-stopped docker-paperclip-1
```

Same policy as every container in this guide (see 3.2): restarts automatically after a crash or a reboot; only a manual `docker stop docker-paperclip-1` keeps it down.

> The quickstart compose file has no `restart:` key, so this setting is **lost whenever Compose recreates the container** (e.g. after an update, 8.7). Re-run the command above after every recreate.

### 8.5 First login

Open `http://localhost:3100`. The first account created on the setup screen automatically becomes the instance admin — same as Open WebUI (Section 3.6), the email field doesn't need to be real (no SMTP is configured, so nothing is sent/verified). **Done** — admin account created.

### 8.6 Verify

```
cd ~/Projects/AI/ai-agents/paperclip
docker ps                                   # confirms docker-paperclip-1 is Up
docker compose --env-file .env -f docker/docker-compose.quickstart.yml config > /dev/null && echo "ENV OK"
diff <(grep BETTER_AUTH_SECRET .env) <(docker exec docker-paperclip-1 env | grep BETTER_AUTH_SECRET) > /dev/null && echo "SECRET MATCHES" || echo "SECRET DIFFERS"
```

The last two lines check the setup without printing the secret: `ENV OK` means Compose can read `.env`; `SECRET MATCHES` means `.env` holds the same value the running container uses.

### 8.7 Updating Paperclip

> Not yet exercised on this install.

```
cd ~/Projects/AI/ai-agents/paperclip
git pull
docker compose --env-file .env -f docker/docker-compose.quickstart.yml up -d --build
docker update --restart unless-stopped docker-paperclip-1
```

`--build` forces a rebuild of the image from the updated source — without it, Compose keeps using the old image. The last line restores the restart policy (see 8.4).

---

## 9. Hermes Agent (Orchestrator for the CrewAI Team)

![Hermes](HermesChat.png)

Autonomous agent framework from Nous Research ([hermes-agent.nousresearch.com](https://hermes-agent.nousresearch.com)), used to orchestrate/delegate work to the CrewAI team (Section 7), particularly for the future "Get Contractors Now" project. Fully local via Ollama, same as everything else in this stack.

> **✅ Status (2026-09-27): reinstalled from scratch and validated** — tool calling and vision both working. A first install earlier the same day was fully reverted after validation problems; that history is condensed in Section 9.12.

**Summary of the working setup:**

| Role | Model | Measured (`ollama ps`) |
|---|---|---|
| Orchestrator (main model) | `gpt-oss:20b-64k` | 12 GB, 100% GPU, context 65,536 |
| Vision (auxiliary) | `gemma4:12b-64k` | 8.5 GB, 100% GPU, context 65,536 |

### 9.1 Install (official installer)

Terminal, no venv needed. Code goes inside the project folder (convention); config/secrets/sessions stay at the default `~/.hermes`:

```
cd ~/Projects/AI
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --dir ~/Projects/AI/ai-agents/hermes-agent
```

- `bash -s --` is required to pass flags through `curl | bash`. Without `--dir`, the installer defaults to `~/.hermes/hermes-agent`.
- The installer provisions its own Python runtime/venv under `~/.hermes/installs/...` (Python 3.14 as of this install) and puts `hermes` in `~/.local/bin` — no manual venv activation ever needed to run `hermes`.
- Without `--skip-setup`, it goes straight into the setup wizard at the end (Section 9.4).

### 9.2 Model choice — why `gpt-oss:20b`

Hermes Agent **hard-requires ≥ 64,000 tokens of context**, and (current versions) checks the context Ollama *actually loaded*, not just the configured value. That rules out every Qwen3-based ~14B model:

| Candidate | Native context | Result |
|---|---|---|
| `qwen3:14b` | 40,960 | ❌ Ollama caps at 40,960 even with a 64K Modelfile; and at 40,960 it already spills to CPU (16 GB, 12%/88% CPU/GPU) |
| Hermes-4-14B (hf.co GGUF) | 40,960 (Qwen3 base) | ❌ same ceiling — no longer an option regardless of bug #110442 (Section 9.12) |
| `gemma4:12b` | 262,144 | Fits (8.5 GB @ 64K, 100% GPU) but **failed as orchestrator**: test 1 — tool command failed, then it *invented* the result; test 2 — no tool call at all, returned a blank template with a leaked `<channel|>` token (its native tool-call format not recognized through Ollama `/v1` → Hermes). Fine as a **vision** model. |
| `gpt-oss:20b` | 131,072 | ✅ **Chosen.** MoE (21B total, ~3.6B active per token → fast), trained for tool use. 12 GB @ 64K, 100% GPU. Correct tool calls, no loop, and it cross-checks tool output instead of blindly repeating it. |

Exception to the "~14B for daily use" rule: `gpt-oss:20b` is justified by its MoE design (small active size) and because it fits fully in VRAM at 64K.

### 9.3 Create the 64K variants (before the wizard)

See Section 2.4 — the `/v1` endpoint Hermes uses ignores per-request context, so the models must be `-64k` variants:

```
ollama create gpt-oss:20b-64k -f ~/Projects/AI/modelfiles/gpt-oss-20b-64k.Modelfile
ollama create gemma4:12b-64k -f ~/Projects/AI/modelfiles/gemma4-12b-64k.Modelfile
```

### 9.4 Setup wizard — key choices

Navigate only with arrows / Enter / Space / Esc — **never Ctrl+C** (Section 9.11).

| Prompt | Choice | Why |
|---|---|---|
| Setup mode | **Full setup** | "Quick Setup" logs into the Nous Portal cloud |
| Provider | **Custom endpoint (enter URL manually)** | Near the bottom of the list. Not "custom (direct API)" |
| API base URL | `http://localhost:11434/v1` | Ollama's OpenAI-compatible endpoint |
| API key | blank (Enter) | Marked optional; not needed locally |
| API compatibility mode | **2 — Chat Completions** | Skips Auto-detect's probing of Ollama-native routes (known source of silent hangs) |
| Model | any from the list | Will be switched to `gpt-oss:20b-64k` afterwards (9.7) — the wizard only lists what's installed at that moment |
| Context length | `65536` | Hermes's 64K minimum |
| Display name | default (`Local (localhost:11434)`) | Cosmetic |
| Reasoning effort | `medium` | Changeable per session with `/reasoning` |
| Terminal backend | **Keep current (local)** | Same machine |
| Messaging platforms | none (Enter with nothing checked) | Can be added later: `hermes setup gateway` |
| Tools (CLI) | see 9.5 | |
| Browser provider | **Local Browser** | Headless Chromium, free |
| Vision backend | Pick a provider and model → Local → Type a custom model id → `gemma4:12b` | ⚠️ **Not persisted by the wizard** — fixed in 9.7 |
| Search provider | **DuckDuckGo (ddgs)** | Free, no key (SearXNG deferred — see Pending) |

### 9.5 Tools

Turned **off** in the wizard: Image Generation, Text-to-Speech, Computer Use (desktop control — much broader than needed). Everything else at defaults, notably **Task Delegation** (`delegate_task`), the core capability here.

### 9.6 Vision backend — known wizard bugs

1. The model picker for the Local provider lists only the main text model (e.g. `qwen3:14b`) as "vision-capable", not the real multimodal models. Workaround in the wizard: "Type a custom model id…".
2. Even then, the choice is **not saved**: `config.yaml` ends up with `auxiliary.vision.provider: auto` / `model: ''`, and `hermes doctor` reports `vision (system dependency not met)`. This (not an `hf.co` capability issue, as first suspected) was the root cause of the `AuxiliaryClientUnavailable` error in the first install.
3. `provider: main` (suggested by comments inside `config.yaml`) is **invalid** in this version — doctor reports `Unknown provider 'main'`, and the task silently falls back to the main (text-only) model.

Fix: create a named provider and point vision to it (9.7).

### 9.7 Post-wizard config fixes

Back up first (terminal, no venv):

```
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak-pre-fix
```

Then:

```
hermes config set model.default gpt-oss:20b-64k
hermes config set model.context_length 65536
hermes config set model.ollama_num_ctx 65536
hermes config set providers.ollama-local.api http://localhost:11434/v1
hermes config set providers.ollama-local.api_mode chat_completions
hermes config set auxiliary.vision.provider custom:ollama-local
hermes config set auxiliary.vision.model gemma4:12b-64k
```

- `providers.ollama-local` is also the migration `hermes doctor` asks for (the wizard still writes the legacy `custom_providers:` list). Once it exists, the doctor warning disappears. `hermes doctor --fix` does **not** perform this migration.
- **Remove the legacy `custom_providers:` block afterwards** (done 2026-09-28). It's safe because `model:` carries its own `provider: "custom"` + `base_url` and vision uses `custom:ollama-local` — nothing active points to the legacy list. Back up, find the block's line numbers, delete them, confirm:
  ```
  cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak-$(date +%Y%m%d)
  grep -n -A8 -E '^(model|custom_providers|providers|auxiliary):' ~/.hermes/config.yaml
  sed -i '<first>,<last>d' ~/.hermes/config.yaml    # use the line numbers of the custom_providers block
  grep -n 'custom_providers' ~/.hermes/config.yaml  # only a comment line may remain
  ```
  Confirm the main model is self-contained first: `sed -n '57,200p' ~/.hermes/config.yaml | grep -vE '^\s*(#|$)'` should show `provider: "custom"` and `base_url` under `model:`.
- `ollama_num_ctx` is harmless but effectively redundant — the `-64k` variants are what actually set the context (Section 2.4).

Verify:

```
hermes doctor 2>&1 | grep -E "✗|custom_providers|vision"
```

Expected: `auxiliary task routing resolves: vision→custom@localhost` and `✓ vision`, no `✗` lines.

### 9.8 Search provider

**DuckDuckGo (ddgs)** — free, no API key. SearXNG self-hosted deferred (see Pending).

### 9.9 VRAM & model swapping

Hermes (`gpt-oss`, 12 GB) and its vision model (`gemma4`, 8.5 GB) don't fit in 16 GB together. With `OLLAMA_MAX_LOADED_MODELS=2` (Section 2.3), Ollama still refuses to load both big models at once — its memory check makes it swap them instead of spilling to CPU — while letting the small `nomic-embed-text` embedder stay loaded next to `gpt-oss` for memory searches (Section 10). In practice, each `vision_analyze` call costs ~30 s: unload `gpt-oss` → load `gemma4` → analyze → reload `gpt-oss`. Acceptable for occasional use.

> Ollama also unloads any model idle for 5 minutes (default `keep_alive`), so the first Hermes turn or Mem0 extraction after a pause pays a reload (~20 s measured).

> CrewAI corollary, once a Crew script exists: use `Process.sequential` so agents never call their models at the same time.

### 9.10 Validation tests (2026-09-27)

Terminal, no venv. After each `hermes chat -q ...`, Hermes stays in interactive mode — type `/exit` to leave.

1. **Basic chat:** `hermes chat -q "Responda em uma frase: qual é a capital do Brasil?"` → correct answer.
2. **Real context check** (not just what Hermes displays):
   ```
   ollama ps
   journalctl -u ollama --since "5 minutes ago" --no-pager | grep -E "starting llama-server" | grep -oE "\-\-model [^ ]+|-c [0-9]+"
   ```
   Every load must show `-c 65536`. (Before the Modelfile variants, the status bar showed `65.5K pinned` while Ollama was really loading `-c 4096`.)
3. **Tool calling:** `hermes chat -q "Liste as pastas que existem dentro de ~/Projects/AI e me diga quantas são. Responda em português."` → one `terminal` call, correct answer (compare with `ls -d ~/Projects/AI/*/`), no loop.
4. **Vision:** `hermes chat -q "Use a ferramenta vision_analyze na imagem /usr/share/backgrounds/<file>.png e descreva em português, em 2 frases, o que aparece nela."` → description returned, no `AuxiliaryClientUnavailable`; `ollama ps` afterwards shows `gpt-oss:20b-64k` loaded again.

### 9.11 Troubleshooting notes

- **Ctrl+C during `hermes setup` cancels the entire wizard** and falls into a Nous Portal login flow. If that happens: Ctrl+C again to stop the polling, re-run `hermes setup`.
- **Loop inside a chat:** use `/stop`, not Ctrl+C.
- **`❌ Ollama runtime context is too small for Hermes tool use`**: the model's real loaded context is < 64K. Check the model's native context (`ollama show <model>`) — if it's below 64K, no config can fix it; pick another model.
- **`hermes doctor` harmless warnings:** Nous Portal / Codex / xAI / MiniMax not logged in; OpenRouter not configured; Telegram/Discord packages not installed; several tools with "system dependency not met" (a2a, spotify, computer_use, etc. — disabled/unused); agent-browser npm vulnerability (upstream lockfile, #116774 — fixed by a future `hermes update`); Skills Hub not initialized.
- **`SyntaxWarning: "\W" is an invalid escape sequence`** from `pm/shell.py` when a tool runs — cosmetic, Hermes code vs. Python 3.14. Ignore.
- **`sudo systemctl edit ollama.service` doesn't save:** stale `nano` lock file in `/etc/systemd/system/ollama.service.d/.#override.conf...` — remove it and write the override directly (Section 2.3).

### 9.12 History — first install (reverted, 2026-09-27)

First attempt used **Hermes-4-14B** (`hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M`) as orchestrator, installed with `--skip-setup`. Problems found during validation:

1. Context: auto-detected at 40,960 (< 64K minimum). Worked around with `context_length`/`ollama_num_ctx: 65536` — which, as learned in the reinstall, Ollama never actually applied via `/v1`.
2. Tool-call loop: collision between the Qwen/Hermes native `<tool_call>` delimiter and Hermes Agent's bridge tool named `tool_call` — [issue #110442](https://github.com/NousResearch/hermes-agent/issues/110442), fix in [PR #110452](https://github.com/NousResearch/hermes-agent/pull/110452) (rename to `invoke_tool`); release status not confirmed. Moot now: the model can't meet the 64K minimum anyway.
3. Vision: `AuxiliaryClientUnavailable` — root cause later identified as the wizard not persisting the vision choice (9.6).

Reverted with:

```
hermes uninstall --yes
rm -rf ~/.hermes
rm -rf ~/Projects/AI/ai-agents/hermes-agent
```

`OLLAMA_MAX_LOADED_MODELS=1` (Section 2.3) was kept at the time (raised to `2` later).

---

## 10. Shared Memory — Mem0 Server (Docker) + Qdrant Server

> **✅ Status (2026-09-29): running and validated with Hermes.** CrewAI migration to the API is the next stage (Section 7 note).

One memory shared by Hermes and the CrewAI team: whatever one agent stores, the others can find. Everything runs locally; nothing reaches a cloud API.

```
Hermes (plugin mem0, self-hosted mode) ─┐
CrewAI tool (REST, pending)  ───────────┼─ HTTP + X-API-Key ─> Mem0 server  127.0.0.1:8888  (Docker, network host)
                                         │                        ├─ LLM (fact extraction): gpt-oss:20b-64k  ─┐
                                         │                        ├─ Embedder: nomic-embed-text (768 dims)  ──┴─ Ollama 127.0.0.1:11434
                                         │                        ├─ Vectors + entities: Qdrant 127.0.0.1:6333 (Docker)
                                         │                        └─ Auth/users/API keys/logs: Postgres 127.0.0.1:8432 (Docker)
```

Key facts that shaped the design:

- **No Neo4j.** Mem0 2.x dropped external graph stores; "graph memory" is now entity linking inside the vector store (`mem0_memories_entities` collection). See the note at the top of Section 5.
- **Extraction is ADD-only** in Mem0 2.x: memories accumulate, nothing is updated or deleted automatically. That makes redundancy the main risk — handled by the auto-sync block (10.8) and the weekly report (10.9).
- **Memories are written in English** by the extraction LLM, even from Portuguese input, and **cross-language search works**: the same question in PT and EN scored 0.376 vs 0.377 against the same memory. So the hard-coded English spaCy model (`en_core_web_sm`) is not a problem.
- **Isolation by IDs:** every client uses `user_id: rizzo` (that's what makes the memory shared) and its own `agent_id` (`hermes`, later one for the Crew) so it's always known who wrote what. Qdrant indexes `user_id`, `agent_id`, `run_id`, `actor_id`, so filtering is cheap.
- **Everything binds to `127.0.0.1`** — not reachable from the LAN.

### 10.1 Folder layout

```
~/Projects/AI/ai-agents/
├── qdrant/storage/            # Qdrant data (gitignored)
├── mem0-server-src/           # upstream clone, tag v2.2.1 (gitignored, nested repo)
└── mem0-server/               # ours, tracked by git
    ├── Dockerfile
    ├── docker-compose.yml
    ├── local_config.py        # Ollama + Qdrant config, read from env vars
    ├── patch_main.py          # 4 exact-text edits to upstream main.py, applied at build
    ├── mem0_report.py         # weekly validation report (10.9)
    ├── systemd/               # mem0-report.service / .timer (10.9)
    ├── .env.example           # template, no secrets
    ├── .env                   # secrets (gitignored)
    ├── postgres-data/         # gitignored
    ├── history/               # Mem0 audit trail, SQLite (gitignored)
    └── reports/               # weekly reports (gitignored — contains memory text)
```

### 10.2 Qdrant server

Terminal, no venv:

```
mkdir -p ~/Projects/AI/ai-agents/qdrant/storage
docker run -d --name qdrant --restart unless-stopped \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v $HOME/Projects/AI/ai-agents/qdrant/storage:/qdrant/storage \
  qdrant/qdrant:latest
curl -s http://localhost:6333/ ; echo
```

Expected: JSON with `"title":"qdrant - vector search engine"` (version 1.19.1 at install). Port 6333 = REST, 6334 = gRPC. Replaces the embedded Qdrant of Section 5 and can be shared by future projects (e.g. the legal RAG).

### 10.3 Upstream Mem0 server source (pinned)

```
cd ~/Projects/AI/ai-agents
git clone --depth 1 --branch v2.2.1 https://github.com/mem0ai/mem0.git mem0-server-src
```

The "detached HEAD" message is normal when cloning a tag. What the official server does, and why it can't be used as-is:

- FastAPI app (`server/main.py`); on start it builds a `Memory` from a hard-coded `DEFAULT_CONFIG` (**pgvector + OpenAI**) merged with overrides stored in Postgres — without an OpenAI key it breaks at boot.
- `POST /configure` rejects any LLM/embedder provider outside `openai`, `anthropic`, `gemini`.
- **Postgres is mandatory** even when vectors go to Qdrant: it holds users, API keys, request logs and settings in a `mem0_app` database (created by `server/init-db.sh`; tables by `alembic upgrade head`). No SQLite fallback.
- **Auth is on by default:** requires `JWT_SECRET`; clients send `ADMIN_API_KEY` in the `X-API-Key` header. `AUTH_DISABLED=true` is for local development only.
- Telemetry on by default; `MEM0_TELEMETRY=false` turns it off.
- An optional Next.js dashboard (port 3000) exists — not used.

### 10.4 Our files (in `ai-agents/mem0-server/`, tracked by git)

**`local_config.py`** — the config that replaces `DEFAULT_CONFIG`. Every value comes from an env var with a default:

| Setting | Env var | Default | Why |
|---|---|---|---|
| LLM (extraction) | `MEM0_LLM_MODEL` | `gpt-oss:20b-64k` | Same model as Hermes → no VRAM swap |
| LLM max output | `MEM0_LLM_MAX_TOKENS` | `4096` | Covers gpt-oss reasoning + a short JSON answer; caps runaway turns. Raise to 8192 if cut-off JSON errors ever show up in the logs (Pending) — env change + recreate, no rebuild |
| LLM temperature | — | `0.1` | Fact extraction should be predictable |
| Embedder | `MEM0_EMBED_MODEL` / `MEM0_EMBED_DIMS` | `nomic-embed-text:latest` / `768` | Must match everywhere; without `768` Mem0 assumes 1536 (OpenAI) and every insert fails |
| Ollama | `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | |
| Qdrant | `QDRANT_HOST` / `QDRANT_PORT` / `MEM0_COLLECTION` | `127.0.0.1` / `6333` / `mem0_memories` | |
| Auto-sync block | `MEM0_SKIP_INFER_AGENTS` | empty (nothing blocked) | See 10.8 |

**`patch_main.py`** — runs during the image build and makes **4 exact-text replacements** in upstream `main.py`, **failing the build on purpose** if any expected text isn't found exactly once (protects against silent breakage when upgrading Mem0):

1. `BUNDLED_LLM_PROVIDERS` += `"ollama"`
2. `BUNDLED_EMBEDDER_PROVIDERS` += `"ollama"`
3. Right before `initialize_state(DEFAULT_CONFIG)`: `from local_config import LOCAL_CONFIG as DEFAULT_CONFIG, SKIP_INFER_AGENTS as _SKIP_INFER_AGENTS` (the original `DEFAULT_CONFIG` stays in the file, unused)
4. In `POST /memories`, right after `params = {...}`: if `infer` is true and `agent_id` is in `MEM0_SKIP_INFER_AGENTS`, log `skip_infer: agent_id=...` and return `{"results": [], "skipped": "infer_blocked_for_agent"}` without calling the LLM

Test the patch against a copy before building (no venv, system Python):

```
cd ~/Projects/AI/ai-agents/mem0-server
cp ../mem0-server-src/server/main.py /tmp/main_test.py
python3 patch_main.py /tmp/main_test.py
grep -n -E 'BUNDLED_(LLM|EMBEDDER)_PROVIDERS =|LOCAL_CONFIG|skip_infer' /tmp/main_test.py
```

**`Dockerfile`** — `python:3.12-slim`; installs upstream `requirements.txt`, then pins **`mem0ai==2.2.1`** (upstream only requires `>=0.1.48`, which could pull a version that doesn't match the server code) plus `ollama`, `qdrant-client`, `fastembed` (BM25 keyword search) and `spacy` + `en_core_web_sm` (entity linking). Copies upstream `server/` through a named build context `upstream`, copies our two `.py` files, runs `patch_main.py`, and starts with `alembic upgrade head && uvicorn main:app --host 127.0.0.1 --port 8888` (no `--reload`, which upstream uses for development).

**`docker-compose.yml`** — two services, both `restart: unless-stopped`:

- `postgres` → container `mem0-postgres`, image `pgvector/pgvector:pg17` (same as upstream), data in `./postgres-data`, upstream `init-db.sh` mounted read-only into `/docker-entrypoint-initdb.d/`, port `127.0.0.1:8432:5432`, `pg_isready` healthcheck.
- `mem0` → container `mem0-server`, image `mem0-server-local:v2.2.1`, built with `additional_contexts: upstream: ../mem0-server-src/server` (needs Compose ≥ 2.17), `network_mode: host` (reaches Ollama, Qdrant and Postgres on `127.0.0.1`, same idea as Open WebUI), `env_file: .env`, `POSTGRES_HOST=127.0.0.1`, `POSTGRES_PORT=8432`, `APP_DB_NAME=mem0_app`, `./history` mounted, starts only after Postgres is healthy.

**`.env`** — generate on a fresh install (on a reinstall, restore the old values instead). `EOF` **without** quotes on purpose, so `$(openssl ...)` runs:

```
cd ~/Projects/AI/ai-agents/mem0-server
cat > .env << EOF
POSTGRES_USER=postgres
POSTGRES_PASSWORD=$(openssl rand -hex 24)
JWT_SECRET=$(openssl rand -hex 48)
ADMIN_API_KEY=$(openssl rand -hex 32)
AUTH_DISABLED=false
MEM0_TELEMETRY=false
MEM0_LLM_MAX_TOKENS=4096
MEM0_SKIP_INFER_AGENTS=hermes
EOF
chmod 600 .env
sed 's/=.*/=***/' .env    # check the variable names without printing values
```

### 10.5 Build and run

Terminal, no venv, inside `ai-agents/mem0-server`:

```
docker compose version                               # needs >= 2.17 (v5.5.1 at install)
docker compose config --quiet && echo "compose OK"
docker compose build mem0                            # ~1 min first time; ~2 s when only our .py files change
docker compose up -d
sleep 25
docker compose ps                                    # mem0-postgres healthy, mem0-server Up
docker logs mem0-server --tail 30
curl -s -o /dev/null -w 'API /docs: HTTP %{http_code}\n' http://127.0.0.1:8888/docs
```

Healthy log: 6 alembic migrations, `GET http://127.0.0.1:11434/api/tags 200`, the `mem0_memories` collection and its `user_id`/`agent_id`/`run_id`/`actor_id` indexes created, `Uvicorn running on http://127.0.0.1:8888`. A `passlib` `DeprecationWarning` about `crypt` is harmless. Interactive API docs: `http://127.0.0.1:8888/docs`.

After changing `.env` only: `docker compose up -d --force-recreate mem0` (no rebuild). After changing `local_config.py`/`patch_main.py`: `docker compose build mem0 && docker compose up -d --force-recreate mem0`.

### 10.6 Test the API

Inside `ai-agents/mem0-server` (reads the key from `.env`, never printed):

```
KEY=$(grep '^ADMIN_API_KEY=' .env | cut -d= -f2-)
curl -s -o /dev/null -w 'sem chave: HTTP %{http_code}\n' -X POST http://127.0.0.1:8888/search -H 'Content-Type: application/json' -d '{"query":"teste","filters":{"user_id":"rizzo"}}'
curl -s -X POST http://127.0.0.1:8888/memories -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Frase de teste com um fato durável."}],"user_id":"rizzo","agent_id":"teste-manual"}' | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8888/search -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"query":"fato de teste","filters":{"user_id":"rizzo"}}' | python3 -m json.tool
curl -s -X DELETE "http://127.0.0.1:8888/memories?user_id=rizzo&agent_id=teste-manual" -H "X-API-Key: $KEY"; echo
unset KEY
```

Expected: `401` without the key; the add returns `"event": "ADD"` with an extracted English fact (~20 s the first time, including loading `gpt-oss`); search returns it with a score; the last line deletes every test memory at once. API notes: in `/search`, IDs go **inside `filters`** (top-level `user_id` is deprecated); the score is a fused score (semantic + BM25 + entities), higher is better, default threshold 0.1. Check entity linking with `curl -s -X POST http://127.0.0.1:6333/collections/mem0_memories_entities/points/count -H 'Content-Type: application/json' -d '{"exact":true}'` (count > 0).

### 10.7 Connect Hermes (memory provider)

The Hermes `mem0` plugin ships installed. **Configure it by files, not by CLI flags:** the documented `hermes memory setup mem0 --mode selfhosted --host ... --api-key ...` does not exist in the installed version (docs are ahead of the release), and on the argument error the CLI **echoes the values passed — including the key**. That happened once here; the key was rotated right away. Never pass secrets as command-line arguments.

Terminal, no venv, inside `ai-agents/mem0-server`:

```
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak-pre-mem0
KEY=$(grep '^ADMIN_API_KEY=' .env | cut -d= -f2-)
echo "MEM0_API_KEY=$KEY" >> ~/.hermes/.env
unset KEY
cat > ~/.hermes/mem0.json << 'EOF'
{
  "host": "http://127.0.0.1:8888",
  "user_id": "rizzo",
  "agent_id": "hermes"
}
EOF
hermes config set memory.provider mem0
hermes memory status          # Provider: mem0, Plugin installed ✓, Status available ✓
```

Keys the plugin reads from `mem0.json` (checked in `plugins/memory/mem0/__init__.py`): `mode`, `api_key`, `host`, `user_id` (default `hermes-user` — must be changed to `rizzo` to share with the Crew), `agent_id` (default `hermes`), `rerank`, `sync_max_chars` (default 450). There is **no** switch for the automatic turn sync — hence 10.8.

To rotate the key: generate a new one, replace `ADMIN_API_KEY` in `ai-agents/mem0-server/.env`, `docker compose up -d --force-recreate mem0`, replace `MEM0_API_KEY` in `~/.hermes/.env`, then confirm with a search using the Hermes copy of the key (`HTTP 200`).

Validated: Hermes found, via `mem0_search`, a memory written by a different `agent_id` — the shared-memory goal.

### 10.8 Auto-sync block + memory rules for Hermes

**Problem found:** after every reply, the plugin's `sync_turn` sends the user message + answer to the server with `infer=true` (LLM fact extraction), in the background. With ADD-only Mem0 this created a redundant memory on the very first test (the question itself became a fact), and it costs one extra `gpt-oss` call per turn, competing for the GPU. Editing the plugin was rejected (overwritten by `hermes update`).

**Decision (2026-09-29):**

1. **Server-side block** — `MEM0_SKIP_INFER_AGENTS=hermes` (replacement #4 in 10.4). Automatic syncs from Hermes are dropped and logged as `skip_infer`. Explicit saves (`mem0_add`, which the plugin sends with `infer=false`, stored verbatim in ~0.1 s) still work. Other agents are unaffected. To undo: remove `hermes` from that line and recreate the container.
   Verified with 3 cases: `hermes`+`infer=true` → skipped; `hermes`+`infer=false` → stored; `teste-manual`+`infer=true` → stored. `docker logs mem0-server 2>&1 | grep skip_infer` shows the drops.
2. **Hermes decides what to save** — reading memory is still automatic (the plugin prefetches relevant memories before each turn). Rules appended to `~/.hermes/SOUL.md` (backup: `SOUL.md.bak-pre-mem0`; outside the repo, so the full text is kept here):

```
## Long-term memory (Mem0)
- You have a long-term memory (tools: mem0_search, mem0_add) shared with the user's CrewAI agents.
- Automatic turn syncing is disabled on purpose: nothing is saved unless you call mem0_add.
- Call mem0_add proactively, without being asked, whenever the user states something durable. Examples (not an exhaustive list): decisions, preferences, constraints, goals, and facts about their projects, people, setup or plans.
- Also call mem0_add whenever the user asks you to remember, note, save or record something, in any wording (e.g. "anota", "lembra", "registre", "guarda").
- Write each memory as one short, self-contained sentence in English, starting with the project name when relevant (e.g. "Get Contractors Now: second niche will be plumbing in Cochrane.").
- Do NOT save small talk, questions, your own answers or explanations, temporary task state, or facts already in memory (use mem0_search first if unsure).
- A statement of a decision or fact is information, not a work request: after saving it, reply with a short acknowledgment. Do not search files, run commands or start tasks unless the user asks for them.
- Whenever you call mem0_add, end your reply with one line in this exact format: "Saved to memory: <the sentence you saved>".
- Before answering questions about the user's projects, preferences or past decisions, search memory first.
```

Append it with `cat >> ~/.hermes/SOUL.md << 'EOF' ... EOF` and check with `grep -c 'Long-term memory (Mem0)' ~/.hermes/SOUL.md` (must print `1`).

**Canary test** — in `hermes`, state a real decision **without** asking to save it. Pass = one `mem0_add`, no file searches, and the reply ends with `Saved to memory: ...`. History: the first version of the rules saved correctly but didn't announce it and treated the statement as a work request (searched files, then asked what to do); the version above passed. Proactive saving depends on the model's judgement — "anota isso" is the guaranteed path.

### 10.9 Validation — weekly report + timer

**`mem0_report.py`** (standard library only — system `python3`, no venv). Run inside `ai-agents/mem0-server`:

```
python3 mem0_report.py                 # report for the last 7 days → reports/YYYY-MM-DD.md (also printed)
python3 mem0_report.py --days 14       # other window
python3 mem0_report.py --delete <id>   # delete one memory by id
```

Report sections: 1) totals per `agent_id` and new ones in the window; 2) every new memory with date, agent and id — the human review list; 3) **candidate** duplicates (nearest neighbours in Qdrant with cosine ≥ `DUP_THRESHOLD`); 4) number of `skip_infer` drops (counts only since the last container recreation — `docker logs` resets); 5) the weekly checklist.

**Threshold calibration (2026-09-29):** a truly redundant pair scored **0.861**, while two *different* memories on the same subject scored **0.843** — `nomic-embed-text` groups by topic, not by redundancy. So similarity alone can't decide: `DUP_THRESHOLD = 0.80` lists candidates, and **the script never deletes anything by itself**.

**Timer** — systemd *user* units, versioned in `ai-agents/mem0-server/systemd/`. Runs even when logged out (linger is enabled — `hermes doctor` confirms it) and catches up after downtime (`Persistent=true`). No `sudo`:

```
cd ~/Projects/AI/ai-agents/mem0-server
mkdir -p ~/.config/systemd/user
cp systemd/mem0-report.service systemd/mem0-report.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now mem0-report.timer
systemctl --user list-timers mem0-report.timer --no-pager    # next run: Monday 08:45
systemctl --user start mem0-report.service                   # run once now as a test
systemctl --user show mem0-report.service -p Result          # Result=success
```

`mem0-report.service`: `Type=oneshot`, `WorkingDirectory=%h/Projects/AI/ai-agents/mem0-server`, `ExecStart=/usr/bin/python3 %h/Projects/AI/ai-agents/mem0-server/mem0_report.py`. `mem0-report.timer`: `OnCalendar=Mon *-*-* 08:45:00`, `Persistent=true`, `WantedBy=timers.target`. To change the schedule, edit `OnCalendar` in the repo copy, copy it again and `daemon-reload`.

**Weekly routine (~5 min, Mondays after 08:45):**

1. Open the latest report: `ls -t ~/Projects/AI/ai-agents/mem0-server/reports | head -1`.
2. Section 2 — is each new memory true, and still true a month from now? If not: `python3 mem0_report.py --delete <id>` (inside `mem0-server`).
3. Section 3 — for each candidate pair: redundancy → delete the older/weaker one; just the same subject → keep both.
4. Canary — once a week, tell Hermes a real decision without asking to save it; check the `Saved to memory:` line in the chat and the memory in the next report.
5. False negatives — anything important said to Hermes that is missing from section 2? Note it on the report's checklist line.

Monthly, across reports: many false negatives → Hermes saves too little; reconsider automatic sync with expiration + scheduled dedup. Section 2 full of noise → tighten the `SOUL.md` rules. Section 4 high while section 2 is nearly empty → Hermes isn't saving on its own; review `SOUL.md`.

### 10.10 Updating

- **Hermes:** `hermes update` is safe for this setup — everything we changed lives in `~/.hermes` (`config.yaml`, `.env`, `mem0.json`, `SOUL.md`), which updates preserve; no Hermes code was modified. After every update: `hermes doctor`, `hermes memory status`, and the canary test (10.8), since the plugin's behavior could change.
- **Mem0 server:** never updates by itself. To upgrade: clone the new tag into `mem0-server-src` (delete the old clone first), change `mem0ai==<version>` and the image tag, test `patch_main.py` against the new `main.py` (10.4), then build and recreate. If the build stops at `patch_main: esperado 1 ocorrencia, achei 0`, upstream changed the text — adapt the patch before going on.
- **Qdrant / Postgres:** a running container never updates itself; only after pulling a new image and recreating. Postgres is pinned to major 17.

### 10.11 Troubleshooting / learnings

- **`hermes: error: unrecognized arguments: --mode ...`** — see 10.7; if a secret was in the command, rotate it.
- **Model reload delays:** Ollama unloads idle models after 5 min; the first add/turn after a pause pays the load (~20 s).
- **Similarity scores in search results look low (~0.37):** they are fused scores; what matters is the ranking between memories, not the absolute value.
- **Embedded Qdrant "already in use"/lock errors:** only affect the legacy Section 5 setup (one process at a time); the Qdrant server has no such limit.
- **Neo4j empty despite "graph memory":** expected on Mem0 2.x (Section 5 note).

---

## Pending / To Investigate

**Next stage — shared memory and the Crew**

- [ ] **Migrate the CrewAI Mem0 tool to the Mem0 server REST API** (Section 7 note) — `user_id: rizzo`, a Crew-specific `agent_id`, key from the Mem0 server `.env`.
- [ ] **Hermes Agent ↔ CrewAI integration** — Hermes orchestrating/delegating to the Crew; first Crew is for the "Get Contractors Now" project (decided 2026-09-28). Idea: a `run_crew.py` in the CrewAI venv that Hermes calls through its terminal tool; evolve to MCP later.
- [ ] **`Process.sequential` in CrewAI** — apply when that Crew script is written (Section 9.9).
- [ ] **Retire the legacy Mem0 library setup** once the Crew uses the API — delete `ai-agents/mem0/qdrant_data/` (3 test memories, checked 2026-09-28), the `venv-mem0` venv, `mem0/config.py`/`test_mem0.py`, and `mem0ai`/spaCy/fastembed from the CrewAI venv. Supersedes the old items "align `mem0ai` versions" and "hardcoded paths in `mem0/config.py`".
- [ ] **Update Hermes Agent** ("update available" shown at startup) — `hermes doctor` + `hermes memory status` + canary before and after (Section 10.10).
- [ ] **First weekly memory review — Monday 2026-10-05** (Section 10.9 routine), then monthly assessment.
- [ ] **Extraction `max_tokens`** — after the server has been running for a while, check `docker logs mem0-server` for cut-off JSON errors; if they appear, raise `MEM0_LLM_MAX_TOKENS` to 8192 (env change + recreate).
- [ ] **Graphiti (Zep)** — temporal knowledge-graph memory that runs on Neo4j (possibly with an MCP server): study whether it can bring back a queryable graph. `neo4j-mem0` stays stopped until then; afterwards decide to keep it (recreate with Section 5.2's command) or remove it.
- [ ] **MCP in the AI project** — check whether Hermes and CrewAI work as MCP clients; if so, test a Mem0 MCP server in HTTP mode (always-on service, many agents), not stdio.

**Hermes Agent**

- [ ] **Test `/reasoning high`** with `gpt-oss:20b` — confirm it reaches the model through Ollama `/v1`.
- [ ] **Test other orchestrator models** (starting with the installed ones; DeepSeek was mentioned, but `deepseek-r1:14b` is weak at tool calling), then look for models known to work well with Hermes.
- [ ] **Hermes-4-14B at 64K via YaRN** (llama.cpp directly, `q8_0` KV cache) vs. `gpt-oss:20b-64k`; follow issue #53347 / PR #32770 (`allow_short_context`) and the model's `config.json` (`max_position_embeddings`, `rope_scaling`). The GGUF stays installed for this.
- [ ] **Multiple Hermes profiles** — different agents/models per profile, each with its own Telegram bot (one gateway per profile), confronting decisions between agents; CLI state shared (default). After the current stages.
- [ ] **Hermes remote control via chat platform** — Telegram chosen; `hermes setup gateway`.
- [ ] **SearXNG instead of DuckDuckGo** for Hermes search (another Docker container).
- [ ] **Vision capability auto-detection** in the wizard (Section 9.6) — only worth reporting upstream if it persists after `hermes update`.
- [ ] **agent-browser npm vulnerability** — upstream (#116774); fixed by a future `hermes update`.

**Infrastructure**

- [ ] **Backup & restore procedure** — write and test it for everything in Section 0.2 (now including Qdrant storage, Mem0 server Postgres/history and `.env`, and `~/.hermes`) before the next Ubuntu wipe. Highest priority for a reinstall guide.
- [ ] **Lock files per venv** — `pip freeze > requirements.lock.txt` in each venv, committed (`crewai` is unpinned).
- [ ] **Paperclip LAN access** — likely `PAPERCLIP_ALLOWED_HOSTNAMES=<LOCAL_IP>` in `.env`, then recreate (8.3 + 8.4). Untested.
- [ ] **Remote access (Section 3.5)** — VPN (e.g. Tailscale); not yet configured.
- [ ] **OpenUI (Weights & Biases) via Docker** — planned, not installed (checked 2026-09-28).
- [ ] **"Coding" agent (remote terminal assistant)** — e.g. Letta Code / App Server. Not yet evaluated.

**Done in this stage (2026-09-28 → 2026-09-29)**

- [x] **`OLLAMA_MAX_LOADED_MODELS`** re-confirmed at `1`, then raised to `2` (Section 2.3).
- [x] **Legacy `custom_providers` removed** from `~/.hermes/config.yaml` (Section 9.7).
- [x] **Qdrant as a standalone server** (Section 10.2).
- [x] **Mem0 Docker server** — patched for Ollama + Qdrant (Section 10).
- [x] **`MEM0_TELEMETRY=false`** — applied on the Mem0 server.
- [x] **Hermes memory provider** — Mem0 server, shared `user_id`, auto-sync block, `SOUL.md` rules, weekly report + timer (Sections 10.7–10.9).
- [x] **Neo4j** — confirmed unused by Mem0 2.x; stopped, port 7687 closed (Section 5.2).
- [x] **Hermes-4-14B GGUF** — decision: keep for the YaRN test.

**Done earlier**

- [x] Mem0 optional extras (spaCy + fastembed in the CrewAI venv); `chromadb`/`posthog` conflict (pinned `posthog<6.0.0`); Paperclip installed; Hermes Agent reinstalled and validated; Hermes vision config root cause; Section 9 cleanup.

## Notes

- This guide assumes a fresh Ubuntu install on the 500GB partition of the Samsung 990 PRO 2TB.
- Update model list in Section 2.2 as new models are added/removed.
- Update Section 3 if additional Docker containers are introduced later (Qdrant and the Mem0 server were — Section 10).
- Ollama is intentionally kept native, not dockerized — see the note at the top of Section 3.
- Section 3.5 (remote access) is a placeholder until that setup is actually done.
- Section 5 (Mem0 library) is legacy since 2026-09-29; the shared memory is the Mem0 server (Section 10).
- Section 7 (CrewAI) runs in its own venv, separate from Mem0's (Section 5) — while it still uses the embedded Qdrant, the two must never have that data open at the same time (see the note in 5.5).
- Sections 8 (Paperclip), 9 (Hermes Agent) and 10 (`mem0-server-src`) include nested git repos inside `~/Projects/AI` — see the `.gitignore` note in Section 6 before running any `git` commands at the repo root.
- Hermes Agent's config/secrets live entirely outside the repo at `~/.hermes`; the memory-related pieces (`mem0.json`, `SOUL.md` rules) are written out in Section 10 so they can be recreated.
- `~/Projects/AI/modelfiles/` (Section 2.4) **is** tracked by git — it holds the Modelfiles for the `-64k` context variants.
- The `OLLAMA_MAX_LOADED_MODELS=2` systemd override (Section 2.3) is a global Ollama setting — it affects every model call from every section of this guide. Ollama still refuses to load a second model that doesn't fit, so two big models never share the GPU.
- Every running Docker container in this guide uses the `unless-stopped` restart policy (standardized on 2026-09-27 — see 3.2). Exception: `neo4j-mem0`, intentionally stopped with `no` (Section 5.2).
- Every service added in Section 10 binds to `127.0.0.1` only; nothing new is exposed to the LAN.
