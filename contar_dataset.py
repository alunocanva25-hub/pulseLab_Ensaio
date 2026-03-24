from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

META_FILE = Path("dataset_led/metadata.jsonl")

if not META_FILE.exists():
    print("metadata.jsonl não encontrado.")
    raise SystemExit(1)

labels = Counter()
sources = Counter()
sessions = Counter()
by_label_session = defaultdict(int)

total = 0

with open(META_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        row = json.loads(line)
        total += 1

        label = row.get("label", "desconhecido")
        source = row.get("source", "sem_fonte")
        session = row.get("session_name", "sem_sessao")

        labels[label] += 1
        sources[source] += 1
        sessions[session] += 1
        by_label_session[(label, session)] += 1

print("=" * 60)
print(f"TOTAL DE AMOSTRAS: {total}")
print("=" * 60)

print("\nPor classe:")
for label, count in labels.most_common():
    print(f"  {label}: {count}")

print("\nPor fonte:")
for source, count in sources.most_common():
    print(f"  {source}: {count}")

print("\nPor sessão:")
for session, count in sessions.most_common():
    print(f"  {session}: {count}")

print("\nClasse x sessão:")
for (label, session), count in sorted(by_label_session.items()):
    print(f"  {label} | {session}: {count}")