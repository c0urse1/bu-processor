#!/usr/bin/env python3
"""
Migrate .env keys to BU_-prefixed schema used by Pydantic Settings.

Actions:
- Read .env (or alternate file passed as arg)
- Map known legacy keys to new BU_ keys
- Preserve unknown keys as-is
- Write .env.migrated alongside original
- Optionally (default) back up original to .env.backup and replace .env

Usage (Windows cmd):
  python scripts\maintenance\migrate_env_keys.py
  python scripts\maintenance\migrate_env_keys.py --source .env.production --no-replace
"""

from pathlib import Path
import argparse
import re


LEGACY_TO_BU = {
    # OpenAI
    "OPENAI_API_KEY": "BU_OPENAI__OPENAI_API_KEY",
    "OPENAI_MODEL": "BU_OPENAI__OPENAI_MODEL",
    "OPENAI_MAX_TOKENS": "BU_OPENAI__MAX_TOKENS",
    "CHATBOT_ENABLED": "BU_OPENAI__ENABLE_CHATBOT",
    # Pinecone / Vector DB
    "PINECONE_API_KEY": "BU_VECTOR_DB__PINECONE_API_KEY",
    "PINECONE_ENVIRONMENT": "BU_VECTOR_DB__PINECONE_ENVIRONMENT",
    "PINECONE_INDEX_NAME": "BU_VECTOR_DB__PINECONE_INDEX_NAME",
    "PINECONE_DIMENSION": "BU_VECTOR_DB__EMBEDDING_DIMENSION",
    "VECTOR_DB_ENABLED": "BU_VECTOR_DB__ENABLE_VECTOR_DB",
    # Environment / Logging / Cache
    "ENVIRONMENT": "BU_ENVIRONMENT",
    "LOG_LEVEL": "BU_LOG_LEVEL",
    "CACHE_TTL": "BU_SEMANTIC__CACHING__CACHE_TTL_SECONDS",
    # API
    "API_HOST": "BU_API__HOST",
    "API_PORT": "BU_API__PORT",
    "SECRET_KEY": "BU_API__SECRET_KEY",
    # Feature flags
    "SEMANTIC_ENABLED": "BU_SEMANTIC__ENABLE_SEMANTIC_CLUSTERING",
    "DEDUPLICATION_ENABLED": "BU_DEDUPLICATION__ENABLE_SEMANTIC_DEDUPLICATION",
    # Other passthroughs
    "DEBUG": "DEBUG",
    "TESTING": "TESTING",
    "BU_LAZY_MODELS": "BU_LAZY_MODELS",
    # Embeddings backend selection
    "EMBEDDINGS_BACKEND": "EMBEDDINGS_BACKEND",
    # OpenAI embedding model name passthrough
    "OPENAI_EMBED_MODEL": "OPENAI_EMBED_MODEL",
}


def parse_env_lines(text: str):
    """Parse simple KEY=VALUE lines; preserve comments/blank lines as is."""
    lines = text.splitlines()
    parsed = []  # list of tuples (type, content) where type in {"kv","raw"}
    kv_pattern = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)\s*$")
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            parsed.append(("raw", line))
            continue
        m = kv_pattern.match(line)
        if m:
            key, value = m.group(1), m.group(2)
            parsed.append(("kv", (key, value)))
        else:
            # Unrecognized, keep raw
            parsed.append(("raw", line))
    return parsed


def migrate_content(text: str):
    parsed = parse_env_lines(text)
    out_lines = []
    seen_new_keys = set()

    for typ, content in parsed:
        if typ == "raw":
            out_lines.append(content)
            continue
        key, value = content
        new_key = LEGACY_TO_BU.get(key, key)

        # Special handling: ensure both BU_ENVIRONMENT and BU_PROCESSOR_ENVIRONMENT are present
        if key in ("ENVIRONMENT", "BU_ENVIRONMENT", "BU_PROCESSOR_ENVIRONMENT"):
            for k in ("BU_ENVIRONMENT", "BU_PROCESSOR_ENVIRONMENT"):
                if k not in seen_new_keys:
                    seen_new_keys.add(k)
                    out_lines.append(f"{k}={value}")
            continue

        # Avoid duplicating if both legacy and BU_ present; prefer BU_
        if new_key in seen_new_keys:
            continue
        seen_new_keys.add(new_key)
        out_lines.append(f"{new_key}={value}")
    return "\n".join(out_lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Migrate .env keys to BU_ schema")
    ap.add_argument("--source", default=".env", help="Source env file (default: .env)")
    ap.add_argument("--no-replace", action="store_true", help="Do not replace .env, only write .env.migrated")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"Source env not found: {src}")
        return 1

    text = src.read_text(encoding="utf-8")
    migrated = migrate_content(text)

    migrated_path = src.with_suffix(src.suffix + ".migrated") if src.name != ".env" else src.parent / ".env.migrated"
    migrated_path.write_text(migrated, encoding="utf-8")
    print(f"Wrote migrated env: {migrated_path}")

    # If working on .env and replacing is allowed, back up and replace
    if src.name == ".env" and not args.no_replace:
        backup = src.parent / ".env.backup"
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
            print(f"Backed up original to: {backup}")
        # Replace .env
        src.write_text(migrated, encoding="utf-8")
        print("Replaced .env with migrated content.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
