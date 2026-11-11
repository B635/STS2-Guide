# Review coach: the Python engine (MCP server) and the Node agent in one image.
# Build from the repository root:  docker build -f coach/Dockerfile -t sts2-coach .
# Run:  docker run --rm -e DEEPSEEK_API_KEY -v "$PWD/runs:/runs" -v "$PWD/out:/out" sts2-coach review /runs/<file>.run --out /out
FROM node:24-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY review/requirements.txt review/requirements.txt
# The agent looks for the engine's Python at .venv-coach (coach/src/engine.ts).
RUN python3 -m venv .venv-coach && .venv-coach/bin/pip install --no-cache-dir -r review/requirements.txt

COPY coach/package.json coach/package-lock.json coach/
RUN cd coach && npm ci

COPY . .
# The release database is rebuilt from tracked files only (data/), so the image needs no game install.
RUN .venv-coach/bin/python -c "from review.engine import ensure_database; ensure_database()" \
    && cd coach && npm run build

ENTRYPOINT ["node", "/app/coach/dist/cli.js"]
CMD ["review"]
