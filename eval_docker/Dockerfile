FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps. build-essential is needed at runtime for torch.compile
# (Inductor backend shells out to gcc to build cached kernels).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /challenge

# Copy the challenge evaluation infrastructure
COPY macro_place/ /challenge/macro_place/
COPY pyproject.toml /challenge/pyproject.toml
COPY requirements.txt /challenge/requirements.txt

# Copy benchmarks (read-only at runtime, but needed in image)
COPY external/MacroPlacement/Testcases/ICCAD04/ /challenge/external/MacroPlacement/Testcases/ICCAD04/

# Copy PlacementCost evaluator (plc_client_os from TILOS submodule)
COPY external/MacroPlacement/CodeElements/Plc_client/ /challenge/external/MacroPlacement/CodeElements/Plc_client/

# Copy example submissions (for smoke testing)
COPY submissions/examples/ /challenge/submissions/examples/

# Install challenge package and base deps.
# NON-editable install: a plain site-packages copy (no PEP660 meta-path finder),
# so a submission bundling its own self-contained macro_place can shadow it via
# PYTHONPATH=/submission. Self-contained submissions (e.g. with import-time
# audits) require their macro_place.* to resolve inside their own repo.
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir scipy tqdm

# Pre-install common extra deps that submissions may need
# (cheaper to bake in than install at runtime)
RUN pip install --no-cache-dir cvxpy dccp numba

# Common pure-Python deps surfaced by submission triage (air-gapped: must bake in)
RUN pip install --no-cache-dir \
    omegaconf scikit-learn networkx pandas matplotlib optuna einops \
    || true
RUN pip install --no-cache-dir torch_geometric || true

# Entrypoint: run the evaluator on a mounted placer file
# Usage: docker run ... -v /path/to/placer.py:/submission/placer.py evaluate /submission/placer.py --all
ENTRYPOINT ["python", "-m", "macro_place.evaluate"]
