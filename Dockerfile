# Dockerfile
# Extends the judges' standard image with our C++ extension.
# Build: docker build -t macro-placer .
# The judges run: docker build . && evaluate submissions/sa_placer.py --all

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Build tools for C++ extension
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pybind11

WORKDIR /workspace
COPY . .

# Build the C++ extension — happens at image-build time so no network
# access is needed at evaluation time (judges run with --network none).
RUN cd placement_ops && python setup.py build_ext --inplace

# Make the compiled .so importable from the submissions directory.
# The glob handles the cpython version suffix (e.g. .cpython-311-x86_64.so).
RUN cp placement_ops/placement_ops*.so submissions/ 2>/dev/null || \
    cp placement_ops/placement_ops*.so . 2>/dev/null || true

# Verify the build succeeded
RUN python -c "import placement_ops; print('placement_ops OK')"do