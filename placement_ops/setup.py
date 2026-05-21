# placement_ops/setup.py
# Run: python setup.py build_ext --inplace
# Produces: placement_ops.cpython-*.so  (or .pyd on Windows)

from setuptools import setup, Extension
import pybind11
import sys
import os

# Compiler flags
extra_compile_args = ["-std=c++17", "-O3", "-ffast-math"]

# Add native CPU optimisations when not cross-compiling for a Docker image.
# On the submission machine (AMD EPYC 9655P) this enables AVX-512;
# on your i7 it enables whatever your chip supports.
# Safe to leave on — if the machine doesn't support an instruction the
# binary will segfault on load (not silently corrupt), which you'd catch
# during local testing.
if sys.platform != "win32":
    extra_compile_args.append("-march=native")

ext = Extension(
    name="placement_ops",
    sources=[os.path.join("src", "placement_ops.cpp")],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=extra_compile_args,
)

setup(
    name="placement_ops",
    version="1.0",
    ext_modules=[ext],
    zip_safe=False,
)