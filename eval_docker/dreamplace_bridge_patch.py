"""
Monkey-patch for UT Austin AS's dreamplace_bridge.py:
Replace Docker subprocess call with direct DREAMPlace invocation.
"""
import subprocess
import time

def _run_dreamplace_direct(workdir, config_path, timeout):
    """Run DREAMPlace directly (not via Docker)."""
    cmd = [
        "python3", "/opt/DREAMPlace/build/dreamplace/Placer.py",
        str(config_path),
    ]

    print(f"  Running DREAMPlace: {' '.join(cmd[-2:])}")
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=str(workdir),
    )
    elapsed = time.time() - t0
    print(f"  DREAMPlace finished in {elapsed:.1f}s")

    if result.returncode != 0:
        stderr_lines = result.stderr.strip().split("\n")[-20:]
        stdout_lines = result.stdout.strip().split("\n")[-20:]
        print(f"  STDOUT (last 20 lines):\n" + "\n".join(stdout_lines))
        print(f"  STDERR (last 20 lines):\n" + "\n".join(stderr_lines))
        raise RuntimeError(
            f"DREAMPlace exited with code {result.returncode}"
        )

    return result

# Apply the patch when imported
import dreamplace_bridge
dreamplace_bridge._run_dreamplace_docker = _run_dreamplace_direct
print("[patch] Replaced Docker DREAMPlace with direct invocation")
