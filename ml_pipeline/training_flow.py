import subprocess
from prefect import flow, task


@task(retries=2, retry_delay_seconds=30)
def run_training():
    result = subprocess.run(
        ["python", "/app/ml_pipeline/training.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


@flow(log_prints=True)
def dxy_training():
    log = run_training()
    for line in log.strip().split("\n"):
        print(f"  {line}")


if __name__ == "__main__":
    dxy_training.serve(name="dxy-training-manual")
