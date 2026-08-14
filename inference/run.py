"""Inference entrypoint. Run with: python -m inference.run

Standalone process by design, same as collector/run.py: collector -> database ->
everything else reads from the database rather than depending on each other directly,
so inference stays up regardless of the collector's or API's own uptime.
"""
from inference.scheduler import run_forever

if __name__ == "__main__":
    run_forever()
