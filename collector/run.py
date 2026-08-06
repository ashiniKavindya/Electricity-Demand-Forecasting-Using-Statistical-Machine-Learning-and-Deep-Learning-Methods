"""Collector entrypoint. Run with: python -m collector.run

Standalone process by design (not embedded in the FastAPI backend) so its
uptime is independent of the API - matches the architecture diagram
discussed in planning: collector -> database -> everything else reads
from the database rather than from each other directly.
"""
from collector.scheduler import run_forever

if __name__ == "__main__":
    run_forever()
