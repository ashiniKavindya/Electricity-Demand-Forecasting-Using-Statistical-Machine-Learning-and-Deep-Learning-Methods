#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
while ! pg_isready -h postgres -U "${POSTGRES_USER:-aemo_user}" 2>/dev/null; do
    sleep 1
done
echo "PostgreSQL is ready!"

# Execute the command passed to docker run
exec "$@"
