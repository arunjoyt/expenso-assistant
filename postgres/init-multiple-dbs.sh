#!/bin/bash
# Create every database listed in POSTGRES_MULTIPLE_DATABASES (comma-separated).
# The checkpointer uses `assistant`; Langfuse uses `langfuse`.
set -euo pipefail

if [ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
	for db in $(echo "$POSTGRES_MULTIPLE_DATABASES" | tr ',' ' '); do
		echo "Creating database '$db'"
		psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
			SELECT 'CREATE DATABASE $db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\gexec
		EOSQL
	done
fi
