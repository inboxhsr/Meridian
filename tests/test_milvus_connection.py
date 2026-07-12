"""
test_milvus_connection.py — Sprint 0 test gate

Verifies Milvus Standalone is running and reachable on localhost:19530.
If this test fails, check that `docker compose up milvus -d` has completed.
"""
import os
import pytest
from pymilvus import MilvusClient


def test_milvus_connection():
    """Connect to Milvus and confirm the server is alive."""
    host = os.environ.get("MILVUS_HOST", "localhost")
    port = os.environ.get("MILVUS_PORT", "19530")
    uri = f"http://{host}:{port}"

    try:
        client = MilvusClient(uri=uri)
        # List collections — succeeds only if Milvus is fully up
        collections = client.list_collections()
        # collections may be an empty list on a fresh Milvus — that's fine
        assert isinstance(collections, list), (
            f"Unexpected response from Milvus: {collections}"
        )
    except Exception as exc:
        pytest.fail(
            f"Could not connect to Milvus at {uri}.\n"
            f"  Error: {exc}\n"
            f"  → Make sure Docker is running and Milvus is healthy:\n"
            f"    docker compose up milvus -d\n"
            f"    docker compose ps"
        )
