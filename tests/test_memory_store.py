"""Unit tests for the SQLite-backed memory store."""

from __future__ import annotations

import pytest

from mcp_picrawler.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=tmp_path / "memory.db")
    yield s
    s.close()


def test_set_and_get_roundtrip(store):
    store.set("user:guy:tea", "strong, milk, no sugar", tags=["preference"], author="claude")
    got = store.get("user:guy:tea")
    assert got is not None
    assert got["key"] == "user:guy:tea"
    assert got["value"] == "strong, milk, no sugar"
    assert got["tags"] == ["preference"]
    assert got["author"] == "claude"


def test_get_missing_returns_none(store):
    assert store.get("nope") is None


def test_set_overwrites_existing(store):
    store.set("k", "first")
    store.set("k", "second", author="nigel")
    got = store.get("k")
    assert got["value"] == "second"
    assert got["author"] == "nigel"


def test_non_string_value_is_json_encoded(store):
    store.set("loc", {"room": "kitchen", "distance": 3.2})
    got = store.get("loc")
    assert '"kitchen"' in got["value"]  # JSON-encoded


def test_search_matches_key_value_tags(store):
    store.set("a:b:c", "ottoman in lounge", tags=["furniture"])
    store.set("a:b:d", "kettle in kitchen", tags=["appliance"])
    store.set("z:z:z", "Pete owns the ottoman",  tags=[])
    hits = [h["key"] for h in store.search("ottoman")]
    assert set(hits) == {"a:b:c", "z:z:z"}


def test_by_tag_exact_match(store):
    store.set("k1", "v1", tags=["preference", "user"])
    store.set("k2", "v2", tags=["prefs"])  # should NOT match "preference"
    store.set("k3", "v3", tags=["preference"])
    hits = {h["key"] for h in store.by_tag("preference")}
    assert hits == {"k1", "k3"}


def test_delete(store):
    store.set("x", "1")
    assert store.delete("x") is True
    assert store.get("x") is None
    assert store.delete("x") is False  # already gone


def test_list_keys_newest_first(store):
    import time
    store.set("older", "a")
    time.sleep(0.01)
    store.set("newer", "b")
    assert store.list_keys() == ["newer", "older"]


def test_empty_key_rejected(store):
    with pytest.raises(ValueError):
        store.set("", "value")


def test_survives_reopen(tmp_path):
    db = tmp_path / "m.db"
    s1 = MemoryStore(db_path=db)
    s1.set("persistent", "data", author="nigel")
    s1.close()
    s2 = MemoryStore(db_path=db)
    got = s2.get("persistent")
    assert got["value"] == "data"
    assert got["author"] == "nigel"
    s2.close()
