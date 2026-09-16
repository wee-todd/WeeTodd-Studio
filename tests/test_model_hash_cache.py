import os

import pytest


def test_warm_hash_reuses_only_unchanged_file_identity(tmp_path):
    from wee_todd_mlx.model_hash_cache import cached_model_hash

    source = tmp_path / 'weight'
    source.write_bytes(b'original')
    calls = []

    def compute(filename):
        import hashlib
        calls.append(filename)
        return hashlib.sha256(filename.read_bytes()).hexdigest()

    database = tmp_path / 'hashes.sqlite3'
    first = cached_model_hash(source, compute, database=database)
    assert cached_model_hash(source, compute, database=database) == first
    assert len(calls) == 1
    before = source.stat()
    source.write_bytes(b'modified')  # Same size, even with restored mtime, must invalidate.
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert cached_model_hash(source, compute, database=database) != first
    assert len(calls) == 2


def test_corrupt_database_falls_back_to_complete_hash(tmp_path):
    from wee_todd_mlx.model_hash_cache import cached_model_hash

    source = tmp_path / 'weight'
    source.write_bytes(b'original')
    database = tmp_path / 'hashes.sqlite3'
    database.write_bytes(b'not a database')
    assert cached_model_hash(source, lambda _: 'a' * 64, database=database) == 'a' * 64
    assert database.read_bytes() == b'not a database'


def test_replaced_inode_and_explicit_disable_force_reverification(tmp_path, monkeypatch):
    from wee_todd_mlx.model_hash_cache import cached_model_hash

    source = tmp_path / 'weight'
    source.write_bytes(b'original')
    database = tmp_path / 'hashes.sqlite3'
    calls = []

    def compute(filename):
        calls.append(filename)
        return 'a' * 64

    cached_model_hash(source, compute, database=database)
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(source.read_bytes())
    original = source.stat()
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(source)
    cached_model_hash(source, compute, database=database)
    assert len(calls) == 2
    monkeypatch.setenv('WEETODD_DISABLE_MODEL_HASH_CACHE', '1')
    cached_model_hash(source, compute, database=database)
    assert len(calls) == 3


def test_file_mutated_while_hashing_is_not_cached(tmp_path):
    from wee_todd_mlx.model_hash_cache import cached_model_hash

    source = tmp_path / 'weight'
    source.write_bytes(b'original')

    def mutate(filename):
        filename.write_bytes(b'modified')
        return 'a' * 64

    with pytest.raises(ValueError, match='changed'):
        cached_model_hash(source, mutate, database=tmp_path / 'hashes.sqlite3')


def test_cache_entry_limit_is_bounded(tmp_path):
    import sqlite3

    from wee_todd_mlx.model_hash_cache import cached_model_hash

    database = tmp_path / 'hashes.sqlite3'
    for index in range(5):
        source = tmp_path / f'weight-{index}'
        source.write_bytes(b'weight')
        cached_model_hash(source, lambda _: 'a' * 64, database=database, max_entries=3)
    with sqlite3.connect(database) as db:
        assert db.execute('SELECT COUNT(*) FROM model_hashes').fetchone()[0] == 3
