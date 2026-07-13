import os

from backend.core import config


def _clear_redis_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("REDIS") or key.startswith("HEROKU_REDIS"):
            monkeypatch.delenv(key, raising=False)


def test_resolve_redis_prefers_explicit_tls(monkeypatch):
    _clear_redis_env(monkeypatch)
    monkeypatch.setenv("REDIS_TLS_URL", "rediss://tls")
    monkeypatch.setenv("REDIS_URL", "redis://plain")
    monkeypatch.setenv("HEROKU_REDIS_CRIMSON_URL", "rediss://crimson")
    assert config._resolve_redis_url() == "rediss://tls"


def test_resolve_redis_explicit_url_beats_heroku_colour(monkeypatch):
    _clear_redis_env(monkeypatch)
    monkeypatch.setenv("REDIS_URL", "rediss://primary")
    monkeypatch.setenv("HEROKU_REDIS_CRIMSON_URL", "rediss://crimson")
    assert config._resolve_redis_url() == "rediss://primary"


def test_resolve_redis_autoscans_any_heroku_colour(monkeypatch):
    # The fix: a recreated/differently-coloured add-on is found even when neither
    # REDIS_URL nor REDIS_TLS_URL is set (no dependence on a hardcoded colour).
    _clear_redis_env(monkeypatch)
    monkeypatch.setenv("HEROKU_REDIS_CRIMSON_URL", "rediss://crimson")
    assert config._resolve_redis_url() == "rediss://crimson"


def test_resolve_redis_prefers_tls_colour_variant(monkeypatch):
    _clear_redis_env(monkeypatch)
    monkeypatch.setenv("HEROKU_REDIS_CRIMSON_URL", "redis://crimson-plain")
    monkeypatch.setenv("HEROKU_REDIS_CRIMSON_TLS_URL", "rediss://crimson-tls")
    assert config._resolve_redis_url() == "rediss://crimson-tls"


def test_resolve_redis_cloud_provider_then_localhost(monkeypatch):
    _clear_redis_env(monkeypatch)
    assert config._resolve_redis_url() == "redis://localhost:6379/0"
    monkeypatch.setenv("REDISCLOUD_URL", "redis://cloud")
    assert config._resolve_redis_url() == "redis://cloud"


def test_resolve_redis_ignores_empty_values(monkeypatch):
    _clear_redis_env(monkeypatch)
    monkeypatch.setenv("REDIS_URL", "   ")  # set-but-blank must not win
    monkeypatch.setenv("HEROKU_REDIS_CRIMSON_URL", "rediss://crimson")
    assert config._resolve_redis_url() == "rediss://crimson"
