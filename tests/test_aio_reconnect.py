"""Unit tests for the aio HTTPClient transport-failure reconnect/retry logic.

These tests are self-contained: they do not need a running consul agent. The
aiohttp session is replaced by a fake that raises a scripted sequence of errors,
so we can assert exactly how many attempts are made and when the DNS cache is
flushed.
"""

import asyncio

import aiohttp
import pytest

import consul.aio


class _FakeConnector:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear_dns_cache(self) -> None:
        self.clear_calls += 1


class _FakeResponse:
    status = 200
    headers: dict = {}

    async def text(self, encoding="utf-8") -> str:
        return "ok"


class _FakeSession:
    """Replays ``side_effects`` one per ``request`` call: an ``Exception``
    instance is raised, anything else is returned."""

    def __init__(self, side_effects) -> None:
        self._side_effects = list(side_effects)
        self.connector = _FakeConnector()
        self.request_calls = 0

    async def request(self, *_args, **_kwargs):
        self.request_calls += 1
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect

    async def close(self) -> None:
        pass


async def _make_client(connection_retries, side_effects):
    client = consul.aio.HTTPClient("example", 8500, "http", connection_retries=connection_retries)
    await client._session.close()  # discard the real session built in __init__
    client._session = _FakeSession(side_effects)
    return client


async def _do_request(client):
    return await client._request_with_reconnect("GET", "http://example:8500/v1/kv/x", None, None, {})


class TestRequestReconnect:
    async def test_retries_then_succeeds(self) -> None:
        resp = _FakeResponse()
        client = await _make_client(2, [aiohttp.ClientOSError("boom"), aiohttp.ServerDisconnectedError(), resp])

        result = await _do_request(client)

        assert result is resp
        assert client._session.request_calls == 3
        # DNS cache flushed before each retry, but not after the successful call.
        assert client._session.connector.clear_calls == 2

    async def test_exhausts_retries_and_raises_last_error(self) -> None:
        client = await _make_client(2, [aiohttp.ClientOSError("down")] * 3)

        with pytest.raises(aiohttp.ClientOSError):
            await _do_request(client)

        assert client._session.request_calls == 3  # connection_retries + 1
        # Not flushed after the final (re-raised) attempt.
        assert client._session.connector.clear_calls == 2

    async def test_no_retry_when_disabled(self) -> None:
        client = await _make_client(0, [aiohttp.ClientOSError("down")])

        with pytest.raises(aiohttp.ClientOSError):
            await _do_request(client)

        assert client._session.request_calls == 1
        assert client._session.connector.clear_calls == 0

    async def test_timeout_is_retried(self) -> None:
        resp = _FakeResponse()
        client = await _make_client(1, [asyncio.TimeoutError(), resp])

        result = await _do_request(client)

        assert result is resp
        assert client._session.request_calls == 2
        assert client._session.connector.clear_calls == 1

    async def test_non_transport_error_is_not_retried(self) -> None:
        # ValueError is not in _RECONNECT_ERRORS: it must propagate immediately
        # without any retry or DNS flush (e.g. a real HTTP response error must
        # not be re-issued).
        client = await _make_client(3, [ValueError("nope")])

        with pytest.raises(ValueError, match="nope"):
            await _do_request(client)

        assert client._session.request_calls == 1
        assert client._session.connector.clear_calls == 0


class TestConnectorConfig:
    async def test_kwargs_propagate_to_connector_and_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        real_ctor = aiohttp.TCPConnector

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real_ctor(*args, **kwargs)

        monkeypatch.setattr(aiohttp, "TCPConnector", spy)

        c = consul.aio.Consul("example", 8500, dns_cache_ttl=7, connection_retries=2)
        try:
            assert captured.get("ttl_dns_cache") == 7
            assert c.http._connection_retries == 2
        finally:
            await c.close()

    async def test_defaults_do_not_set_ttl_or_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        real_ctor = aiohttp.TCPConnector

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real_ctor(*args, **kwargs)

        monkeypatch.setattr(aiohttp, "TCPConnector", spy)

        c = consul.aio.Consul("example", 8500)
        try:
            assert "ttl_dns_cache" not in captured  # keep aiohttp's default behaviour
            assert c.http._connection_retries == 0
        finally:
            await c.close()
