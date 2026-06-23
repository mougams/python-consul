import asyncio
import ssl

import aiohttp

from consul import Timeout, base

__all__ = ["Consul"]


class HTTPClient(base.HTTPClient):
    """Asyncio adapter for python consul using aiohttp library"""

    # Errors considered as remote server unavailable
    _RECONNECT_ERRORS = (
        aiohttp.ClientConnectorError,
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientOSError,
        asyncio.TimeoutError,
    )

    def __init__(
        self,
        *args,
        loop=None,
        connections_limit=None,
        connections_timeout=None,
        dns_cache_ttl=None,
        connection_retries=0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loop = loop
        self._connection_retries = connection_retries
        connector_kwargs = {}
        if connections_limit:
            connector_kwargs["limit"] = connections_limit
        if dns_cache_ttl is not None:
            connector_kwargs["ttl_dns_cache"] = dns_cache_ttl
        if self.verify:
            ssl_context = ssl.create_default_context()
            if self.cert:
                if isinstance(self.cert, tuple):
                    ssl_context.load_cert_chain(*self.cert)
                else:
                    ssl_context.load_cert_chain(self.cert)
            if isinstance(self.verify, str):
                ssl_context.load_verify_locations(self.verify)
            connector_kwargs["ssl_context"] = ssl_context
        connector = aiohttp.TCPConnector(loop=self.loop, verify_ssl=bool(self.verify), **connector_kwargs)
        session_kwargs = {}
        if connections_timeout:
            timeout = aiohttp.ClientTimeout(total=connections_timeout)
            session_kwargs["timeout"] = timeout
        self._session = aiohttp.ClientSession(connector=connector, **session_kwargs)  # type: ignore

    async def _request(
        self, callback, method, uri, headers: dict[str, str] | None, data=None, connections_timeout=None
    ):
        session_kwargs = {}
        if connections_timeout:
            timeout = aiohttp.ClientTimeout(total=connections_timeout)
            session_kwargs["timeout"] = timeout
        resp = await self._request_with_reconnect(method, uri, headers, data, session_kwargs)
        body = await resp.text(encoding="utf-8")
        if resp.status == 599:
            raise Timeout
        r = base.Response(resp.status, resp.headers, body)
        return callback(r)

    async def _request_with_reconnect(self, method, uri, headers, data, session_kwargs):
        """Issue the request, retrying transport failures after flushing the DNS
        cache so the next attempt re-resolves the address and can land on
        another healthy instance instead of the dead cached IP."""
        for attempt in range(self._connection_retries + 1):
            try:
                return await self._session.request(  # type: ignore
                    method, uri, headers=headers, data=data, **session_kwargs
                )
            except self._RECONNECT_ERRORS:
                if attempt == self._connection_retries:
                    raise
                self._session.connector.clear_dns_cache()

    def get(self, callback, path, params=None, headers: dict[str, str] | None = None, connections_timeout=None):
        uri = self.uri(path, params)
        return self._request(callback, "GET", uri, headers=headers, connections_timeout=connections_timeout)

    def put(
        self,
        callback,
        path,
        params=None,
        data: str = "",
        headers: dict[str, str] | None = None,
        connections_timeout=None,
    ):
        uri = self.uri(path, params)
        return self._request(callback, "PUT", uri, headers=headers, data=data, connections_timeout=connections_timeout)

    def delete(self, callback, path, params=None, headers: dict[str, str] | None = None, connections_timeout=None):
        uri = self.uri(path, params)
        return self._request(callback, "DELETE", uri, headers=headers, connections_timeout=connections_timeout)

    def post(
        self,
        callback,
        path,
        params=None,
        data: str = "",
        headers: dict[str, str] | None = None,
        connections_timeout=None,
    ):
        uri = self.uri(path, params)
        return self._request(callback, "POST", uri, headers=headers, data=data, connections_timeout=connections_timeout)

    def close(self):
        return self._session.close()


class Consul(base.Consul):
    def __init__(
        self,
        *args,
        loop=None,
        connections_limit=None,
        connections_timeout=None,
        dns_cache_ttl=None,
        connection_retries=0,
        **kwargs,
    ) -> None:
        self.loop = loop
        self.connections_limit = connections_limit
        self.connections_timeout = connections_timeout
        self.dns_cache_ttl = dns_cache_ttl
        self.connection_retries = connection_retries
        super().__init__(*args, **kwargs)

    def http_connect(self, host: str, port: int, scheme, verify: bool | str = True, cert=None):
        return HTTPClient(
            host,
            port,
            scheme,
            loop=self.loop,
            connections_limit=self.connections_limit,
            connections_timeout=self.connections_timeout,
            dns_cache_ttl=self.dns_cache_ttl,
            connection_retries=self.connection_retries,
            verify=verify,
            cert=cert,
        )

    def close(self):
        """Close all opened http connections"""
        return self.http.close()
