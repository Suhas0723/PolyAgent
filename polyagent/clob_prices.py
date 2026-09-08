import asyncio
import logging

import httpx

from .gamma import to_float

LOG = logging.getLogger("polyagent.clob_prices")


class BookPriceClient:
    def __init__(self, config):
        self.config = config
        self.base_url = config.clob_base_url.rstrip("/")
        self._client = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.gamma_timeout_seconds),
            headers={"Accept": "application/json", "User-Agent": "polyagent/1.0"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _price(self, token_id, side):
        try:
            response = await self._client.get(
                f"{self.base_url}/price", params={"token_id": token_id, "side": side}
            )
            if response.status_code != 200:
                return None
            return to_float(response.json().get("price"))
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            LOG.debug("clob price lookup failed for %s/%s: %s", token_id, side, exc)
            return None

    async def annotate(self, markets):
        semaphore = asyncio.Semaphore(self.config.gamma_concurrency)
        annotated = 0

        async def worker(market):
            nonlocal annotated
            yes_token = market.yes_token_id
            no_token = market.no_token_id
            if not yes_token or not no_token:
                return
            async with semaphore:
                yes_ask, no_ask = await asyncio.gather(
                    self._price(yes_token, "sell"),
                    self._price(no_token, "sell"),
                )
            if yes_ask is not None and no_ask is not None:
                market.yes_ask = yes_ask
                market.no_ask = no_ask
                annotated += 1

        await asyncio.gather(*(worker(market) for market in markets))
        LOG.info("annotated %s/%s markets with executable book prices", annotated, len(markets))
        return annotated
