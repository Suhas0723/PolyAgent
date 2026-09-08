import logging
from dataclasses import dataclass

LOG = logging.getLogger("polyagent.executor")


@dataclass
class FillResult:
    status: str
    shares: float
    price: float
    notional: float
    fees: float
    external_order_id: str = ""
    error: str = ""


class ExecutionError(RuntimeError):
    pass


class BaseExecutor:
    mode = "base"

    def __init__(self, config):
        self.config = config

    def buy(self, market, side, shares, limit_price):
        raise NotImplementedError

    def close(self):
        pass


class PaperExecutor(BaseExecutor):
    mode = "paper"

    def buy(self, market, side, shares, limit_price):
        book_price = market.price_for(side)
        if book_price is None:
            return FillResult("rejected", 0.0, 0.0, 0.0, 0.0, error="no price for side")

        fill_price = min(book_price + self.config.slippage_buffer, 0.999)
        if fill_price > limit_price:
            return FillResult(
                "rejected", 0.0, fill_price, 0.0, 0.0,
                error=f"simulated fill {fill_price:.4f} exceeds limit {limit_price:.4f}",
            )

        notional = shares * fill_price
        fees = notional * self.config.fee_rate
        LOG.info("paper fill %s %s %.2f shares @ %.4f (%.2f USD)",
                 market.condition_id, side, shares, fill_price, notional)
        return FillResult(
            "filled", shares, fill_price, notional, fees,
            external_order_id=f"paper-{market.condition_id[:10]}-{side.lower()}",
        )


class ClobExecutor(BaseExecutor):
    mode = "live"

    def __init__(self, config):
        super().__init__(config)
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY
        except ImportError as exc:
            raise ExecutionError(
                "live mode requires py-clob-client; pip install py-clob-client"
            ) from exc

        self._OrderArgs = OrderArgs
        self._BUY = BUY

        if not config.clob_private_key:
            raise ExecutionError("live mode requires POLYAGENT_CLOB_PRIVATE_KEY")

        self.client = ClobClient(
            host=config.clob_base_url,
            key=config.clob_private_key,
            chain_id=config.clob_chain_id,
            funder=config.clob_funder_address or None,
            signature_type=2 if config.clob_funder_address else 0,
        )
        if config.clob_api_key and config.clob_api_secret and config.clob_api_passphrase:
            from py_clob_client.clob_types import ApiCreds
            self.client.set_api_creds(
                ApiCreds(
                    api_key=config.clob_api_key,
                    api_secret=config.clob_api_secret,
                    api_passphrase=config.clob_api_passphrase,
                )
            )
        else:
            self.client.set_api_creds(self.client.create_or_derive_api_creds())
        LOG.warning("live CLOB executor initialized against %s", config.clob_base_url)

    def buy(self, market, side, shares, limit_price):
        token_id = market.token_for(side)
        if not token_id:
            return FillResult("rejected", 0.0, 0.0, 0.0, 0.0, error="no CLOB token id for side")

        order_args = self._OrderArgs(
            token_id=token_id,
            price=round(min(limit_price, 0.999), 3),
            size=round(shares, 2),
            side=self._BUY,
        )
        try:
            signed = self.client.create_order(order_args)
            response = self.client.post_order(signed)
        except Exception as exc:
            LOG.exception("live order failed for %s", market.condition_id)
            return FillResult("error", 0.0, 0.0, 0.0, 0.0, error=str(exc)[:500])

        success = bool(response.get("success", False)) if isinstance(response, dict) else False
        order_id = str(response.get("orderID", "")) if isinstance(response, dict) else ""
        if not success:
            message = response.get("errorMsg", "unknown") if isinstance(response, dict) else str(response)
            return FillResult("rejected", 0.0, order_args.price, 0.0, 0.0,
                              external_order_id=order_id, error=str(message)[:500])

        notional = order_args.size * order_args.price
        return FillResult(
            "submitted", order_args.size, order_args.price, notional,
            notional * self.config.fee_rate, external_order_id=order_id,
        )


def build_executor(config):
    if config.execution_mode == "live":
        return ClobExecutor(config)
    return PaperExecutor(config)
