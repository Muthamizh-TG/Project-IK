# ============================================================================
# LIVE PRICE FEED (REAL-TIME, LOCAL TIME PRINTING)
# Prints NEW 1-minute bars only. No history.
# ============================================================================

import alpaca_trade_api as alpaca
import pandas as pd
import asyncio
import logging
from alpaca_trade_api.rest import TimeFrame

ALPACA_API_KEY = "PKBYKHPY55D7GIPJKUSFG4M5GR"
ALPACA_SECRET_KEY = "8JTRKC7mQN1FY6XXZzPDMu9ZBuvRVQtzkKufvdEn93da"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("feed")


class LiveFeed:
    def __init__(self, api, symbol, crypto=False):
        self.api = api
        self.crypto = crypto
        self.symbol = symbol.upper()

        if crypto:
            if "/" in self.symbol:
                self.api_symbol = self.symbol
            elif self.symbol.endswith("USD"):
                self.api_symbol = self.symbol[:-3] + "/USD"
            else:
                self.api_symbol = self.symbol
        else:
            self.api_symbol = self.symbol

        self.last_ts = None

        log.info(f"Started live feed for {self.symbol} (API: {self.api_symbol})")

    def _now(self):
        return pd.Timestamp.utcnow().floor("1min")

    def fetch_new_bar(self):
        try:
            now = self._now()

            if self.last_ts is None:
                self.last_ts = now
                return

            if now <= self.last_ts:
                return

            start = (self.last_ts + pd.Timedelta("1min")).strftime('%Y-%m-%dT%H:%M:%SZ')
            end = now.strftime('%Y-%m-%dT%H:%M:%SZ')

            if self.crypto:
                try:
                    data = self.api.get_crypto_bars(
                        self.api_symbol, TimeFrame.Minute, start=start, end=end
                    )
                    data = getattr(data, "df", data)
                except AttributeError:
                    data = self.api.get_bars(
                        self.api_symbol, TimeFrame.Minute, start, end, "raw"
                    ).df
            else:
                data = self.api.get_bars(
                    self.api_symbol, TimeFrame.Minute, start, end, "raw"
                ).df

            if len(data) == 0:
                return

            for ts, row in data.iterrows():
                if ts > self.last_ts:
                    self.last_ts = ts

                    # Convert UTC timestamp → Local Time (IST)
                    local_ts = ts.tz_convert("Asia/Kolkata")

                    log.info(
                        f"[{self.symbol}] {local_ts} | "
                        f"O:{row['open']} H:{row['high']} L:{row['low']} "
                        f"C:{row['close']} Vol:{row['volume']}"
                    )

        except Exception as e:
            log.error(f"Error during fetch: {e}")


async def main(args):
    api = alpaca.REST(
        key_id=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        base_url="https://paper-api.alpaca.markets"
    )

    feeds = [LiveFeed(api, s, crypto=args.crypto) for s in args.symbols]

    while True:
        for f in feeds:
            f.fetch_new_bar()
        await asyncio.sleep(5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("symbols", nargs="*", default=["XRPUSD"])
    parser.add_argument("--crypto", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args))
