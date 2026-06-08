from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd

from v8641_production.config import ProductionConfig
from v8641_production.data_update import MarketDataCache, DailyMarketDataUpdater


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        cfg = ProductionConfig(
            input_dir=Path('/mnt/data'),
            out_dir=Path(td) / 'out',
            assets=['QQQ'],
            cache_dir=Path(td) / 'cache',
            export_json=False,
            export_csv=False,
            export_markdown=False,
            make_zip=False,
        )
        cache = MarketDataCache(cfg)
        path = cache.ohlcv_path('QQQ', 'yahoo')
        path.parent.mkdir(parents=True, exist_ok=True)
        today = pd.Timestamp.utcnow().date().isoformat()
        pd.DataFrame([
            {'Date': today, 'Open': 1, 'High': 1, 'Low': 1, 'Close': 1, 'Adj_Close': 1, 'Volume': 100}
        ]).to_csv(path, index=False)
        fresh = cache.freshness(['QQQ'], 'yahoo')[0]
        assert fresh.exists is True
        assert fresh.is_fresh_daily is True
        assert fresh.row_count == 1
        updater = DailyMarketDataUpdater(cfg)
        status = updater.update_daily(['QQQ'], provider='kis')
        assert status.ok is False
        assert 'kis' in status.errors
        print('daily cache freshness test passed')


if __name__ == '__main__':
    main()
