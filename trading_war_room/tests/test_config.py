import pytest

from warroom.config import ConfigError, Secrets, load_config, parse_config, validate_for_mode
from warroom.models.schemas import Mode
from warroom.providers import factory


def test_default_config_loads(cfg):
    assert cfg.app.tickers and cfg.risk.max_risk_pct == 0.01 and cfg.risk.max_position_pct == 0.25


@pytest.mark.parametrize("bad,needle", [
    ({"risk": {"max_risk_pt": 0.01}}, "Extra inputs"),
    ({"risk": {"max_risk_pct": 0.5}}, "less than or equal"),
    ({"risk": {"tp1_r_multiple": 3, "tp2_r_multiple": 2}}, "tp2_r_multiple"),
    ({"risk": {"tp1_r_multiple": 1.5}}, "every trade would be vetoed"),
    ({"risk": {"max_risk_pct": 0.04, "max_daily_loss_pct": 0.03}}, "max_risk_pct must be smaller"),
    ({"app": {"tickers": ["SPY"]}}, "overlap"),
    ({"app": {"tickers": ["BAD TICKER"]}}, "invalid ticker"),
    ({"demo": {"start_time_et": "08:00"}}, "regular hours"),
    ({"providers": {"rss_feeds": ["ftp://x"]}}, "http"),
])
def test_invalid_configs_fail_fast(bad, needle):
    with pytest.raises(ConfigError, match=needle):
        parse_config(bad)


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    p = tmp_path / "c.yaml"
    p.write_text("app: [unclosed")
    with pytest.raises(ConfigError, match="parse"):
        load_config(p)


def test_paper_mode_rejects_mock_data_and_live_money(cfg):
    with pytest.raises(ConfigError, match="real market data provider"):
        validate_for_mode(cfg, Secrets.from_env({}), Mode.PAPER)
    with pytest.raises(ConfigError, match="ALPACA_PAPER"):
        validate_for_mode(cfg, Secrets.from_env({"ALPACA_PAPER": "false"}), Mode.DEMO)
    assert validate_for_mode(cfg, Secrets.from_env({}), Mode.DEMO) == []


def test_paper_alpaca_needs_keys_and_yfinance_warns():
    alp = parse_config({"providers": {"market_data": "alpaca"}})
    with pytest.raises(ConfigError, match="ALPACA_API_KEY"):
        validate_for_mode(alp, Secrets.from_env({}), Mode.PAPER)
    yf = parse_config({"providers": {"market_data": "yfinance"}})
    warns = validate_for_mode(yf, Secrets.from_env({}), Mode.PAPER)
    assert any("spread vetoes" in w for w in warns)


def test_secrets_never_leak_in_repr():
    s = Secrets.from_env({"ANTHROPIC_API_KEY": "sk-secret-123"})
    assert "sk-secret-123" not in repr(s) and "sk-secret-123" not in str(s.model_dump())


def test_per_mode_database_paths(cfg):
    assert cfg.storage.resolve_db_path(Mode.PAPER) == "data/trades_paper.db"
    assert cfg.storage.resolve_db_path(Mode.DEMO) == "data/trades_demo.db"


def test_paper_never_silently_falls_back_to_mock(cal):
    from warroom.core.clock import SystemClock

    yf = parse_config({"providers": {"market_data": "yfinance"}})
    try:
        import yfinance  # noqa: F401
    except ImportError:
        with pytest.raises(ConfigError, match="yfinance"):
            factory.build_market_data(yf, Secrets.from_env({}), SystemClock(), cal, Mode.PAPER)
    demo = factory.build_market_data(yf, Secrets.from_env({}), SystemClock(), cal, Mode.DEMO)
    assert type(demo).__name__ == "RoutedMarketData"
    assert type(demo.equity).__name__ == "MockDataProvider" and type(demo.crypto).__name__ == "MockCryptoProvider"
    with pytest.raises(ConfigError, match="fake prices"):
        raw = yf.model_dump(mode="json")
        raw["crypto"]["data_provider"] = "mock"
        factory.build_crypto_market_data(parse_config(raw), SystemClock(), Mode.PAPER)
