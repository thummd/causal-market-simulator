"""ABIDES ground-truth market-impact episodes (Phase 2).

Inject a metaorder into an agent-based market, replay the *same seed* without
it, and package the pair as a dotime :class:`Episode` — the agent-based
analogue of the synthetic prior's shared-noise counterfactual pair, except
here the "SCM" is a full limit-order-book ecology and nobody knows lambda.

Design (see HANDOVER 2026-07-08 for the feasibility notes):

- **Config** is an rmsc04 derivative with ``order_size_model=None`` everywhere
  (agents fall back to uniform sizes) — this is what removes the
  ``pomegranate`` dependency that blocks stock rmsc04 on modern Python.
- **Seed replay**: per-agent seeds are drawn sequentially from the global
  numpy seed in agent-list order, so both runs build the identical background
  ecology. The metaorder agent is placed *last* and exists in BOTH runs —
  with ``quantity_per_wake = 0`` in the observational run — so the agent
  count (which the latency model and kernel seed derive from) and every
  background RNG stream match exactly. Pre-onset paths are then identical;
  post-onset divergence is the causal effect of the metaorder.
- **Episode encoding** (canonical order): variable 0 = signed transacted
  flow per bar (buy volume − sell volume, shares); variable 1 = mid price in
  log basis points (``1e4 * log(mid / open_mid)``). The intervention is
  ``targets=[0]`` over the execution-window bars, HARD, with value = the
  metaorder's signed shares per bar. Queries go on the price post-onset,
  ``y_true`` from the WITH-metaorder run.

Requires the ``abides`` extra (git install of abides-core + abides-markets;
see pyproject). All abides imports live inside this module so the rest of
``dotime_market.data`` imports without it.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from dotime import InterventionSpec, InterventionType
from dotime.benchmarks import BenchmarkSuite, Episode, SuiteMetadata

__all__ = ["MetaorderAgent", "build_market_config", "run_metaorder_pair",
           "abides_episode", "build_abides_suite"]

_STRUCTURE = "abides_metaorder"

try:  # soft dependency: the module is importable without abides installed
    from abides_core import NanosecondTime  # noqa: F401
    from abides_markets.agents import (
        AdaptiveMarketMakerAgent,
        ExchangeAgent,
        MomentumAgent,
        NoiseAgent,
        TradingAgent,
        ValueAgent,
    )
    from abides_markets.oracles import SparseMeanRevertingOracle
    from abides_markets.orders import Side
    from abides_markets.utils import generate_latency_model

    ABIDES_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    ABIDES_AVAILABLE = False
    TradingAgent = object  # type: ignore[misc,assignment]


def str_to_ns(string: str) -> int:
    """Duration string -> integer nanoseconds.

    Deliberately NOT ``abides_core.utils.str_to_ns``: that helper does
    ``to_timedelta64().astype(int)``, which under pandas >= 2 yields
    *microseconds* (non-nano resolution), silently shrinking every duration
    1000x — with a 0.9-second "trading day" as the symptom.
    ``pd.Timedelta.value`` is nanoseconds in every pandas version.
    """
    import pandas as pd

    return int(pd.Timedelta(string).value)


class MetaorderAgent(TradingAgent):
    """Executes a parent order as fixed-size market orders on a fixed clock.

    With ``quantity_per_wake = 0`` the agent still joins the simulation (same
    wakeups, same market-hours handshake) but never sends an order — this is
    the observational twin, keeping both runs structurally identical.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        exec_start: int,
        exec_end: int,
        wake_freq: int,
        quantity_per_wake: int,
        side: str = "BUY",
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders=False)
        self.symbol = symbol
        self.exec_start = exec_start
        self.exec_end = exec_end
        self.wake_freq = wake_freq
        self.quantity_per_wake = int(quantity_per_wake)
        self.side = Side.BID if side == "BUY" else Side.ASK
        self.submitted_quantity = 0

    def get_wake_frequency(self) -> int:
        # Called once market hours are known: offset from mkt_open such that
        # the first trading wakeup lands exactly on exec_start.
        return self.exec_start - self.mkt_open

    def wakeup(self, current_time) -> None:
        can_trade = super().wakeup(current_time)
        if not can_trade:
            return  # base class schedules the exec_start wakeup on hours reply
        if current_time < self.exec_start:
            self.set_wakeup(self.exec_start)
            return
        if current_time >= self.exec_end:
            return
        if self.quantity_per_wake > 0:
            self.place_market_order(self.symbol, self.quantity_per_wake, self.side)
            self.submitted_quantity += self.quantity_per_wake
        self.set_wakeup(current_time + self.wake_freq)


def build_market_config(
    seed: int,
    metaorder_quantity_per_wake: int,
    date: str = "20210205",
    end_time: str = "10:30:00",
    exec_window: tuple = ("09:45:00", "10:00:00"),
    metaorder_wake_freq: str = "20s",
    metaorder_side: str = "BUY",
    ticker: str = "ABM",
    starting_cash: int = 10_000_000,
    num_noise_agents: int = 200,
    num_value_agents: int = 50,
    num_momentum_agents: int = 6,
    num_mm_agents: int = 2,
    r_bar: int = 100_000,
    kappa: float = 1.67e-15,
    lambda_a: float = 5.7e-12,
    fund_vol: float = 5e-5,
    value_sigma_n: Optional[float] = None,
    book_log_depth: int = 10,
) -> dict:
    """rmsc04-derived, pomegranate-free kernel config with the metaorder last.

    Every random draw happens in the same order regardless of
    ``metaorder_quantity_per_wake``, so a (0, Q) pair of configs at the same
    seed differs ONLY in the orders the metaorder agent submits.
    """
    if not ABIDES_AVAILABLE:
        raise ImportError(
            "abides-core/abides-markets are not installed; "
            'install the "abides" extra (see pyproject.toml)'
        )
    np.random.seed(seed)

    date_ns = int(np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:]}").astype("datetime64[ns]").astype(np.int64))
    mkt_open = date_ns + str_to_ns("09:30:00")
    mkt_close = date_ns + str_to_ns(end_time)
    # Noise arrivals are spread uniformly over [noise_open, noise_close].
    # rmsc04 uses a full-day close (16:00) for a 6.5h session; keeping that
    # for a short session parks most noise agents after mkt_close and the
    # book goes so thin that the adaptive MM quotes negative prices. Tie the
    # window to the actual session instead, and size num_noise_agents for
    # the rmsc04-like density of ~150 arrivals per simulated hour.
    noise_open = mkt_open - str_to_ns("00:30:00")
    noise_close = mkt_close

    symbols = {
        ticker: {
            "r_bar": r_bar,
            "kappa": 1.67e-16,
            "sigma_s": 0,
            "fund_vol": fund_vol,
            "megashock_lambda_a": 2.77778e-18,
            "megashock_mean": 1000,
            "megashock_var": 50_000,
            "random_state": np.random.RandomState(
                seed=np.random.randint(low=0, high=2**32)
            ),
        }
    }
    oracle = SparseMeanRevertingOracle(mkt_open, noise_close, symbols)

    def _rs() -> np.random.RandomState:
        return np.random.RandomState(seed=np.random.randint(low=0, high=2**32, dtype="uint64"))

    agents: list = [
        ExchangeAgent(
            id=0,
            name="EXCHANGE_AGENT",
            type="ExchangeAgent",
            mkt_open=mkt_open,
            mkt_close=mkt_close,
            symbols=[ticker],
            book_logging=True,
            book_log_depth=book_log_depth,
            log_orders=False,
            pipeline_delay=0,
            computation_delay=0,
            stream_history=500,
            random_state=_rs(),
        )
    ]
    aid = 1

    from abides_core.utils import get_wake_time

    for _ in range(num_noise_agents):
        agents.append(
            NoiseAgent(
                id=aid, name=f"NoiseAgent {aid}", type="NoiseAgent", symbol=ticker,
                starting_cash=starting_cash,
                wakeup_time=get_wake_time(noise_open, noise_close),
                log_orders=False, order_size_model=None, random_state=_rs(),
            )
        )
        aid += 1

    for _ in range(num_value_agents):
        agents.append(
            ValueAgent(
                id=aid, name=f"Value Agent {aid}", type="ValueAgent", symbol=ticker,
                starting_cash=starting_cash,
                sigma_n=value_sigma_n if value_sigma_n is not None else r_bar / 100,
                r_bar=r_bar, kappa=kappa, lambda_a=lambda_a,
                log_orders=False, order_size_model=None, random_state=_rs(),
            )
        )
        aid += 1

    for _ in range(num_mm_agents):
        agents.append(
            AdaptiveMarketMakerAgent(
                id=aid, name=f"ADAPTIVE_POV_MARKET_MAKER_AGENT_{aid}",
                type="AdaptivePOVMarketMakerAgent", symbol=ticker,
                starting_cash=starting_cash, pov=0.025, min_order_size=1,
                window_size="adaptive", num_ticks=10,
                wake_up_freq=str_to_ns("60s"), poisson_arrival=True,
                cancel_limit_delay=50, skew_beta=0, price_skew_param=4,
                level_spacing=5, spread_alpha=0.75, backstop_quantity=0,
                log_orders=False, random_state=_rs(),
            )
        )
        aid += 1

    for _ in range(num_momentum_agents):
        agents.append(
            MomentumAgent(
                id=aid, name=f"MOMENTUM_AGENT_{aid}", type="MomentumAgent",
                symbol=ticker, starting_cash=starting_cash, min_size=1, max_size=10,
                wake_up_freq=str_to_ns("37s"), poisson_arrival=True,
                log_orders=False, order_size_model=None, random_state=_rs(),
            )
        )
        aid += 1

    # The metaorder agent comes LAST so all background seeds above are
    # untouched by its presence; it exists in both runs (Q=0 observational).
    agents.append(
        MetaorderAgent(
            id=aid, name="METAORDER_AGENT", type="MetaorderAgent", symbol=ticker,
            starting_cash=starting_cash,
            exec_start=date_ns + str_to_ns(exec_window[0]),
            exec_end=date_ns + str_to_ns(exec_window[1]),
            wake_freq=str_to_ns(metaorder_wake_freq),
            quantity_per_wake=metaorder_quantity_per_wake,
            side=metaorder_side, random_state=_rs(),
        )
    )
    aid += 1

    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
    )
    latency_model = generate_latency_model(aid)

    return {
        "seed": seed,
        "start_time": date_ns,
        "stop_time": mkt_close + str_to_ns("1s"),
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": 50,
        "custom_properties": {"oracle": oracle},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": "ERROR",
        # bookkeeping for the bar extraction below
        "_ticker": ticker,
        "_mkt_open": mkt_open,
        "_mkt_close": mkt_close,
        "_exec_start": date_ns + str_to_ns(exec_window[0]),
        "_exec_end": date_ns + str_to_ns(exec_window[1]),
    }


def _run_and_bin(config: dict, bar_ns: int) -> dict:
    """Run one kernel and return per-bar (times, mid, flow, submitted)."""
    from abides_core import abides

    meta = {k: config[k] for k in list(config) if k.startswith("_")}
    kernel_config = {k: v for k, v in config.items() if not k.startswith("_")}
    end_state = abides.run(kernel_config)

    exchange = end_state["agents"][0]
    book = exchange.order_books[meta["_ticker"]]
    metaorder = end_state["agents"][-1]

    t0, t1 = meta["_mkt_open"], meta["_mkt_close"]
    n_bars = int((t1 - t0) // bar_ns)
    edges = t0 + bar_ns * np.arange(n_bars + 1)

    # Mid price: last L2 snapshot in each bar, forward-filled.
    mid = np.full(n_bars, np.nan)
    for row in book.book_log2:
        t = row["QuoteTime"]
        if t < t0 or t >= t1 or len(row["bids"]) == 0 or len(row["asks"]) == 0:
            continue
        b = int((t - t0) // bar_ns)
        mid[b] = 0.5 * (row["bids"][0][0] + row["asks"][0][0])
    for i in range(1, n_bars):
        if math.isnan(mid[i]):
            mid[i] = mid[i - 1]
    first = next((i for i in range(n_bars) if not math.isnan(mid[i])), 0)
    mid[: first + 1] = mid[first]

    # Signed transacted flow per bar (all trades, metaorder included).
    flow = np.zeros(n_bars)
    for t, q in book.buy_transactions:
        if t0 <= t < t1:
            flow[int((t - t0) // bar_ns)] += q
    for t, q in book.sell_transactions:
        if t0 <= t < t1:
            flow[int((t - t0) // bar_ns)] -= q

    return {
        "times": (edges[1:] - t0) / 1e9,  # bar-end seconds since open
        "mid": mid,
        "flow": flow,
        "submitted": metaorder.submitted_quantity,
        "onset_bar": int((meta["_exec_start"] - t0) // bar_ns),
        "end_bar": int(np.ceil((meta["_exec_end"] - t0) / bar_ns)),
    }


def run_metaorder_pair(
    seed: int,
    quantity_per_wake: int = 40,
    bar: str = "30s",
    **config_kw,
) -> dict:
    """(observational, interventional) binned pair at one seed."""
    bar_ns = str_to_ns(bar)
    obs = _run_and_bin(build_market_config(seed, 0, **config_kw), bar_ns)
    intv = _run_and_bin(build_market_config(seed, quantity_per_wake, **config_kw), bar_ns)
    n_exec_bars = max(intv["end_bar"] - intv["onset_bar"], 1)
    return {
        "obs": obs,
        "int": intv,
        "seed": seed,
        "shares_per_bar": intv["submitted"] / n_exec_bars,
    }


def abides_episode(
    pair: dict,
    rng: np.random.RandomState,
    n_queries: int = 1,
    standardize: bool = True,
    intervention_kind: str = "hard",
) -> Episode:
    """Package one seed-replay pair as a dotime Episode (flow=0, price=1).

    ``standardize=True`` (default) divides each column by its pre-onset
    standard deviation and expresses the do-value in the same rescaled flow
    units. This puts ABM episodes in training-prior-like units: the trained
    PFN's ``intervention_value`` mixer input is consumed raw (only the
    trajectories are z-scored downstream), so an unscaled ~75-shares/bar
    value lands ~19 sigma outside the N(0, 2) values seen in training —
    the observed +2 log-bps transfer overshoot. Scaling is uniform across
    methods (all are affine-equivariant), so rankings are unaffected and
    errors are reported in pre-onset-sigma units; the scalers are recorded
    in ``metadata`` for conversion back to shares / log-bps.
    """
    obs, intv = pair["obs"], pair["int"]
    T = len(obs["mid"])
    onset, end = intv["onset_bar"], intv["end_bar"]

    def _xy(run: dict) -> torch.Tensor:
        price_bps = 1e4 * np.log(run["mid"] / run["mid"][0])
        return torch.tensor(np.column_stack([run["flow"], price_bps]), dtype=torch.float32)

    x_obs, x_int = _xy(obs), _xy(intv)
    sign = 1.0 if intv["submitted"] >= 0 else -1.0
    value = sign * pair["shares_per_bar"]

    flow_scale = price_scale = 1.0
    if standardize:
        # Pre-onset windows of the two runs are identical (seed replay), so
        # one set of scalers applies to both trajectories and the do-value.
        flow_scale = float(x_obs[:onset, 0].std().clamp_min(1e-6))
        price_scale = float(x_obs[:onset, 1].std().clamp_min(1e-6))
        scales = torch.tensor([flow_scale, price_scale], dtype=torch.float32)
        x_obs = x_obs / scales
        x_int = x_int / scales
        value = value / flow_scale

    if intervention_kind not in ("hard", "soft"):
        raise ValueError(f"intervention_kind must be 'hard' or 'soft', got {intervention_kind!r}")
    # "soft" is the semantically faithful encoding: the metaorder ADDS
    # ~value shares/bar to the background flow (dotime SOFT = additive drift
    # shift on the target), it does not clamp total flow. "hard" is kept for
    # models trained hard-only (v1-v3).
    kind = InterventionType.HARD if intervention_kind == "hard" else InterventionType.SOFT
    intervention = InterventionSpec(
        targets=[0],
        times=list(range(onset, max(end, onset + 1))),
        intervention_type=kind,
        values=float(value),
    )
    hi = max(onset + 1, T - 1)
    q_idx = rng.randint(onset, hi + 1, size=n_queries)
    return Episode(
        x_obs=x_obs,
        x_int=x_int,
        intervention=intervention,
        y_true=x_int[q_idx, 1].to(torch.float32),
        query_target=torch.full((n_queries,), 1, dtype=torch.long),
        query_time=torch.tensor(q_idx / max(T - 1, 1), dtype=torch.float32),
        structure=_STRUCTURE,
        scm_id=pair["seed"],
        metadata={
            "query_idx": [int(i) for i in q_idx],
            "submitted_shares": int(intv["submitted"]),
            "onset_bar": onset,
            "end_bar": end,
            "flow_scale": flow_scale,
            "price_scale": price_scale,
        },
    )


def build_abides_suite(
    n_episodes: int,
    seed0: int = 0,
    quantity_per_wake: int = 40,
    bar: str = "30s",
    n_queries: int = 1,
    name: str = "abides-metaorder",
    standardize: bool = True,
    intervention_kind: str = "hard",
    **config_kw,
) -> BenchmarkSuite:
    """Consecutive-seed metaorder episodes as an in-memory suite."""
    rng = np.random.RandomState(seed0)
    episodes = [
        abides_episode(
            run_metaorder_pair(seed0 + i, quantity_per_wake=quantity_per_wake,
                               bar=bar, **config_kw),
            rng=rng,
            n_queries=n_queries,
            standardize=standardize,
            intervention_kind=intervention_kind,
        )
        for i in range(n_episodes)
    ]
    meta = SuiteMetadata(
        name=name, version="0.0.1", zenodo_record_id="", doi="",
        description="ABIDES metaorder episodes with seed-replay counterfactuals.",
        n_episodes=n_episodes, structures=(_STRUCTURE,),
    )
    return BenchmarkSuite(meta, episodes)
