"""The Odds API — cup fixtures and prices behind one key (Plan §4 Phase D).

The only source found that covers all twelve competitions in the taxonomy,
including the six cups that have no keyless option: openfootball stopped
publishing them, and OpenLigaDB covers Germany alone.

Without a key this module is a no-op. Every entry point returns None and the
caller keeps whatever it already had, exactly as `api_football` and
`football_data_org` behave. A missing credential is a capability this deployment
does not have, not an error.

The budget is the design constraint
===================================

The free tier is 500 credits a **month**, not a day, which makes exhaustion far
less recoverable than API-Football's: burn it on the 3rd and there is nothing
until the 1st. Two things follow, and both are enforced here rather than left to
callers.

**`/events` is free.** It returns id, both clubs and the kick-off time for every
upcoming fixture in a competition, and costs 0 credits. That is the whole of what
a fixture list needs, so cup *fixtures* cost nothing at all. Credits are spent
only on `/odds`.

**The remaining balance is read from the response, not counted locally.** Every
reply carries `x-requests-remaining` and `x-requests-used`. A local counter is
wrong the moment the key is used from anywhere else — another machine, a
scheduled job, a manual curl — and being wrong in the optimistic direction is
what empties a monthly budget. The header is authoritative; the local counter
exists only to bound spending between reads, and a cache hit reports no headers
at all so it must never be read as "zero remaining".

What it does not give
=====================

**No round.** An event carries clubs and a kick-off, not "quarter-final". Stage
drives `taxonomy.resolve_format`, so a fixture sourced here is emitted with
`stage_confirmed=false` and falls back to the competition default. For a
single-leg cup that is right; for a two-legged tie it is not, and the flag is
what lets a consumer tell the difference rather than discovering it from a
mispriced second leg.

**No history.** The historical endpoints are paid-tier only, so a cup priced
from here gains `live_odds_coverage` and never `benchmark_coverage` — which is
precisely the split those two flags exist to express. Phase C measured why that
matters: a price with nothing to validate it against is not a recommendation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from statpitch import taxonomy
from statpitch.data.http import FetchError, PoliteSession

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

ENV_KEY = "STATPITCH_ODDS_API_KEY"

#: Free tier. Monthly, not daily — see the module docstring.
MONTHLY_CREDITS = 500

#: Stop here rather than at zero, so a scheduled job cannot leave the month with
#: no way to look at anything. Mirrors `quota.DEFAULT_HARD_STOP`'s reasoning.
DEFAULT_RESERVE = 50

#: competition_id -> The Odds API sport key. Verified against the published sport
#: list; the six cups here are the ones with no free alternative.
SPORT_KEYS: dict[str, str] = {
    "ENG.PL": "soccer_epl",
    "ESP.LALIGA": "soccer_spain_la_liga",
    "GER.BUNDESLIGA": "soccer_germany_bundesliga",
    "ITA.SERIEA": "soccer_italy_serie_a",
    "FRA.LIGUE1": "soccer_france_ligue_one",
    "ENG.FA_CUP": "soccer_fa_cup",
    "ESP.COPA_DEL_REY": "soccer_spain_copa_del_rey",
    "GER.DFB_POKAL": "soccer_germany_dfb_pokal",
    "ITA.COPPA_ITALIA": "soccer_italy_coppa_italia",
    "FRA.COUPE_DE_FRANCE": "soccer_france_coupe_de_france",
    "UEFA.UCL": "soccer_uefa_champs_league",
    "UEFA.UEL": "soccer_uefa_europa_league",
}

#: Competitions with no free alternative, which is what this key buys.
UNSOURCED_WITHOUT_A_KEY = (
    "ENG.FA_CUP", "ESP.COPA_DEL_REY", "ITA.COPPA_ITALIA",
    "FRA.COUPE_DE_FRANCE", "UEFA.UCL", "UEFA.UEL",
)

#: Bookmaker keys worth extracting by name rather than only aggregating.
#:
#: `pinnacle` is the one that matters, and its presence here is why this module
#: grew a price fetcher at all. MODEL_CARD 5's +0.51% CLV (t=+7.53 clustered,
#: five pre-break seasons) is defined on Pinnacle-referenced selections, and
#: Phase C's blocker was that football-data.co.uk's fixture feed does not publish
#: Pinnacle -- so the only rule with multi-season evidence could be measured
#: backwards and never traded forwards. This source publishes it.
#:
#: `betfair_ex_eu` is the post-break candidate Phase C ranked highest but could
#: not validate on a single season. Carried so that comparison can continue live.
NAMED_BOOKS: dict[str, str] = {
    "pinnacle": "odds_pinnacle",
    "betfair_ex_eu": "odds_bfe",
}

#: Odds API market key -> the market name used in `live_odds.parquet`.
MARKET_NAMES: dict[str, str] = {
    "h2h": "1x2", "totals": "ou", "spreads": "ah",
}

#: What a daily sweep asks for. One market, so one credit per competition.
DAILY_MARKETS = ("h2h",)

#: What a competition playing today gets. Three markets, so three credits, spent
#: only where a fixture actually kicks off, which is where they are worth having.
MATCHDAY_MARKETS = ("h2h", "totals", "spreads")

#: How near a kickoff has to be before a credit is spent on it.
#:
#: `/events` is free and reports every fixture the API knows about, which runs
#: well over a week ahead. Paying for all of them is the single largest waste
#: available here: a bookmaker opens a market roughly a week out, so a request
#: for a fixture beyond that returns an empty or near-empty board and is charged
#: in full — and it would be charged again tomorrow, and the day after.
#:
#: Three days is chosen from what the price is *for* rather than from the
#: budget. MODEL_CARD §5's finding is Friday-to-close CLV, so the baseline it is
#: defined on sits about two days out; three keeps that reachable with a day in
#: hand for a run that fails and retries. Fixtures further out are not left
#: blank — `build_card` prices them at `1/p_model` and marks them
#: `pricing="model"`, which is the honest label for a price nobody is offering.
PRICING_HORIZON_DAYS = 3

#: Events describe the future; a cached copy must expire. One hour rather than
#: the schedule sources' six, because this is also the path a near-kickoff odds
#: capture would key off.
EVENTS_MAX_AGE_SECONDS = 3600


def api_key() -> str | None:
    key = os.environ.get(ENV_KEY, "").strip()
    return key or None


def configured() -> bool:
    """Whether a key is present. False is a normal state, not a failure."""
    return api_key() is not None


@dataclass
class Budget:
    """What the API last said was left, and what has been spent since.

    Deliberately not persisted. `quota.QuotaBudget` writes a daily counter to
    disk because API-Football reports nothing back; here the server reports the
    truth on every call, so a local file could only ever be a stale second
    opinion competing with it.
    """

    remaining: int | None = None
    used: int | None = None
    spent_this_run: int = 0
    reserve: int = DEFAULT_RESERVE

    def observe(self, headers: dict[str, str]) -> None:
        """Record what the response said. An empty mapping is a cache hit."""
        if not headers:
            return
        for field, key in (("remaining", "x-requests-remaining"),
                           ("used", "x-requests-used")):
            raw = headers.get(key)
            if raw is None:
                continue
            try:
                setattr(self, field, int(float(raw)))
            except (TypeError, ValueError):
                log.debug("odds-api: unparseable %s header %r", key, raw)

    @property
    def exhausted(self) -> bool:
        """True only when the server has actually told us so.

        Unknown is not exhausted. Refusing to call because nothing has been
        observed yet would mean the first call of every month never happens.
        """
        return self.remaining is not None and self.remaining <= self.reserve

    def describe(self) -> dict:
        return {
            "remaining": self.remaining,
            "used": self.used,
            "spent_this_run": self.spent_this_run,
            "reserve": self.reserve,
            "exhausted": self.exhausted,
        }


def _url(path: str, **params: str) -> str:
    key = api_key()
    query = "&".join(f"{k}={v}" for k, v in {"apiKey": key, **params}.items())
    return f"{BASE_URL}/{path}?{query}"


def _get(
    url: str,
    session: PoliteSession,
    budget: Budget,
    *,
    max_age: float | None,
    costs_credits: bool,
    cost: int = 1,
) -> list | dict | None:
    """One request, with the budget observed from the response it returns."""
    if costs_credits and budget.exhausted:
        log.warning(
            "odds-api: %d credit(s) remaining, at or below the %d reserve — "
            "declining to spend. The free tier resets monthly, so an exhausted "
            "budget stays exhausted for the rest of the month.",
            budget.remaining, budget.reserve,
        )
        return None
    try:
        body, headers = session.get_with_headers(
            url, suffix=".json", max_age=max_age
        )
    except FetchError as exc:
        log.warning("odds-api: %s", exc)
        return None
    budget.observe(headers)
    if costs_credits and headers:
        # Billed per market per region, not per request. Counting requests said
        # 7 for a sweep the API charged 11 for — and a local counter that
        # undercounts is the one direction that empties a monthly budget.
        budget.spent_this_run += cost
    try:
        return json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning("odds-api: malformed payload: %s", exc)
        return None


def fetch_events(
    competition_id: str,
    *,
    session: PoliteSession | None = None,
    budget: Budget | None = None,
    max_age: float | None = EVENTS_MAX_AGE_SECONDS,
) -> list[dict] | None:
    """Upcoming fixtures for one competition. **Costs no credits.**

    This is the whole reason the key is worth having for cups: the fixture list
    is free, and only prices are metered.
    """
    if not configured():
        log.debug("odds-api: no key configured")
        return None
    sport = SPORT_KEYS.get(competition_id)
    if sport is None:
        return None
    payload = _get(
        _url(f"sports/{sport}/events"),
        session or PoliteSession(),
        budget or Budget(),
        max_age=max_age,
        costs_credits=False,
    )
    if payload is None:
        return None
    if not isinstance(payload, list):
        log.warning("odds-api: %s — expected a list of events", competition_id)
        return None
    return payload


def parse_events(payload: list[dict], competition_id: str) -> pd.DataFrame:
    """Flatten events into scheduled-fixture rows.

    Emitted with `stage_confirmed=False`: the API does not say which round a
    fixture belongs to, and `taxonomy.resolve_format` keys the tie format off
    exactly that. Guessing from the date would price a two-legged quarter-final
    as a single leg, so the fixture takes the competition default and says it
    is a default.
    """
    competition = taxonomy.get(competition_id)
    rows: list[dict] = []
    skipped = 0

    for event in payload:
        home = (event.get("home_team") or "").strip()
        away = (event.get("away_team") or "").strip()
        stamp = event.get("commence_time")
        if not (home and away and stamp):
            skipped += 1
            continue
        kickoff = pd.to_datetime(stamp, utc=True, errors="coerce")
        if pd.isna(kickoff):
            skipped += 1
            continue
        kickoff = kickoff.tz_convert(None)
        season = _season_label(kickoff)
        stage = "unknown"
        rows.append(
            {
                "fixture_id": (
                    f"{competition_id}|{season}|{home}|{away}"
                    if not competition.is_knockout
                    else f"{competition_id}|{season}|{stage}|{home}|{away}"
                ),
                "competition_id": competition_id,
                "season": season,
                "stage": stage,
                "stage_detail": None,
                # The competition default, because the round is unknown.
                "format": competition.resolve_format(season=season),
                "stage_confirmed": False,
                "neutral_venue": False,
                "date": kickoff.normalize(),
                "kickoff": kickoff.strftime("%H:%M"),
                "date_confirmed": True,
                "home_team": home,
                "away_team": away,
                "home_country": None,
                "away_country": None,
                "source": "odds_api",
                "odds_coverage": competition.odds_coverage,
                "api_event_id": event.get("id"),
            }
        )

    if skipped:
        log.warning(
            "odds-api: %s — dropped %d event(s) missing a club or a kickoff",
            competition_id, skipped,
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _season_label(kickoff: pd.Timestamp) -> str:
    """A season is named for the year it starts, and starts in July."""
    year = kickoff.year if kickoff.month >= 7 else kickoff.year - 1
    return f"{year}-{year + 1}"


def build_all_schedules(
    competitions: tuple[str, ...] = UNSOURCED_WITHOUT_A_KEY,
    *,
    session: PoliteSession | None = None,
    budget: Budget | None = None,
) -> pd.DataFrame:
    """Fixtures for the competitions nothing else can reach. Costs no credits.

    Defaults to the six with no keyless alternative rather than to all twelve:
    the five leagues come from openfootball and the DFB-Pokal from OpenLigaDB,
    both of which supply a round label that this source cannot.
    """
    if not configured():
        log.info(
            "odds-api: %s is not set, so %d cup competition(s) have no fixture "
            "source. Set it to cover %s.",
            ENV_KEY, len(competitions), ", ".join(competitions),
        )
        return pd.DataFrame()

    session = session or PoliteSession()
    budget = budget or Budget()
    frames = []
    for competition_id in competitions:
        payload = fetch_events(competition_id, session=session, budget=budget)
        if not payload:
            log.info("odds-api: %s — no upcoming events", competition_id)
            continue
        frame = parse_events(payload, competition_id)
        if not frame.empty:
            log.info(
                "odds-api: %s — %d scheduled fixture(s)", competition_id, len(frame)
            )
            frames.append(frame)

    log.info("odds-api: budget %s", budget.describe())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_odds(
    competition_id: str,
    *,
    markets: tuple[str, ...] = DAILY_MARKETS,
    regions: str = "eu",
    session: PoliteSession | None = None,
    budget: Budget | None = None,
) -> list[dict] | None:
    """Priced events for one competition. **Costs `len(markets)` credits.**

    The API bills per request per market per region, not per fixture, so twenty
    fixtures cost exactly what one does. That is what makes a daily sweep of
    every competition affordable on the free tier, and what makes adding a
    second market to every competition every day not affordable.
    """
    if not configured():
        return None
    sport = SPORT_KEYS.get(competition_id)
    if sport is None:
        return None
    payload = _get(
        _url(
            f"sports/{sport}/odds",
            regions=regions,
            markets=",".join(markets),
            oddsFormat="decimal",
        ),
        session or PoliteSession(),
        budget or Budget(),
        # Never cached: a price is the thing being captured, and serving a cached
        # body would append a stale quote under a fresh capture_id.
        max_age=0,
        costs_credits=True,
        cost=len(markets) * len(regions.split(",")),
    )
    if payload is None:
        return None
    if not isinstance(payload, list):
        log.warning("odds-api: %s - expected a list of priced events", competition_id)
        return None
    return payload


def _selection_of(outcome: dict, event: dict, market: str) -> tuple[str, float | None]:
    """Map one Odds API outcome onto (selection, line) in this project's names."""
    name = str(outcome.get("name") or "").strip()
    point = outcome.get("point")
    line = float(point) if point is not None else None
    if market == "h2h":
        if name == event.get("home_team"):
            return "home", None
        if name == event.get("away_team"):
            return "away", None
        if name.lower() == "draw":
            return "draw", None
        return "", None
    if market == "totals":
        lowered = name.lower()
        if lowered in ("over", "under"):
            return lowered, line
        return "", None
    if market == "spreads":
        if name == event.get("home_team"):
            return "ah_home", line
        if name == event.get("away_team"):
            return "ah_away", line
    return "", None


def parse_odds(
    payload: list[dict],
    competition_id: str,
    *,
    cid: str,
    captured_at: datetime | None = None,
) -> pd.DataFrame:
    """Priced events -> the same tidy shape `live_odds.parquet` already holds.

    One row per fixture x market x selection, with the bookmaker panel collapsed
    the way FR-16a requires: `odds_avg` is the mean across books and is what fair
    probability is derived from, `odds_max` is the best quote and is the price
    that would be taken. Named books keep their own columns so a reference can be
    selected on without being mistaken for the consensus.
    """
    from statpitch.data import football_data_live as live

    stamp = (captured_at or datetime.now(UTC)).astimezone(UTC)
    rows: list[dict] = []

    for event in payload:
        home = (event.get("home_team") or "").strip()
        away = (event.get("away_team") or "").strip()
        kickoff = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if not (home and away) or pd.isna(kickoff):
            continue
        kickoff = kickoff.tz_convert(None)

        quotes: dict[tuple[str, str, float | None], dict[str, float]] = {}
        for bookmaker in event.get("bookmakers") or []:
            book = str(bookmaker.get("key") or "")
            for block in bookmaker.get("markets") or []:
                market = str(block.get("key") or "")
                if market not in MARKET_NAMES:
                    continue
                for outcome in block.get("outcomes") or []:
                    selection, line = _selection_of(outcome, event, market)
                    price = outcome.get("price")
                    if not selection or price is None:
                        continue
                    try:
                        value = float(price)
                    except (TypeError, ValueError):
                        continue
                    if value <= 1.0:
                        continue
                    quotes.setdefault((market, selection, line), {})[book] = value

        for (market, selection, line), prices in quotes.items():
            values = list(prices.values())
            our_market = MARKET_NAMES[market]
            # The away side of a handicap is already quoted at its own line here,
            # so unlike football-data's single `AHh` there is nothing to negate.
            key = live.selection_key(our_market, selection, line)
            if key is None:
                continue
            row = {
                "capture_id": cid,
                "captured_at": stamp.isoformat(timespec="seconds"),
                "competition_id": competition_id,
                "div_code": None,
                "date": kickoff.normalize(),
                "kickoff_utc": kickoff,
                "fd_home": home,
                "fd_away": away,
                "snapshot": "preclose",
                "market": our_market,
                "selection": selection,
                "line": line,
                "selection_key": key,
                "odds_avg": float(np.mean(values)),
                "odds_max": float(max(values)),
                "odds_panel_avg": float(np.mean(values)),
                "odds_panel_max": float(max(values)),
                "n_panel_books": len(values),
                "n_books": len(values),
                "source": "odds_api",
            }
            for book, column in NAMED_BOOKS.items():
                row[column] = prices.get(book)
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in ("odds_pinnacle", "odds_b365", "odds_bfe"):
        if column not in frame.columns:
            frame[column] = None
    return frame.sort_values(
        ["date", "fd_home", "selection_key"]
    ).reset_index(drop=True)


def markets_for(
    competition_id: str, fixtures, *, now: datetime | None = None
) -> tuple[str, ...]:
    """Three markets on a matchday, one otherwise.

    A competition with a fixture kicking off today gets totals and handicaps as
    well as 1X2, because that is the day the extra markets are worth their
    credits. Every other day it gets 1X2 alone, which keeps a price series
    running for CLV at one credit per competition.
    """
    today = (now or datetime.now(UTC)).astimezone(UTC).date()
    if fixtures is None or fixtures.empty:
        return DAILY_MARKETS
    playing = fixtures[
        (fixtures["competition_id"] == competition_id)
        & (fixtures["date"].dt.date == today)
    ]
    return MATCHDAY_MARKETS if not playing.empty else DAILY_MARKETS


def next_kickoff(events: list[dict] | None) -> datetime | None:
    """Earliest `commence_time` in a free `/events` payload."""
    soonest: datetime | None = None
    for event in events or []:
        raw = event.get("commence_time")
        if not raw:
            continue
        try:
            kickoff = pd.Timestamp(raw).to_pydatetime()
        except (TypeError, ValueError):
            continue
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=UTC)
        kickoff = kickoff.astimezone(UTC)
        if soonest is None or kickoff < soonest:
            soonest = kickoff
    return soonest


def worth_a_credit(
    events: list[dict] | None,
    *,
    horizon_days: int = PRICING_HORIZON_DAYS,
    now: datetime | None = None,
) -> bool:
    """Whether this competition's next fixture is near enough to pay for.

    Decided from the FREE events feed, which is the point: the question "is
    anyone quoting this yet" is answerable at no cost, and asking it first is
    what keeps a daily sweep of twelve competitions inside a monthly budget.

    A competition whose next fixture is eleven days out is not skipped because
    it does not matter. It is skipped because the market does not exist yet, and
    a credit spent on it buys an empty board.
    """
    kickoff = next_kickoff(events)
    if kickoff is None:
        return False
    reference = (now or datetime.now(UTC)).astimezone(UTC)
    return (kickoff - reference) <= timedelta(days=horizon_days)


def describe() -> dict:
    """Capability report, for a job log or a health check."""
    return {
        "configured": configured(),
        "env_key": ENV_KEY,
        "monthly_credits": MONTHLY_CREDITS,
        "reserve": DEFAULT_RESERVE,
        "competitions_covered": sorted(SPORT_KEYS),
        "unsourced_without_a_key": list(UNSOURCED_WITHOUT_A_KEY),
        "events_cost_credits": False,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
