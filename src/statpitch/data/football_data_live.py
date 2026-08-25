"""Live pre-match prices from football-data.co.uk (Plan §4 Phase A).

The archive module beside this one answers "what did the market close at"; this
one answers "what can be had right now". They are the same publisher and very
nearly the same file — `fixtures.csv` carries the modern-era schema column for
column — which is the entire reason this source was chosen over a paid feed.

Why this source and not an odds API
===================================

MODEL_CARD §5 measures the project's only positive finding as *Friday-to-close
CLV*, and `clv_tracker` refuses to compute CLV across two sources. A price taken
from anywhere else cannot be compared to an `AvgC*` close without changing what
the number means. Taking the Friday snapshot from the publisher that also writes
the close keeps the measurement and the trade on one ruler.

It is also free, keyless and already parseable: `ODDS_MAPS[OddsEra.MODERN]` is
reused verbatim below rather than restated.

What it covers, and what it does not
====================================

About 180 fixtures across 20 divisions, of which five are in the taxonomy.

**One matchday block at a time, not a rolling week.** This was originally
described here as "roughly a week ahead", and that is wrong in a way that
matters. The feed publishes the next block of fixtures, holds it while they are
played, and only then rolls forward — so between blocks it contains nothing but
matches that have already kicked off. Observed on 2026-08-25: the feed still
listed 21-24 August, all played, while three fixtures were scheduled that day.

The consequence is that a card built midweek can be empty for a reason that has
nothing to do with the model: today's fixtures simply have no price yet.
`serving.app._why_the_card_is_empty` separates that from "nothing qualified",
because a consumer seeing an empty slate deserves to know which.

**Pinnacle is not in this file.** The archive carries `PSH/PSD/PSA`; the fixture
feed carries B365, BFD, BV, BW, PP, SKB and BFE. The sharp-reference rule that
produced +0.51% CLV was defined on *Pinnacle* edge, so it cannot be reproduced
here as measured. `odds_bfe` (Betfair Exchange) is carried as the candidate
replacement, and choosing between them is Phase C's job, not this module's — so
the column is recorded and nothing here selects on it.

**`Max` is best-of-N over the whole quoting period, not a live quote.** In this
week's file Everton were `AvgH` 2.23 against `MaxH` 3.20. Some of that spread is
a genuine outlier book and some is a price that has already gone. Recording it
faithfully is this module's job; deciding how much of it is takeable is the
grading layer's, and it is exactly the "fabricated edge" FR-16a exists to stop.

Append-only, because the baseline is the asset
==============================================

Every capture writes its own raw CSV under `data/raw/football_data_live/` and
appends to `live_odds.parquet` under a `capture_id`. Nothing is ever overwritten.
A CLV measurement is the difference between two prices for the same selection,
so overwriting Friday's snapshot with Saturday's does not update the data — it
destroys the only half of the measurement that cannot be recovered later.

Times are Europe/London, not UTC
================================

The `Time` column is UK local: Serie A's 20:45 CEST kick-offs appear as 19:45,
and LaLiga's 21:30 as 20:30. `fixtures.parquet` stores UTC. Reading one as the
other would shift every continental fixture by an hour in summer and silently
mistime any near-kickoff capture, so the conversion is explicit here.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from statpitch import paths, taxonomy
from statpitch.data import football_data as fd
from statpitch.data.http import PoliteSession

log = logging.getLogger(__name__)

LIVE_URL = "https://www.football-data.co.uk/fixtures.csv"

#: The `Time` column's zone. See the module docstring — this is not UTC.
SOURCE_TZ = "Europe/London"

#: Books quoting 1X2 in the fixture feed, which is *not* the archive's set.
#: Recorded for the record; `n_books` is counted by the shared archive helper so
#: the live and historical counts mean the same thing.
LIVE_BOOK_PREFIXES = ("B365", "BFD", "BV", "BW", "PP", "SKB", "BFE")

#: Betfair Exchange, carried alongside the archive's four price kinds as the
#: candidate sharp reference now that Pinnacle is absent (module docstring).
#:
#: Keyed by snapshot as well as selection, which is not decoration. Without the
#: `close` branch the closing block reads the *pre-close* `BFEH` column, and the
#: result is a row labelled `snapshot="close"` carrying a Friday price — a
#: fabricated close, and the one error that would corrupt CLV silently rather
#: than loudly, since both ends of the measurement would come from one capture.
_EXCHANGE_COLUMNS: dict[str, dict[str, dict[str, str]]] = {
    "1x2": {
        "preclose": {"home": "BFEH", "draw": "BFED", "away": "BFEA"},
        "close": {"home": "BFECH", "draw": "BFECD", "away": "BFECA"},
    },
    "ou": {
        "preclose": {"over": "BFE>2.5", "under": "BFE<2.5"},
        "close": {"over": "BFEC>2.5", "under": "BFEC<2.5"},
    },
    "ah": {
        "preclose": {"ah_home": "BFEAHH", "ah_away": "BFEAHA"},
        "close": {"ah_home": "BFECAHH", "ah_away": "BFECAHA"},
    },
}

#: Tokens that carry no identifying information across the two naming
#: conventions. Deliberately conservative: dropping a token that *does*
#: distinguish two clubs turns a rejected match into a wrong one.
_NOISE_TOKENS = frozenset({
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "us", "ud", "cd", "sd",
    "ca", "rc", "rcd", "sv", "vfb", "vfl", "tsg", "bsc", "de", "del", "club",
    "calcio", "balompie", "futbol", "united", "city", "town", "hotspur",
    "albion", "and", "the", "la", "el", "los",
})

#: football-data.co.uk short form -> openfootball formal name, per competition.
#:
#: Curated, not derived, and the reason is `Paris SG`. Its only distinctive token
#: is "paris", which it shares with `Paris FC` — a different club in the same
#: division. A plain best-match resolver scores `Paris SG -> Paris FC` at 1.00
#: and `Paris SG -> Paris Saint-Germain FC` at 0.50, i.e. it picks the wrong club
#: with maximum confidence. `resolve_clubs` rejects that pair rather than
#: resolving it, and the correct answer is written here by hand.
#:
#: Every entry below was verified against the names actually present in
#: `fixtures.parquet`. This is the same arrangement, for the same reason, as
#: `club_elo.OPENFOOTBALL_ALIASES`.
CLUB_ALIASES: dict[str, dict[str, str]] = {
    "ENG.PL": {
        "Man City": "Manchester City FC",
        "Man United": "Manchester United FC",
    },
    "ESP.LALIGA": {
        # "Ath" prefixes both Athletic and Atlético; token overlap puts
        # "Ath Madrid" nearer "Athletic Club" than "Club Atlético de Madrid".
        "Ath Bilbao": "Athletic Club",
        "Ath Madrid": "Club Atlético de Madrid",
        # "Barcelona" is a token of both Barcelona clubs.
        "Barcelona": "FC Barcelona",
        "Espanol": "RCD Espanyol de Barcelona",
        "Vallecano": "Rayo Vallecano de Madrid",
        "Alaves": "Deportivo Alavés",
    },
    "GER.BUNDESLIGA": {
        "Hamburg": "Hamburger SV",
        "M'gladbach": "Borussia Mönchengladbach",
    },
    "ITA.SERIEA": {
        "Inter": "FC Internazionale Milano",
    },
    "FRA.LIGUE1": {
        "Brest": "Stade Brestois 29",
        "Lyon": "Olympique Lyonnais",
        "Paris FC": "Paris FC",
        "Paris SG": "Paris Saint-Germain FC",
        "Rennes": "Stade Rennais FC 1901",
    },
}

#: Columns of the tidy live-odds artifact, in order.
TIDY_COLUMNS = (
    "capture_id", "captured_at", "competition_id", "div_code", "date",
    "kickoff_utc", "fd_home", "fd_away", "snapshot", "market", "selection",
    "line", "selection_key", "odds_avg", "odds_max", "odds_pinnacle",
    "odds_b365", "odds_bfe", "odds_panel_avg", "odds_panel_max",
    "n_panel_books", "n_books",
)

_PRICE_COLUMNS = (
    "odds_avg", "odds_max", "odds_pinnacle", "odds_b365", "odds_bfe",
    "odds_panel_avg", "odds_panel_max",
)


# --- capture ------------------------------------------------------------------

def capture_id(when: datetime | None = None) -> str:
    """Identity of one snapshot: the UTC minute it was taken.

    Minute resolution rather than second because two captures a minute apart are
    the same snapshot for every purpose here, and a filename-safe stamp is worth
    more than precision nobody consumes.
    """
    stamp = (when or datetime.now(UTC)).astimezone(UTC)
    return stamp.strftime("%Y%m%dT%H%MZ")


def raw_path(cid: str) -> Path:
    return paths.raw_dir() / "football_data_live" / f"fixtures_{cid}.csv"


def fetch(
    *,
    session: PoliteSession | None = None,
    when: datetime | None = None,
    url: str = LIVE_URL,
) -> tuple[Path, str]:
    """Download one snapshot verbatim. Returns (path, capture_id).

    `download_to` is used rather than `get_bytes` deliberately: the polite
    session's disk cache is keyed on URL, and this URL's body is *supposed* to
    change under it. A cached read would return Friday's prices on Saturday and
    look entirely healthy while doing it.
    """
    cid = capture_id(when)
    dest = raw_path(cid)
    if dest.exists():
        log.info("football-data live: %s already captured", cid)
        return dest, cid
    (session or PoliteSession()).download_to(url, dest, force=True)
    log.info("football-data live: captured %s -> %s", cid, dest)
    return dest, cid


# --- parsing ------------------------------------------------------------------

def selection_key(market: str, selection: str, line: float | None) -> str | None:
    """Map one football-data selection onto its `market_engine` key.

    `line` is the line this selection is quoted at, already negated for the away
    side by the caller — not football-data's home-side `AHh`. The formatting
    mirrors `market_engine.derive` character for character, so `-0.0` for the
    away side of a level handicap is reproduced rather than tidied: these two
    strings are joined on, and a prettier one would not match.
    """
    if market == "1x2":
        return f"1x2_{selection}"
    if market == "ou":
        if line is None or pd.isna(line):
            return None
        return f"{selection}_{float(line)}"
    if market == "ah":
        if line is None or pd.isna(line):
            return None
        return f"{selection}_{float(line)}"
    return None


def _kickoff_utc(dates: pd.Series, times: pd.Series) -> pd.Series:
    """Combine the date and UK-local time columns into a UTC timestamp.

    A missing or unparseable time yields NaT rather than midnight: an invented
    kick-off would mistime a near-kickoff capture, and "unknown" is a state every
    consumer here already handles.
    """
    text = times.astype(str).str.strip() if times is not None else None
    if text is None:
        return pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    clock = pd.to_datetime(text, format="%H:%M", errors="coerce")
    combined = dates + pd.to_timedelta(
        clock.dt.hour * 3600 + clock.dt.minute * 60, unit="s"
    )
    combined = combined.where(clock.notna())
    localised = combined.dt.tz_localize(
        SOURCE_TZ, ambiguous=True, nonexistent="shift_forward"
    )
    return localised.dt.tz_convert("UTC").dt.tz_localize(None)


def _competition_for_div(code: str) -> str | None:
    try:
        return taxonomy.registry().by_football_data_code(code).competition_id
    except taxonomy.TaxonomyError:
        return None


def parse(
    path: Path,
    *,
    cid: str | None = None,
    captured_at: datetime | None = None,
) -> pd.DataFrame:
    """One captured `fixtures.csv` -> tidy live-odds rows.

    Same long shape as `football_data.parse_odds` — one row per fixture x
    snapshot x market x line x selection — so de-vigging a market is a groupby
    here exactly as it is on the archive.
    """
    raw = fd._read_raw(path)
    if raw.empty:
        return pd.DataFrame(columns=list(TIDY_COLUMNS))

    if "Div" not in raw.columns:
        raise fd.IngestError(f"{path} has no Div column — not a fixtures feed")

    div = raw["Div"].astype(str).str.strip()
    competition = div.map(_competition_for_div)
    known = competition.notna()
    if not known.any():
        log.warning(
            "football-data live: no divisions in the taxonomy (saw %s)",
            sorted(set(div)),
        )
        return pd.DataFrame(columns=list(TIDY_COLUMNS))

    raw = raw[known].reset_index(drop=True)
    div = div[known].reset_index(drop=True)
    competition = competition[known].reset_index(drop=True)

    date = fd._parse_dates(raw["Date"])
    dated = date.notna()
    if not dated.all():
        log.warning(
            "football-data live: dropping %d row(s) with an unparseable date",
            int((~dated).sum()),
        )
    raw = raw[dated].reset_index(drop=True)
    div = div[dated].reset_index(drop=True)
    competition = competition[dated].reset_index(drop=True)
    date = date[dated].reset_index(drop=True)

    if raw.empty:
        return pd.DataFrame(columns=list(TIDY_COLUMNS))

    stamp = (captured_at or datetime.now(UTC)).astimezone(UTC)
    base = pd.DataFrame(
        {
            "capture_id": cid or capture_id(stamp),
            "captured_at": stamp.isoformat(timespec="seconds"),
            "competition_id": competition,
            "div_code": div,
            "date": date,
            "kickoff_utc": _kickoff_utc(date, raw.get("Time")),
            "fd_home": raw["HomeTeam"].map(fd.normalise_team),
            "fd_away": raw["AwayTeam"].map(fd.normalise_team),
        }
    )

    n_books = fd._count_quoting_books(raw)
    frames: list[pd.DataFrame] = []

    for market, snapshots in fd.ODDS_MAPS[fd.OddsEra.MODERN].items():
        for snapshot, selections in snapshots.items():
            line_col = selections.get("_line")
            line = fd._num(raw, line_col) if line_col else None
            for selection, price_cols in selections.items():
                if selection == "_line":
                    continue
                block = base.copy()
                block["snapshot"] = snapshot
                block["market"] = market
                block["selection"] = selection
                # `AHh` is the handicap given to the HOME side, so the away
                # selection is quoted at its negation. Carrying the home line on
                # both rows would leave `line` disagreeing with `selection_key`
                # and with `market_engine.Selection.line`, which is what any
                # later join on the pair would trip over.
                block["line"] = (
                    -line if line is not None and selection == "ah_away"
                    else line if line is not None
                    else pd.Series(
                        [2.5 if market == "ou" else pd.NA] * len(raw),
                        index=raw.index, dtype="Float64",
                    )
                )
                for kind in ("avg", "max", "pinnacle", "b365"):
                    block[f"odds_{kind}"] = fd._num(raw, price_cols.get(kind))
                block["odds_bfe"] = fd._num(
                    raw,
                    _EXCHANGE_COLUMNS.get(market, {}).get(snapshot, {}).get(selection),
                )

                if market == "1x2":
                    p_avg, p_max, p_n = fd._panel_prices(raw, snapshot, selection)
                    block["odds_panel_avg"] = p_avg
                    block["odds_panel_max"] = p_max
                    block["n_panel_books"] = p_n
                else:
                    block["odds_panel_avg"] = pd.NA
                    block["odds_panel_max"] = pd.NA
                    block["n_panel_books"] = pd.NA

                block["n_books"] = n_books
                block["selection_key"] = [
                    selection_key(market, selection, value) for value in block["line"]
                ]
                frames.append(block)

    odds = pd.concat(frames, ignore_index=True)

    # A price at or below evens on the stake is impossible, and a selection with
    # no price at all is not information. Both are dropped rather than carried
    # into the de-vig groupby as holes.
    for col in _PRICE_COLUMNS:
        odds.loc[odds[col] <= 1.0, col] = pd.NA
    odds = odds[odds[list(_PRICE_COLUMNS)].notna().any(axis=1)]
    odds = odds[odds["selection_key"].notna()]

    return odds[list(TIDY_COLUMNS)].reset_index(drop=True)


# --- club reconciliation ------------------------------------------------------

def _tokens(name: str) -> frozenset[str]:
    ascii_only = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in ascii_only)
    return frozenset(w for w in cleaned.split() if w and w not in _NOISE_TOKENS)


def _overlap(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    shared = left & right
    return len(shared) / min(len(left), len(right)) if shared else 0.0


def _unique_best(target: str, pool: list[str]) -> str | None:
    """The single best match in `pool`, or None when tied or unmatched.

    A tie is returned as no-match on purpose. Two clubs that score equally are
    two clubs the tokens cannot tell apart, and picking either is a coin flip
    that will be wrong for a whole season of fixtures.
    """
    target_tokens = _tokens(target)
    scored = sorted(
        ((_overlap(target_tokens, _tokens(candidate)), candidate) for candidate in pool),
        key=lambda pair: (-pair[0], pair[1]),
    )
    if not scored or scored[0][0] <= 0.0:
        return None
    if len(scored) > 1 and scored[1][0] >= scored[0][0]:
        return None
    return scored[0][1]


@dataclass(frozen=True, slots=True)
class ClubResolution:
    """Outcome of reconciling one competition's club names."""

    mapping: dict[str, str] = field(default_factory=dict)
    curated: dict[str, str] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        total = len(self.mapping) + len(self.unmatched)
        return len(self.mapping) / total if total else 0.0


def resolve_clubs(
    fd_names: list[str], of_names: list[str], competition_id: str
) -> ClubResolution:
    """Reconcile football-data short forms with openfootball formal names.

    Automatic matching is deliberately strict — a candidate must be the unique
    best match in *both* directions. That is what rejects `Paris SG -> Paris FC`,
    which a one-directional resolver accepts at full confidence. Whatever the
    strict rule refuses is answered from `CLUB_ALIASES` or reported unmatched;
    nothing is resolved by proximity alone.
    """
    curated = {
        name: target
        for name, target in CLUB_ALIASES.get(competition_id, {}).items()
        if name in set(fd_names)
    }

    mapping: dict[str, str] = {}
    for name in fd_names:
        if name in curated:
            continue
        candidate = _unique_best(name, of_names)
        if candidate is not None and _unique_best(candidate, fd_names) == name:
            mapping[name] = candidate

    # Curated entries win outright: they exist precisely where the automatic
    # rule was wrong, and re-deriving them would reintroduce the error.
    combined = {**mapping, **curated}
    unmatched = sorted(set(fd_names) - set(combined))
    return ClubResolution(mapping=combined, curated=curated, unmatched=unmatched)


def resolve_all(odds: pd.DataFrame, fixtures: pd.DataFrame) -> dict[str, ClubResolution]:
    """Run `resolve_clubs` per competition over a parsed live-odds frame."""
    out: dict[str, ClubResolution] = {}
    for competition_id, group in odds.groupby("competition_id"):
        fd_names = sorted(set(group["fd_home"]) | set(group["fd_away"]))
        listed = fixtures[fixtures["competition_id"] == competition_id]
        of_names = sorted(set(listed["home_team"]) | set(listed["away_team"]))
        if not of_names:
            out[str(competition_id)] = ClubResolution(unmatched=fd_names)
            continue
        out[str(competition_id)] = resolve_clubs(fd_names, of_names, str(competition_id))
    return out


def attach_fixture_ids(
    odds: pd.DataFrame, fixtures: pd.DataFrame, mapping: dict[str, dict[str, str]]
) -> tuple[pd.DataFrame, dict]:
    """Key live odds onto the fixture list's own `fixture_id`.

    The id is *joined*, never re-derived. `openfootball.fixture_id` deliberately
    excludes the date so a postponement keeps a fixture's identity; rebuilding
    that string here would duplicate the rule and drift from it. Joining on the
    club pair takes whatever the fixture list already decided.

    A priced fixture that is not in the list is dropped and counted, not
    invented. Two very different things cause that, and they are counted apart:

    `already_played`
        The feed prices a rolling window that includes matches which have
        already kicked off, while the fixture list holds only unplayed
        fixtures. On any matchday this is most of the difference — 20 of 38
        priced fixtures the first time this ran — and it is not a fault.

    `unlisted`
        A fixture inside the list's own horizon that still did not match: an
        unaliased club, or a schedule openfootball has not published. This is
        the number worth alarming on.

    Conflating the two makes the coverage floor fire every matchday, which is
    how a real gap ends up buried in a warning nobody trusts.
    """
    if odds.empty:
        return odds.assign(fixture_id=pd.Series(dtype="string")), {
            "priced": 0, "keyed": 0, "unlisted": 0, "already_played": 0,
            "unmapped_club": 0, "listable": 0,
        }

    work = odds.copy()
    per_competition = work["competition_id"].astype(str)
    work["of_home"] = [
        mapping.get(comp, {}).get(name)
        for comp, name in zip(per_competition, work["fd_home"], strict=True)
    ]
    work["of_away"] = [
        mapping.get(comp, {}).get(name)
        for comp, name in zip(per_competition, work["fd_away"], strict=True)
    ]

    mapped = work["of_home"].notna() & work["of_away"].notna()
    unmapped_club = int((~mapped).sum())
    work = work[mapped]

    listed = fixtures[
        ["competition_id", "home_team", "away_team", "fixture_id", "date"]
    ].rename(columns={"date": "listed_date"})
    joined = work.merge(
        listed,
        left_on=["competition_id", "of_home", "of_away"],
        right_on=["competition_id", "home_team", "away_team"],
        how="left",
    )
    keyed = joined["fixture_id"].notna()

    # The list contains only unplayed fixtures, so its earliest row IS its lower
    # horizon: anything priced before that cannot be matched by construction, no
    # matter how good the club map is. Derived from the list rather than from a
    # clock so the split is reproducible from the two artifacts alone — a
    # re-run tomorrow classifies today's capture the same way.
    horizon = fixtures["date"].min() if not fixtures.empty else None
    before_horizon = (
        joined["date"] < horizon if horizon is not None
        else pd.Series(False, index=joined.index)
    )
    already_played = int((~keyed & before_horizon).sum())
    unlisted = int((~keyed & ~before_horizon).sum())
    joined = joined[keyed]

    # The odds date is confirmed by the bookmakers; the fixture list's may still
    # be openfootball's nominal matchday. The gap is reported so the collector
    # can correct the list, and is never used to reject the pairing.
    joined["date_shift_days"] = (
        (joined["date"] - joined["listed_date"]).dt.days.astype("Int64")
    )

    stats = {
        "priced": int(len(odds)),
        "keyed": int(len(joined)),
        "unlisted": unlisted,
        "already_played": already_played,
        "unmapped_club": unmapped_club,
        # The denominator the coverage floor is meaningful against.
        "listable": int(len(odds)) - already_played,
    }
    columns = [*TIDY_COLUMNS, "fixture_id", "of_home", "of_away", "date_shift_days"]
    return joined[columns].reset_index(drop=True), stats


def append_snapshot(frame: pd.DataFrame, path: Path | None = None) -> tuple[Path, int]:
    """Append one capture to the live-odds artifact, never overwriting a prior one.

    Re-running the same capture is a no-op rather than a duplicate: rows carrying
    a `capture_id` already present are dropped before the write. That makes the
    collector safe to re-run after a partial failure, which an append-only file
    otherwise makes unrecoverable.
    """
    destination = path or paths.live_odds_file()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        existing = pd.read_parquet(destination)
        seen = set(existing["capture_id"].unique())
        incoming = frame[~frame["capture_id"].isin(seen)]
        if incoming.empty:
            log.info(
                "live odds: capture %s already recorded — nothing appended",
                sorted(set(frame["capture_id"])),
            )
            return destination, 0
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        incoming = frame
        combined = frame

    combined.to_parquet(destination, index=False)
    return destination, int(len(incoming))
