"""football-data.co.uk ingestion (Design §3, §3.1 — Phase 1).

Free, no key, no signup; 25 seasons of results and odds for the five in-scope
leagues. This module downloads the CSVs, normalises them into two tidy tables,
and — most importantly — is honest about what odds actually exist when.

Three odds schema eras, verified against the live files rather than assumed
=========================================================================

Requirements §7.3 describes the `C`-suffixed closing columns as though they run
the length of the archive. They do not. The real picture:

    era       seasons          consensus pre-close  consensus close  AH     kickoff time
    legacy    ...-2004/05      none                 none             none   no
    betbrain  2005/06-2018/19  BbAv* / BbMx*        none             BbAHh  no
    modern    2019/20-         Avg* / Max*          AvgC* / MaxC*    AHh    yes

Consequences that propagate through the whole project, and which are better known
now than discovered in Phase 5:

* **CLV, de-vig selection and the entire Decision Layer need consensus *closing*
  odds, so they are confined to 2019/20 onward.** That is 6 completed seasons
  through 2024/25 — comfortably past the "≥2 full seasons" bar in Requirements
  §8.3, but a quarter of the archive, not all of it.
* Pinnacle's own closing price (`PSC*`) does reach back to 2012/13. It is a
  single book rather than a consensus, but Requirements §7.3 licenses Pinnacle as
  a sharp benchmark *before* 2025/26 — which is exactly the window it covers. It
  gives calibration and RPS work a 13-season runway.
* The Pinnacle regime break (23/07/2025) sits inside the modern era, and pooling
  across it is forbidden by default. The clean consensus-closing backtest window
  is therefore **2019/20-2024/25 pre-break**, with 2025/26 held separately.
* `Time` (kickoff) also starts in 2019/20, so FR-34's edge-decay-by-hours-to-
  kickoff analysis shares the same window. No extra loss.

Everything before 2019/20 remains fully usable for *training* the model, which is
what the long archive is for. It is the market benchmark that is window-limited.
"""

from __future__ import annotations

import csv
import logging
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from statpitch import paths, taxonomy
from statpitch.data.http import FetchError, PoliteSession

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"

#: Requirements §7.3 — Pinnacle dropped from Max/Avg from this date.
PINNACLE_BREAK_DATE = pd.Timestamp("2025-07-23")

#: First season whose files carry consensus closing odds and kickoff times.
FIRST_MODERN_SEASON = 2019
#: First season carrying BetBrain consensus aggregates.
FIRST_BETBRAIN_SEASON = 2005
#: First season carrying Pinnacle closing prices.
FIRST_PINNACLE_CLOSING_SEASON = 2012


class OddsEra(StrEnum):
    LEGACY = "legacy"
    BETBRAIN = "betbrain"
    MODERN = "modern"


def era_for_season(start_year: int) -> OddsEra:
    if start_year >= FIRST_MODERN_SEASON:
        return OddsEra.MODERN
    if start_year >= FIRST_BETBRAIN_SEASON:
        return OddsEra.BETBRAIN
    return OddsEra.LEGACY


def season_code(start_year: int) -> str:
    """2019 -> '1920', the code football-data.co.uk uses in its URLs."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2019 -> '2019-2020', the canonical season string used across the project."""
    return f"{start_year}-{start_year + 1}"


def csv_url(start_year: int, div_code: str) -> str:
    return f"{BASE_URL}/{season_code(start_year)}/{div_code}.csv"


# --- odds column maps ---------------------------------------------------------
# Keyed (era, market, snapshot) -> {selection: {price_kind: column}}.
# `price_kind` separates the two market numbers FR-16a insists must never be
# conflated: `avg` is the consensus used for fair probability, `max` is the price
# actually obtainable. They are carried in different columns all the way through.

_MODERN_1X2 = {
    "preclose": {
        "home": {"avg": "AvgH", "max": "MaxH", "pinnacle": "PSH", "b365": "B365H"},
        "draw": {"avg": "AvgD", "max": "MaxD", "pinnacle": "PSD", "b365": "B365D"},
        "away": {"avg": "AvgA", "max": "MaxA", "pinnacle": "PSA", "b365": "B365A"},
    },
    "close": {
        "home": {"avg": "AvgCH", "max": "MaxCH", "pinnacle": "PSCH", "b365": "B365CH"},
        "draw": {"avg": "AvgCD", "max": "MaxCD", "pinnacle": "PSCD", "b365": "B365CD"},
        "away": {"avg": "AvgCA", "max": "MaxCA", "pinnacle": "PSCA", "b365": "B365CA"},
    },
}

_MODERN_OU = {
    "preclose": {
        "over": {"avg": "Avg>2.5", "max": "Max>2.5", "pinnacle": "P>2.5", "b365": "B365>2.5"},
        "under": {"avg": "Avg<2.5", "max": "Max<2.5", "pinnacle": "P<2.5", "b365": "B365<2.5"},
    },
    "close": {
        "over": {
            "avg": "AvgC>2.5", "max": "MaxC>2.5", "pinnacle": "PC>2.5", "b365": "B365C>2.5",
        },
        "under": {
            "avg": "AvgC<2.5", "max": "MaxC<2.5", "pinnacle": "PC<2.5", "b365": "B365C<2.5",
        },
    },
}

_MODERN_AH = {
    "preclose": {
        "_line": "AHh",
        "ah_home": {"avg": "AvgAHH", "max": "MaxAHH", "pinnacle": "PAHH", "b365": "B365AHH"},
        "ah_away": {"avg": "AvgAHA", "max": "MaxAHA", "pinnacle": "PAHA", "b365": "B365AHA"},
    },
    "close": {
        "_line": "AHCh",
        "ah_home": {"avg": "AvgCAHH", "max": "MaxCAHH", "pinnacle": "PCAHH", "b365": "B365CAHH"},
        "ah_away": {"avg": "AvgCAHA", "max": "MaxCAHA", "pinnacle": "PCAHA", "b365": "B365CAHA"},
    },
}

_BETBRAIN_1X2 = {
    "preclose": {
        "home": {"avg": "BbAvH", "max": "BbMxH", "pinnacle": "PSH", "b365": "B365H"},
        "draw": {"avg": "BbAvD", "max": "BbMxD", "pinnacle": "PSD", "b365": "B365D"},
        "away": {"avg": "BbAvA", "max": "BbMxA", "pinnacle": "PSA", "b365": "B365A"},
    },
    # No consensus closing existed in this era. Pinnacle's closing price does from
    # 2012/13, so the close snapshot is emitted with `avg`/`max` empty rather than
    # omitted — a single-book benchmark, and typed as such.
    "close": {
        "home": {"avg": None, "max": None, "pinnacle": "PSCH", "b365": "B365CH"},
        "draw": {"avg": None, "max": None, "pinnacle": "PSCD", "b365": "B365CD"},
        "away": {"avg": None, "max": None, "pinnacle": "PSCA", "b365": "B365CA"},
    },
}

_BETBRAIN_OU = {
    "preclose": {
        "over": {"avg": "BbAv>2.5", "max": "BbMx>2.5", "pinnacle": None, "b365": "B365>2.5"},
        "under": {"avg": "BbAv<2.5", "max": "BbMx<2.5", "pinnacle": None, "b365": "B365<2.5"},
    },
}

_BETBRAIN_AH = {
    "preclose": {
        "_line": "BbAHh",
        "ah_home": {"avg": "BbAvAHH", "max": "BbMxAHH", "pinnacle": None, "b365": "B365AHH"},
        "ah_away": {"avg": "BbAvAHA", "max": "BbMxAHA", "pinnacle": None, "b365": "B365AHA"},
    },
}

_LEGACY_1X2 = {
    # Individual books only — no consensus aggregate was published. `avg`/`max`
    # stay empty rather than being synthesised into them, because an average over
    # the five books of 2000/01 does not mean the same thing as the ~30-book Avg
    # column of 2024/25, and silently equating them would corrupt calibration.
    # The panel columns below carry the reconstruction instead, under a name that
    # cannot be mistaken for the published consensus.
    "preclose": {
        "home": {"avg": None, "max": None, "pinnacle": None, "b365": "B365H"},
        "draw": {"avg": None, "max": None, "pinnacle": None, "b365": "B365D"},
        "away": {"avg": None, "max": None, "pinnacle": None, "b365": "B365A"},
    },
}

#: Per-book column prefixes by snapshot. Closing per-book columns insert a `C`
#: after the prefix (`B365H` -> `B365CH`), which is what makes the panel
#: reconstruction below snapshot-aware.
_BOOK_SUFFIX_BY_SNAPSHOT = {"preclose": "", "close": "C"}
_SELECTION_SUFFIX_1X2 = {"home": "H", "draw": "D", "away": "A"}

ODDS_MAPS: dict[OddsEra, dict[str, dict]] = {
    OddsEra.MODERN: {"1x2": _MODERN_1X2, "ou": _MODERN_OU, "ah": _MODERN_AH},
    OddsEra.BETBRAIN: {"1x2": _BETBRAIN_1X2, "ou": _BETBRAIN_OU, "ah": _BETBRAIN_AH},
    OddsEra.LEGACY: {"1x2": _LEGACY_1X2},
}

#: Per-book 1X2 home-odds columns, used to count how many books quoted a match.
#: Design §6.4's `c_market` sub-score increases in the number of quoting books;
#: the BetBrain era published that count directly (`Bb1X2`), the modern era does
#: not, so it is reconstructed by counting non-empty book columns.
_BOOK_PREFIXES_1X2 = (
    "B365", "BW", "BF", "PS", "WH", "1XB", "BFE", "VC", "IW", "LB", "SB", "SJ", "BS", "GB",
)

#: Match-level columns carried through to the canonical schema.
_STAT_COLUMNS = {
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_shots_target", "AST": "away_shots_target",
    "HF": "home_fouls", "AF": "away_fouls",
    "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellows", "AY": "away_yellows",
    "HR": "home_reds", "AR": "away_reds",
}


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SeasonFile:
    competition_id: str
    div_code: str
    start_year: int
    path: Path

    @property
    def season(self) -> str:
        return season_label(self.start_year)

    @property
    def era(self) -> OddsEra:
        return era_for_season(self.start_year)


# --- download -----------------------------------------------------------------

def download_season(
    div_code: str,
    start_year: int,
    *,
    session: PoliteSession | None = None,
    force: bool = False,
    dest_root: Path | None = None,
) -> Path | None:
    """Fetch one division-season CSV into `data/raw/football_data/`.

    Returns None when the file does not exist upstream (a season that has not
    started, or a division that did not run), which is normal and not an error.
    """
    session = session or PoliteSession()
    root = dest_root or (paths.raw_dir() / "football_data")
    dest = root / season_code(start_year) / f"{div_code}.csv"
    try:
        return session.download_to(csv_url(start_year, div_code), dest, force=force)
    except FetchError as exc:
        if "404" in str(exc):
            log.info("football-data: no file for %s %s", div_code, season_label(start_year))
            return None
        raise


def download_all(
    *,
    first_season: int = 1993,
    last_season: int | None = None,
    competitions: list[str] | None = None,
    session: PoliteSession | None = None,
    force: bool = False,
) -> list[SeasonFile]:
    """Download every in-scope league-season available.

    Only competitions with a football-data division code are fetched — the cups
    and continental competitions have no odds coverage and come from openfootball.
    """
    reg = taxonomy.registry()
    if last_season is None:
        # Season N runs into calendar year N+1, so today's season started last year
        # if we are before July.
        today = pd.Timestamp.today()
        last_season = today.year if today.month >= 7 else today.year - 1

    targets = [
        c for c in reg
        if c.football_data_code and (competitions is None or c.competition_id in competitions)
    ]
    if not targets:
        raise IngestError(f"no competitions matched {competitions!r}")

    session = session or PoliteSession()
    out: list[SeasonFile] = []
    for comp in targets:
        for year in range(first_season, last_season + 1):
            path = download_season(
                comp.football_data_code, year, session=session, force=force
            )
            if path is not None:
                out.append(
                    SeasonFile(comp.competition_id, comp.football_data_code, year, path)
                )
    log.info("football-data: %d season files available", len(out))
    return out


# --- parse --------------------------------------------------------------------

def _read_ragged(path: Path, encoding: str) -> pd.DataFrame | None:
    """Re-read a CSV whose rows carry more fields than its header.

    Extra fields are truncated, but only after checking they are empty. If a row
    would lose real data the row is dropped and logged loudly — silently trimming
    a populated column is how a whole odds series goes missing without anyone
    noticing.

    Both pandas engines truncate overflow fields silently once `index_col=False`
    is set, and `on_bad_lines` is never invoked, so the row-width decision is made
    here with the csv module rather than delegated to the parser.
    """
    try:
        with path.open(encoding=encoding, newline="") as fh:
            rows = list(csv.reader(fh))
    except (UnicodeDecodeError, csv.Error):
        return None

    if not rows:
        return None

    header = rows[0]
    width = len(header)
    kept: list[list[str]] = []
    trimmed = 0
    dropped = 0

    for row in rows[1:]:
        if not any(str(v).strip() for v in row):
            continue  # padding line
        if len(row) > width:
            if any(str(v).strip() for v in row[width:]):
                dropped += 1
                continue
            row = row[:width]
            trimmed += 1
        elif len(row) < width:
            row = row + [""] * (width - len(row))
        kept.append(row)

    if dropped:
        log.warning(
            "football-data: %s — dropped %d row(s) carrying data beyond the header width; "
            "the header may be wrong for this file",
            path.name, dropped,
        )
    if trimmed:
        log.info(
            "football-data: %s — trimmed empty overflow fields on %d row(s)",
            path.name, trimmed,
        )

    return pd.DataFrame(kept, columns=header, dtype=str).replace("", pd.NA)


def _read_raw(path: Path) -> pd.DataFrame:
    """Read a football-data CSV defensively.

    The archive has real-world defects: latin-1 accents in Spanish and French team
    names, trailing all-empty rows, and stray unnamed columns from trailing commas.
    Everything is read as text and converted explicitly afterwards, so a stray
    value can never silently coerce a whole odds column to NaN.
    """
    last_exc: Exception | None = None
    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            # index_col=False is essential, not cosmetic: many of these files end
            # their header with trailing commas, and pandas responds by silently
            # promoting the first columns into a MultiIndex, which turns Div/Date/
            # HomeTeam into index levels and empties the frame of real data.
            #
            # It has a second effect worth being deliberate about: it downgrades a
            # ragged row from a ParserError to a ParserWarning and truncates the
            # overflow without asking. That is the right outcome when the overflow
            # is trailing commas and the wrong one when it is data, so the warning
            # is captured and routed to the checked path rather than ignored.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", pd.errors.ParserWarning)
                df = pd.read_csv(
                    path, encoding=encoding, dtype=str, keep_default_na=True,
                    index_col=False,
                )
            if any(issubclass(w.category, pd.errors.ParserWarning) for w in caught):
                checked = _read_ragged(path, encoding)
                if checked is not None:
                    df = checked
            break
        except UnicodeDecodeError as exc:
            last_exc = exc
        except pd.errors.ParserError as exc:
            # Ragged rows the C parser cannot recover from at all. Nine files in
            # the 2002/03-2004/05 range are like this — invariably trailing empty
            # commas — and losing them would cost ~3,000 matches.
            last_exc = exc
            df = _read_ragged(path, encoding)
            if df is not None:
                break
    if df is None:
        raise IngestError(f"could not read {path}") from last_exc

    df = df.loc[:, [c for c in df.columns if c and not str(c).startswith("Unnamed")]]
    # A row without a date or a home team is padding, not a match.
    if "Date" in df.columns:
        df = df[df["Date"].notna() & (df["Date"].astype(str).str.strip() != "")]
    if "HomeTeam" in df.columns:
        df = df[df["HomeTeam"].notna() & (df["HomeTeam"].astype(str).str.strip() != "")]
    return df.reset_index(drop=True)


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse dd/mm/yy and dd/mm/yyyy, which both appear across the archive."""
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    short = parsed.isna()
    if short.any():
        parsed = parsed.fillna(
            pd.to_datetime(text.where(short), format="%d/%m/%y", errors="coerce")
        )
    return parsed


def _num(df: pd.DataFrame, col: str | None) -> pd.Series:
    """Numeric view of a column, or an all-NaN column when absent from this era."""
    if col is None or col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    return pd.to_numeric(df[col], errors="coerce").astype("Float64")


def _int(df: pd.DataFrame, col: str) -> pd.Series:
    """Nullable-integer view of a column.

    Columns genuinely vary by era — the 2015/16 files carry no shot counts and the
    1990s files no half-time scores — so an absent column must yield an empty
    typed column, not an error.
    """
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    return pd.to_numeric(df[col], errors="coerce").astype("Int64")


def _text(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    return df[col].astype("string").str.strip()


def normalise_team(name: str) -> str:
    """Stable team key within a source.

    Cross-source reconciliation (football-data vs openfootball vs Club Elo) is a
    separate mapping table; this only guarantees that whitespace and casing do not
    fork one club into two.
    """
    return " ".join(str(name).split()).strip()


def _match_ids(df: pd.DataFrame, competition_id: str) -> pd.Series:
    """Deterministic match id, stable across re-ingestion.

    Date plus both clubs is unique within a league season. FA Cup replays would
    collide, which is why the cup path (openfootball) adds a stage component.
    """
    return (
        competition_id
        + "|" + df["date"].dt.strftime("%Y-%m-%d")
        + "|" + df["home_team"].str.replace(r"\s+", "", regex=True)
        + "|" + df["away_team"].str.replace(r"\s+", "", regex=True)
    )


def parse_matches(sf: SeasonFile) -> pd.DataFrame:
    """One division-season CSV -> canonical match rows."""
    raw = _read_raw(sf.path)
    if raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=raw.index)
    out["competition_id"] = sf.competition_id
    out["season"] = sf.season
    out["season_start_year"] = sf.start_year
    out["div_code"] = sf.div_code
    out["date"] = _parse_dates(raw["Date"])
    out["kickoff_local"] = _text(raw, "Time")
    out["home_team"] = raw["HomeTeam"].map(normalise_team)
    out["away_team"] = raw["AwayTeam"].map(normalise_team)

    for src, dst in (("FTHG", "home_goals"), ("FTAG", "away_goals"),
                     ("HTHG", "home_goals_ht"), ("HTAG", "away_goals_ht")):
        out[dst] = _int(raw, src)

    out["result"] = _text(raw, "FTR")
    out["result_ht"] = _text(raw, "HTR")
    out["referee"] = _text(raw, "Referee")

    for src, dst in _STAT_COLUMNS.items():
        out[dst] = _int(raw, src)

    # Every league match is a round-robin fixture at the home side's ground; the
    # cup formats arrive via openfootball with real stage information.
    comp = taxonomy.get(sf.competition_id)
    out["stage"] = pd.NA
    out["format"] = comp.resolve_format(season=sf.season)
    out["leg_number"] = pd.NA
    out["neutral_venue"] = False
    out["odds_schema_era"] = str(sf.era)
    out["odds_regime"] = _odds_regime(out["date"])
    out["source"] = "football-data.co.uk"

    # A match with no date or no result is not usable downstream.
    out = out[out["date"].notna()].copy()
    dropped = out["home_goals"].isna() | out["away_goals"].isna()
    if dropped.any():
        log.info(
            "football-data: %s %s — %d rows without a full-time score (unplayed/void)",
            sf.div_code, sf.season, int(dropped.sum()),
        )
        out = out[~dropped].copy()
    if out.empty:
        return out

    out["match_id"] = _match_ids(out, sf.competition_id)
    out = out.drop_duplicates(subset="match_id", keep="first")
    return out.reset_index(drop=True)


def _odds_regime(dates: pd.Series) -> pd.Series:
    """Requirements §7.3 — pre/post the 23/07/2025 Pinnacle break."""
    return pd.Series(
        [
            "post_2025_07_23" if pd.notna(d) and d >= PINNACLE_BREAK_DATE else "pre_2025_07_23"
            for d in dates
        ],
        index=dates.index,
        dtype="string",
    )


def _count_quoting_books(raw: pd.DataFrame) -> pd.Series:
    """Number of books with a 1X2 home price, for the `c_market` grading sub-score."""
    if "Bb1X2" in raw.columns:
        return pd.to_numeric(raw["Bb1X2"], errors="coerce").astype("Int64")
    cols = [f"{p}H" for p in _BOOK_PREFIXES_1X2 if f"{p}H" in raw.columns]
    if not cols:
        return pd.Series(pd.NA, index=raw.index, dtype="Int64")
    present = raw[cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).notna()
    return present.sum(axis=1).astype("Int64")


def _panel_prices(
    raw: pd.DataFrame, snapshot: str, selection: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Mean, max and count over the individual-book 1X2 panel.

    Why this exists alongside the published `Avg*`/`Max*` columns: the consensus
    columns only start in 2005/06, but individual books run the length of the
    archive. Reconstructing a panel consensus gives one price series covering all
    25 seasons, which extends the market benchmark (FR-13, FR-14) back past the
    window the Decision Layer can reach.

    It is deliberately kept in *separate columns* from `odds_avg`/`odds_max`. A
    five-book panel and a thirty-book consensus are different estimators, and the
    eras where both exist are exactly what lets Phase 5 measure that difference
    instead of assuming it away.
    """
    infix = _BOOK_SUFFIX_BY_SNAPSHOT[snapshot]
    tail = _SELECTION_SUFFIX_1X2[selection]
    cols = [
        f"{p}{infix}{tail}" for p in _BOOK_PREFIXES_1X2 if f"{p}{infix}{tail}" in raw.columns
    ]
    if not cols:
        empty = pd.Series([pd.NA] * len(raw), index=raw.index, dtype="Float64")
        return empty, empty, pd.Series([pd.NA] * len(raw), index=raw.index, dtype="Int64")

    panel = raw[cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    panel = panel.where(panel > 1.0)  # a price never returns less than the stake
    return (
        panel.mean(axis=1).astype("Float64"),
        panel.max(axis=1).astype("Float64"),
        panel.notna().sum(axis=1).astype("Int64"),
    )


def parse_odds(sf: SeasonFile, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """One division-season CSV -> tidy odds rows.

    Long rather than wide, one row per match x snapshot x market x line x
    selection. De-vigging (FR-28) needs the complete selection set for a market to
    sum its implied probabilities, and a tidy table makes that a groupby instead of
    a hand-maintained list of 120 column names.
    """
    raw = _read_raw(sf.path)
    if raw.empty:
        return pd.DataFrame()

    matches = parse_matches(sf) if matches is None else matches
    if matches.empty:
        return pd.DataFrame()

    # parse_matches drops rows; re-derive the key on the raw frame and join on it
    # so odds land on the right match rather than on a positional guess.
    keyed = pd.DataFrame(index=raw.index)
    keyed["date"] = _parse_dates(raw["Date"])
    keyed["home_team"] = raw["HomeTeam"].map(normalise_team)
    keyed["away_team"] = raw["AwayTeam"].map(normalise_team)
    valid = keyed["date"].notna()
    keyed = keyed[valid]
    raw = raw[valid]
    keyed["match_id"] = _match_ids(keyed, sf.competition_id)

    known = set(matches["match_id"])
    n_books = _count_quoting_books(raw)

    frames: list[pd.DataFrame] = []
    for market, snapshots in ODDS_MAPS[sf.era].items():
        for snapshot, selections in snapshots.items():
            line_col = selections.get("_line")
            line = _num(raw, line_col) if line_col else None
            for selection, price_cols in selections.items():
                if selection == "_line":
                    continue
                block = pd.DataFrame(index=raw.index)
                block["match_id"] = keyed["match_id"]
                block["snapshot"] = snapshot
                block["market"] = market
                block["selection"] = selection
                block["line"] = (
                    line if line is not None
                    else (2.5 if market == "ou" else pd.NA)
                )
                for kind in ("avg", "max", "pinnacle", "b365"):
                    block[f"odds_{kind}"] = _num(raw, price_cols.get(kind))

                if market == "1x2":
                    p_avg, p_max, p_n = _panel_prices(raw, snapshot, selection)
                    block["odds_panel_avg"] = p_avg
                    block["odds_panel_max"] = p_max
                    block["n_panel_books"] = p_n
                else:
                    # Only 1X2 has a per-book panel spanning the whole archive;
                    # O/U and AH per-book columns start with the aggregates anyway.
                    block["odds_panel_avg"] = pd.NA
                    block["odds_panel_max"] = pd.NA
                    block["n_panel_books"] = pd.NA

                block["n_books"] = n_books
                frames.append(block)

    if not frames:
        return pd.DataFrame()

    odds = pd.concat(frames, ignore_index=True)

    # A selection with no price at all is not information; drop it rather than
    # carrying empty rows into the de-vig groupby.
    price_cols = [
        "odds_avg", "odds_max", "odds_pinnacle", "odds_b365",
        "odds_panel_avg", "odds_panel_max",
    ]
    odds = odds[odds[price_cols].notna().any(axis=1)]
    odds = odds[odds["match_id"].isin(known)]
    if odds.empty:
        return odds

    meta = matches[["match_id", "competition_id", "season", "date", "odds_regime"]]
    odds = odds.merge(meta, on="match_id", how="left")
    odds["odds_schema_era"] = str(sf.era)

    # Odds below 1.0 are impossible (a price never returns less than the stake) and
    # appear in the archive as data-entry noise.
    for col in price_cols:
        odds.loc[odds[col] <= 1.0, col] = pd.NA
    odds = odds[odds[price_cols].notna().any(axis=1)]

    return odds.reset_index(drop=True)


def build(
    season_files: list[SeasonFile],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse many season files into (matches, odds)."""
    match_frames, odds_frames = [], []
    for sf in season_files:
        try:
            m = parse_matches(sf)
        except Exception:
            log.exception("football-data: failed to parse %s %s", sf.div_code, sf.season)
            continue
        if m.empty:
            continue
        match_frames.append(m)
        try:
            o = parse_odds(sf, matches=m)
        except Exception:
            log.exception("football-data: failed to parse odds for %s %s", sf.div_code, sf.season)
            continue
        if not o.empty:
            odds_frames.append(o)

    matches = (
        pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    )
    odds = pd.concat(odds_frames, ignore_index=True) if odds_frames else pd.DataFrame()

    if not matches.empty:
        matches = matches.sort_values(["date", "competition_id", "home_team"])
        matches = matches.reset_index(drop=True)
    return matches, odds


def has_consensus_closing(start_year: int) -> bool:
    """Whether a season carries `AvgC*`/`MaxC*` — i.e. is usable by the Decision Layer."""
    return era_for_season(start_year) is OddsEra.MODERN


def decision_layer_seasons(
    first: int = FIRST_MODERN_SEASON, last: int | None = None
) -> list[int]:
    """Season start years the Decision Layer may operate on (consensus closing odds)."""
    if last is None:
        today = pd.Timestamp.today()
        last = today.year if today.month >= 7 else today.year - 1
    return [y for y in range(max(first, FIRST_MODERN_SEASON), last + 1)]
