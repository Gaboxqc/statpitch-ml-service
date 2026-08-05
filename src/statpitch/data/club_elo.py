"""Club Elo ingestion (Design §3, FR-9, FR-11).

clubelo.com publishes a free, keyless CSV API covering European club football
since the 1940s — crucially including the *full pyramid*, not just top flights.
That is what makes the FR-9 lower-division prior possible: when a Segunda
División club turns up in the Copa del Rey, Club Elo already has a rating for it.

Two endpoints are used:

* ``api.clubelo.com/YYYY-MM-DD`` — every club's rating on one date, with a
  ``Level`` column giving the club's tier. Used to build the roster of canonical
  Club Elo names and to source `tier` (Design §2 leaves tier null on cups because
  it is a property of the entrant club; this is where that value comes from).
* ``api.clubelo.com/<ClubName>`` — one club's full history as ``From``/``To``
  date intervals. Used for as-of-date lookups, which is what feature building
  needs: the rating a club held *the day before* a match, never after it.

Name reconciliation
===================

football-data.co.uk and Club Elo name clubs differently, and the tasks document
flags this as a known pain point. It is handled in three ordered stages:

1. accent- and punctuation-insensitive normalisation, which resolves most of it;
2. a curated alias table for names that genuinely differ ("Ath Madrid" ->
   "Atletico", "Nott'm Forest" -> "Forest");
3. anything still unresolved is *reported*, never guessed.

Stage 3 matters more than it looks. Fuzzy matching "Ath Madrid" against the
roster proposes "Real Madrid" as its closest candidate — a wrong mapping that
would silently attach the wrong club's strength rating to every Atlético fixture
and be nearly impossible to spot downstream. So automatic fuzzy matching is used
only to *suggest* aliases to a human, and never to accept one.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date as Date
from io import StringIO
from pathlib import Path

import pandas as pd

from statpitch import paths
from statpitch.data.http import FetchError, PoliteSession

log = logging.getLogger(__name__)

BASE_URL = "http://api.clubelo.com"

#: Country codes for the five in-scope leagues, as Club Elo spells them.
BIG5_COUNTRIES = ("ENG", "ESP", "GER", "ITA", "FRA")

#: football-data.co.uk name -> Club Elo name, for names that differ beyond
#: normalisation. Every entry here was confirmed against the Club Elo roster;
#: none was accepted from a fuzzy match.
NAME_ALIASES: dict[str, str] = {
    # England
    "Nott'm Forest": "Forest",
    # Spain
    "Ath Bilbao": "Bilbao",
    "Ath Madrid": "Atletico",          # NOT Real Madrid — fuzzy matching gets this wrong
    "Espanol": "Espanyol",
    "La Coruna": "Depor",
    "Sp Gijon": "Gijon",
    "Vallecano": "Rayo Vallecano",
    "Villareal": "Villarreal",         # football-data misspells it with one 'r'
    "Gimnastic": "Tarragona",
    "Lerida": "Lleida",
    # Germany
    "Bayern Munich": "Bayern",
    "Dusseldorf": "Duesseldorf",
    "Fortuna Dusseldorf": "Duesseldorf",
    "Ein Frankfurt": "Frankfurt",
    "FC Koln": "Koeln",
    "Greuther Furth": "Fuerth",
    "Hansa Rostock": "Rostock",
    "Holstein Kiel": "Holstein",
    "Kaiserslautern": "Lautern",
    "Leipzig": "RB Leipzig",
    "M'Gladbach": "Gladbach",
    "M'gladbach": "Gladbach",
    "Munich 1860": "Muenchen 60",
    "Nurnberg": "Nuernberg",
    "Schalke 04": "Schalke",
    "Werder Bremen": "Werder",
    # France
    "Ajaccio GFCO": "Gazelec",  # Gazelec Ajaccio, a different club from AC Ajaccio
    "Arles": "Arles-Avignon",
    "Evian Thonon Gaillard": "Evian TG",
    "St Etienne": "Saint-Etienne",
    "Paris SG": "Paris SG",
}


class ClubEloError(RuntimeError):
    pass


def normalise(name: str) -> str:
    """Accent-, case- and punctuation-insensitive key for club names."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = decomposed.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_only.lower())


def club_slug(clubelo_name: str) -> str:
    """Club Elo's per-club endpoint uses the name with spaces removed."""
    return str(clubelo_name).replace(" ", "")


# --- fetching -----------------------------------------------------------------

def _read_csv(body: bytes) -> pd.DataFrame:
    return pd.read_csv(StringIO(body.decode("utf-8-sig", errors="replace")))


def fetch_snapshot(
    on: Date | str, *, session: PoliteSession | None = None, force: bool = False
) -> pd.DataFrame:
    """Every club's Elo on one date, with country and tier (`Level`)."""
    session = session or PoliteSession()
    day = pd.Timestamp(on).date().isoformat()
    body = session.get_bytes(f"{BASE_URL}/{day}", suffix=".csv", force=force)
    df = _read_csv(body)
    if "Club" not in df.columns:
        raise ClubEloError(f"unexpected Club Elo snapshot schema for {day}: {list(df.columns)}")
    df["snapshot_date"] = pd.Timestamp(day)
    return df


def snapshot_dates(first_season: int, last_season: int) -> list[str]:
    """Two probe dates per season — autumn and spring.

    A single date per season misses clubs whose top-flight spell was brief, and
    those are exactly the promoted-then-relegated sides whose ratings matter most
    for the lower-division prior.
    """
    out: list[str] = []
    for year in range(first_season, last_season + 1):
        out.append(f"{year}-10-15")
        out.append(f"{year + 1}-03-15")
    return out


def build_roster(
    first_season: int = 1993,
    last_season: int | None = None,
    *,
    countries: tuple[str, ...] | None = BIG5_COUNTRIES,
    session: PoliteSession | None = None,
) -> pd.DataFrame:
    """Union of every club seen across per-season snapshots.

    Returns one row per club with its country and the best (numerically lowest,
    excluding 0) tier it was ever observed at.
    """
    if last_season is None:
        today = pd.Timestamp.today()
        last_season = today.year if today.month >= 7 else today.year - 1

    session = session or PoliteSession()
    frames = []
    for day in snapshot_dates(first_season, last_season):
        if pd.Timestamp(day) > pd.Timestamp.today():
            continue
        try:
            frames.append(fetch_snapshot(day, session=session))
        except (FetchError, ClubEloError):
            log.warning("club-elo: no snapshot for %s", day)

    if not frames:
        raise ClubEloError("no Club Elo snapshots could be fetched")

    allrows = pd.concat(frames, ignore_index=True)
    if countries:
        allrows = allrows[allrows["Country"].isin(countries)]

    allrows["Level"] = pd.to_numeric(allrows["Level"], errors="coerce")
    # Level 0 means "not in a ranked division at this date" and is not a tier.
    ranked = allrows[allrows["Level"] > 0]

    roster = (
        ranked.groupby("Club")
        .agg(country=("Country", "first"), best_tier=("Level", "min"))
        .reset_index()
        .rename(columns={"Club": "clubelo_name"})
    )
    roster["norm"] = roster["clubelo_name"].map(normalise)
    log.info("club-elo: roster of %d clubs", len(roster))
    return roster


def fetch_club_history(
    clubelo_name: str, *, session: PoliteSession | None = None, force: bool = False
) -> pd.DataFrame:
    """One club's full rating history as From/To intervals."""
    session = session or PoliteSession()
    body = session.get_bytes(
        f"{BASE_URL}/{club_slug(clubelo_name)}", suffix=".csv", force=force
    )
    df = _read_csv(body)
    if "Elo" not in df.columns or df.empty:
        raise ClubEloError(f"no Elo history returned for {clubelo_name!r}")

    out = pd.DataFrame(
        {
            "clubelo_name": df["Club"].astype("string"),
            "country": df["Country"].astype("string"),
            "tier": pd.to_numeric(df["Level"], errors="coerce").astype("Int64"),
            "elo": pd.to_numeric(df["Elo"], errors="coerce").astype("Float64"),
            "valid_from": pd.to_datetime(df["From"], errors="coerce"),
            "valid_to": pd.to_datetime(df["To"], errors="coerce"),
        }
    )
    out = out[out["elo"].notna() & out["valid_from"].notna()]
    return out.sort_values("valid_from").reset_index(drop=True)


# --- name resolution ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NameResolution:
    mapping: dict[str, str]          # football-data name -> Club Elo name
    unmatched: tuple[str, ...]
    suggestions: dict[str, tuple[str, ...]]
    #: Aliases accepted on trust because the roster has never seen the target.
    #: Reported separately because a *typo'd* alias lands here too, and would
    #: otherwise be indistinguishable from a correct one until the club silently
    #: vanished at fetch time — Club Elo answers an unknown club with an empty
    #: CSV rather than a 404, so nothing raises.
    unverified: tuple[str, ...] = ()

    @property
    def coverage(self) -> float:
        """Share of names mapped to a target the roster actually contains.

        Deliberately excludes `unverified`: counting trusted-but-unseen aliases as
        successes is what let four wrong aliases report as 100% coverage.
        """
        total = len(self.mapping) + len(self.unmatched)
        if not total:
            return 0.0
        return (len(self.mapping) - len(self.unverified)) / total


def resolve_names(
    source_names: list[str], roster: pd.DataFrame, *, aliases: dict[str, str] | None = None
) -> NameResolution:
    """Map football-data club names onto Club Elo names.

    Unresolved names are returned with fuzzy *suggestions* attached, but are never
    auto-accepted: the closest roster match to "Ath Madrid" is "Real Madrid", and
    accepting that would attach a title-winning club's rating to a different club
    for every fixture it played.
    """
    import difflib

    aliases = NAME_ALIASES if aliases is None else aliases
    by_norm = dict(zip(roster["norm"], roster["clubelo_name"], strict=False))
    known_norms = list(by_norm)

    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    unverified: list[str] = []
    suggestions: dict[str, tuple[str, ...]] = {}

    for name in sorted(set(source_names)):
        alias = aliases.get(name)
        if alias is not None and normalise(alias) in by_norm:
            mapping[name] = by_norm[normalise(alias)]
            continue

        hit = by_norm.get(normalise(name))
        if hit is not None:
            mapping[name] = hit
            continue

        # Alias points at a club the roster never saw. That can be legitimate — a
        # top-flight spell too brief for the probe dates to catch — so the alias is
        # still attempted. But it is flagged, because a typo lands here too and
        # Club Elo answers an unknown club with an empty CSV rather than an error.
        if alias is not None:
            mapping[name] = alias
            unverified.append(name)
            continue

        unmatched.append(name)
        close = difflib.get_close_matches(normalise(name), known_norms, n=3, cutoff=0.55)
        suggestions[name] = tuple(by_norm[c] for c in close)

    if unmatched:
        log.warning(
            "club-elo: %d/%d club names unresolved — add them to NAME_ALIASES: %s",
            len(unmatched), len(unmatched) + len(mapping), unmatched,
        )
    if unverified:
        log.warning(
            "club-elo: %d alias target(s) absent from the roster — verify against the "
            "API before trusting them, since an unknown club returns an empty CSV "
            "rather than an error: %s",
            len(unverified), {n: mapping[n] for n in unverified},
        )

    return NameResolution(mapping, tuple(unmatched), suggestions, tuple(unverified))


# --- as-of lookups ------------------------------------------------------------

def build_elo_table(
    mapping: dict[str, str], *, session: PoliteSession | None = None
) -> pd.DataFrame:
    """Fetch histories for every mapped club into one long table."""
    session = session or PoliteSession()
    frames = []
    for source_name, elo_name in sorted(mapping.items()):
        try:
            hist = fetch_club_history(elo_name, session=session)
        except (FetchError, ClubEloError):
            log.warning("club-elo: no history for %s (%s)", source_name, elo_name)
            continue
        hist["source_name"] = source_name
        frames.append(hist)

    if not frames:
        raise ClubEloError("no Elo histories fetched")
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(["source_name", "valid_from"]).reset_index(drop=True)


def elo_as_of(table: pd.DataFrame, source_name: str, on: Date | str) -> float | None:
    """Rating held by a club strictly *before* a given date.

    The strict inequality is the point. Club Elo's interval covering a match date
    already reflects that match's result, so using it as a pre-match feature would
    leak the outcome into the model (NFR-10). Feature building must ask for the
    rating as of the day before kick-off, and this enforces it.
    """
    day = pd.Timestamp(on)
    rows = table[(table["source_name"] == source_name) & (table["valid_from"] < day)]
    if rows.empty:
        return None
    return float(rows.iloc[-1]["elo"])


def tier_as_of(table: pd.DataFrame, source_name: str, on: Date | str) -> int | None:
    """Club's division tier before a date — the FR-9 lower-division prior input."""
    day = pd.Timestamp(on)
    rows = table[(table["source_name"] == source_name) & (table["valid_from"] < day)]
    if rows.empty:
        return None
    tier = rows.iloc[-1]["tier"]
    return None if pd.isna(tier) else int(tier)


def elo_file() -> Path:
    return paths.elo_file()
