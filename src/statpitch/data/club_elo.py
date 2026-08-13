"""Club Elo ingestion (Design §3, FR-9, FR-11).

clubelo.com publishes a free, keyless CSV API covering European club football
since the 1940s.

**It does not cover the full pyramid.** Requirements §7.1 and FR-9 describe it as
doing so; verified against the API, it rates only the top two tiers of each
country. Saarbrücken, Wrexham, Elversberg and Ulm (tiers 1-2) all return history;
AFC Sudbury, AFC Fylde, SC Verl, 1. FC Düren and AD Tardienta (tier 3+) return an
empty CSV — not a 404, which is why the gap is easy to miss.

FR-9's own worked example still holds: a Segunda División club in the Copa del
Rey is tier 2 and *is* rated. But domestic cups admit entrants from far deeper,
so the lower-division prior needs a documented fallback for clubs below tier 2
rather than an Elo lookup. See `CLUB_ELO_TIER_LIMIT`.

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

#: openfootball name -> Club Elo name. A separate table from `NAME_ALIASES`
#: because the two sources spell clubs differently in opposite directions:
#: football-data.co.uk abbreviates ("Ath Bilbao"), openfootball uses the formal
#: registered name ("Athletic Club"). One dictionary serving both would have to
#: hold every club twice and would hide which source a fix was for.
#:
#: Only the residue is listed. Country-constrained matching resolves 80 of the 96
#: clubs in a five-league fixture list on its own; these are the ones it cannot,
#: and every entry was confirmed against the roster rather than accepted from a
#: fuzzy match.
#:
#: The first two are why auto-accepting is not an option. "FC Bayern München"
#: matches both `Bayern` and `Muenchen 60`, and "RCD Espanyol de Barcelona"
#: matches both `Espanyol` and `Barcelona` — resolving either by picking the
#: closest string would attach a title-winning club's rating to a different club
#: for every fixture it plays, which is the failure `NAME_ALIASES` documents for
#: "Ath Madrid".
OPENFOOTBALL_ALIASES: dict[str, str] = {
    # Ambiguous without help — see above.
    "FC Bayern München": "Bayern",
    "RCD Espanyol de Barcelona": "Espanyol",
    # England
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Brighton & Hove Albion FC": "Brighton",
    # Spain
    "Athletic Club": "Bilbao",
    "Real Betis Balompié": "Betis",
    "RC Deportivo La Coruña": "Depor",
    "Real Racing Club de Santander": "Santander",
    # Germany
    "Borussia Mönchengladbach": "Gladbach",
    "Hamburger SV": "Hamburg",
    # Italy
    "FC Internazionale Milano": "Inter",
    # France
    "Olympique Lyonnais": "Lyon",
    "Paris Saint-Germain FC": "Paris SG",
    "Stade Brestois 29": "Brest",
    "Stade Rennais FC 1901": "Rennes",
}


class ClubEloError(RuntimeError):
    pass


def normalise(name: str) -> str:
    """Accent-, case- and punctuation-insensitive key for club names."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = decomposed.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_only.lower())


#: Club Elo transliterates German umlauts rather than stripping them — "Köln" is
#: "Koeln", "Nürnberg" is "Nuernberg". Unicode NFKD decomposition gives "koln"
#: and "nurnberg" instead, so both forms have to be generated or every German
#: club with an umlaut silently fails to match.
_TRANSLITERATION = str.maketrans(
    {"ö": "oe", "ü": "ue", "ä": "ae", "Ö": "oe", "Ü": "ue", "Ä": "ae",
     "ß": "ss", "ø": "o", "å": "a", "æ": "ae"}
)

#: Legal-form abbreviations carried by openfootball's formal names ("1. FC Köln",
#: "AC Milan", "SpVgg Greuther Fürth") but not by Club Elo's short ones.
#:
#: Strictly legal forms only. Words like "Real", "Atletico", "Sporting" and
#: "Union" look like boilerplate but are load-bearing parts of club names —
#: stripping "Real" from "Real Madrid" leaves "madrid", which no longer matches
#: Club Elo's "Real Madrid" and starts colliding with Atlético.
_CLUB_TYPE_TOKENS = frozenset([
    # club/association abbreviations
    "fc", "cf", "ac", "as", "ss", "ssc", "asd", "ssd", "sd", "sr", "afc",
    "bk", "if", "ik", "nk", "hnk", "mfk", "fk", "sk", "sv", "tsv", "tsg",
    "bsc", "vfb", "vfl", "spvgg", "sc", "cs", "us", "usl", "rc", "rcd",
    "ogc", "sco", "fco", "cd", "ud", "ca", "aa", "ec", "cfr", "acf", "aca",
    # spelled-out legal forms and connectives
    "calcio", "club", "de", "la", "le",
    "futbol", "fussball", "fotball", "sportiva", "societa", "associazione",
])


def _candidate_keys(name: str, *, reduce_to_single_token: bool = False) -> set[str]:
    """Every reasonable normalised form of a club name.

    openfootball writes formal names ("1. FC Köln", "SpVgg Greuther Fürth") while
    Club Elo writes short ones ("Koeln", "Fuerth"). Matching them needs several
    candidate forms per name: with and without umlaut transliteration, with and
    without club-type tokens, and with or without founding-year numbers.

    `reduce_to_single_token` reduces a name to its first or last distinctive word.
    It buys roughly 23 points of coverage and is **off by default because it is
    unsafe**: it resolved "Paris Saint-Germain FC" to "Paris FC", a different
    club, since both reduce to "paris". City and regional names are shared across
    clubs, so no token rule can separate them — only a curated alias can. It is
    retained for generating *suggestions* to a human, never for accepting a match.
    """
    text = str(name).strip()
    keys: set[str] = set()

    for base in (text, text.translate(_TRANSLITERATION)):
        ascii_only = (
            unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode().lower()
        )
        tokens = [t for t in re.sub(r"[^a-z0-9 ]", " ", ascii_only).split() if t]
        if not tokens:
            continue
        keys.add("".join(tokens))

        # Drop club-type words and bare founding years ("1. FC Schweinfurt 05").
        core = [
            t for t in tokens
            if t not in _CLUB_TYPE_TOKENS and not re.fullmatch(r"\d{2,4}", t)
        ]
        if core:
            keys.add("".join(core))
            if reduce_to_single_token and len(core) > 1:
                keys.add(core[-1])
                keys.add(core[0])

    return {k for k in keys if len(k) > 2}


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


@dataclass(frozen=True, slots=True)
class CupClubResolution:
    """Outcome of matching cup entrants against Club Elo.

    Three-way, deliberately. `ambiguous` exists so that a name matching several
    Club Elo clubs is never silently resolved to one of them, and `unmatched` is
    dominated by a real coverage limit rather than by naming noise — see
    `CLUB_ELO_TIER_LIMIT`.
    """

    mapping: dict[str, str]
    ambiguous: dict[str, tuple[str, ...]]
    unmatched: tuple[str, ...]

    @property
    def coverage(self) -> float:
        total = len(self.mapping) + len(self.ambiguous) + len(self.unmatched)
        return len(self.mapping) / total if total else 0.0


#: Club Elo rates only the top two tiers of each country. Verified directly
#: against the API: Saarbrücken, Wrexham, Elversberg and Ulm (tiers 1-2) all
#: return history, while AFC Sudbury, AFC Fylde, SC Verl, 1. FC Düren and AD
#: Tardienta (tier 3+) return an EMPTY CSV.
#:
#: This qualifies Requirements §7.1 and FR-9, which describe Club Elo as covering
#: "the full pyramid". FR-9's own worked example — a Segunda División club in the
#: Copa del Rey — is tier 2 and *is* covered. Deeper entrants are not, and
#: domestic cups are full of them, so the lower-division prior needs a documented
#: fallback for clubs below tier 2 rather than an Elo lookup.
CLUB_ELO_TIER_LIMIT = 2


def _name_tokens(name: str) -> frozenset[str]:
    """Distinctive lowercase word set, umlauts transliterated, legal forms dropped."""
    ascii_only = (
        unicodedata.normalize("NFKD", str(name).translate(_TRANSLITERATION))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    tokens = re.sub(r"[^a-z0-9 ]", " ", ascii_only).split()
    # Single characters are kept deliberately. Club Elo marks reserve sides with a
    # trailing "B" ("Atletico B", "Bilbao B", "Sociedad B"); dropping it makes a
    # reserve team's token set identical to the first team's, so "Atlético Madrid"
    # matches both and is discarded as ambiguous. Keeping it lets the subset rule
    # reject the reserve side and resolve the first team cleanly.
    return frozenset(
        t for t in tokens
        if t not in _CLUB_TYPE_TOKENS and not re.fullmatch(r"\d{1,4}", t)
    )


def _token_subset_matches(
    name: str,
    country: str | None,
    token_index: list[tuple[str | None, frozenset[str], str]],
) -> set[str]:
    """Roster clubs whose every distinctive word appears in `name`."""
    query = _name_tokens(name)
    if not query:
        return set()
    return {
        clubelo_name
        for roster_country, tokens, clubelo_name in token_index
        if tokens
        and tokens <= query
        # A short roster name is a subset of far too much: "Paris FC" is a subset
        # of "Paris Saint-Germain", and accepting that attaches a second-tier
        # club's rating to the French champions. Requiring the match to account
        # for all but one of the query's distinctive words keeps "Dortmund" ⊂
        # "Borussia Dortmund" while rejecting the Paris collision.
        and len(query - tokens) <= 1
        and (country is None or roster_country == country)
    }


def resolve_cup_clubs(
    names: dict[str, str | None], roster: pd.DataFrame
) -> CupClubResolution:
    """Match cup entrants to Club Elo, constrained by country.

    `names` maps a club name to its ISO-3 country code (or None when unknown).
    The country constraint is doing real work: without it "Union" and "Atletico"
    match clubs in half a dozen countries, and picking one would repeat exactly
    the mistake that mapped Atlético Madrid to Real Madrid.
    """
    index: dict[tuple[str | None, str], set[str]] = {}
    for row in roster.itertuples():
        # Roster side: canonical forms only, no single-token reduction.
        for key in _candidate_keys(row.clubelo_name):
            index.setdefault((row.country, key), set()).add(row.clubelo_name)
            index.setdefault((None, key), set()).add(row.clubelo_name)

    token_index = [
        (row.country, _name_tokens(row.clubelo_name), row.clubelo_name)
        for row in roster.itertuples()
    ]

    mapping: dict[str, str] = {}
    ambiguous: dict[str, tuple[str, ...]] = {}
    unmatched: list[str] = []

    for name, raw_country in sorted(names.items()):
        # NaN is truthy in Python, so a caller writing `row.home_country or
        # fallback` against a DataFrame quietly produces NaN rather than the
        # fallback, and every lookup keyed on it misses. Normalising here means
        # callers reading straight from pandas cannot fall into that trap.
        country = None if raw_country is None or pd.isna(raw_country) else str(raw_country)

        # Safe forms only. Single-token reduction is deliberately not used here
        # (see _candidate_keys) — it maps "Paris Saint-Germain FC" onto "Paris FC".
        query_keys = _candidate_keys(name)
        candidates: set[str] = set()
        for key in query_keys:
            candidates |= index.get((country, key), set())
        if not candidates and country is None:
            for key in query_keys:
                candidates |= index.get((None, key), set())

        if not candidates:
            # Token-subset fallback: every word of the Club Elo name must appear
            # in the query. "Brugge" is a subset of "Club Brugge KV" while
            # "Cercle Brugge" is not, so it accepts the right club and rejects the
            # near-miss that fuzzy matching ranks first. Where several roster
            # clubs qualify — "Sporting" and "Braga" are both subsets of
            # "Sporting Braga" — the result is ambiguous and is reported, not
            # picked.
            candidates = _token_subset_matches(name, country, token_index)

        if len(candidates) == 1:
            mapping[name] = next(iter(candidates))
        elif candidates:
            ambiguous[name] = tuple(sorted(candidates))
        else:
            unmatched.append(name)

    if ambiguous:
        log.info(
            "club-elo: %d cup club name(s) matched more than one club and were left "
            "unresolved rather than guessed", len(ambiguous),
        )
    if unmatched:
        log.info(
            "club-elo: %d cup club name(s) have no Club Elo entry — expected, since "
            "Club Elo rates only tiers 1-%d and cups admit far deeper entrants",
            len(unmatched), CLUB_ELO_TIER_LIMIT,
        )

    return CupClubResolution(mapping, ambiguous, tuple(unmatched))


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
