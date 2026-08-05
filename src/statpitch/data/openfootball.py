"""openfootball ingestion — domestic cups and continental competitions (Phase 1).

CC0 public domain, plain-text, no key. This is the only free source covering the
five domestic cups and UEFA club competitions, which is why Requirements §7.1
depends on it for everything football-data.co.uk does not carry.

What is actually there, verified against the live repos
======================================================

Requirements §7.1 describes "per-country repos + Champions League / Europa League
repo" as though cup history matched the league archive's depth. It does not, and
two of its assumptions are wrong outright:

* There is **no `openfootball/europa-league` repo**. Europa League data lives as
  `el.txt` inside the `champions-league` repo.
* **`openfootball/france` now redirects to `openfootball/europe`**, a consolidated
  multi-country repo with a different path layout.

Coverage is recent and uneven:

    competition            seasons  range
    UEFA.UCL                    15  2011-12 .. 2025-26
    GER.DFB_POKAL                8  2018-19 .. 2025-26
    ENG.FA_CUP                   7  2018-19 .. 2024-25
    UEFA.UEL                     6  2020-21 .. 2025-26
    ESP.COPA_DEL_REY             5  2020-21 .. 2024-25
    ITA.COPPA_ITALIA             5  2020-21 .. 2024-25
    FRA.COUPE_DE_FRANCE          1  2024-25 only

One season of Coupe de France cannot train a competition-specific model. This is
what makes Design §5.1's joint multi-competition training with a `competition_id`
embedding load-bearing rather than an elegance: the thin competitions are only
predictable at all by borrowing structure from the data-rich ones.

The file format
===============

A line-oriented DSL::

    = DFB Pokal 2023/24
    ▪ Round 1
    Fri Aug 11
      18:00  1. FC Saarbrücken   2-1 (0-0)              Karlsruher SC
             SV Sandhausen       4-2 pen. 3-3 a.e.t. (3-3, 1-2)  Hannover 96

Continental files add a ``v`` separator and country suffixes::

    21:00  FC København (DEN)  v Manchester City FC (ENG)  1-3 (1-2)

The extra-time and penalty annotations are the reason this parser earns its keep:
they are exactly the observations FR-8's extra-time/penalty sub-model needs, and
no other free source in the stack carries them. Regulation, extra-time and
shootout scores are kept in separate columns, because a goals model must train on
the 90-minute score while the *qualification* outcome is decided by the shootout.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd

from statpitch import taxonomy
from statpitch.data.http import FetchError, PoliteSession

log = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/openfootball"

#: (repo, path template) per competition. `{season}` is the "2023-24" form.
#: Note ESP/GER/ITA all use "cup.txt" while England distinguishes facup from
#: eflcup (the League Cup is out of scope), and France lives in the consolidated
#: `europe` repo under a different layout entirely.
SOURCES: dict[str, tuple[str, str]] = {
    "ENG.FA_CUP": ("england", "{season}/facup.txt"),
    "ESP.COPA_DEL_REY": ("espana", "{season}/cup.txt"),
    "GER.DFB_POKAL": ("deutschland", "{season}/cup.txt"),
    "ITA.COPPA_ITALIA": ("italy", "{season}/cup.txt"),
    "FRA.COUPE_DE_FRANCE": ("europe", "france/{season}_frcup.txt"),
    "UEFA.UCL": ("champions-league", "{season}/cl.txt"),
    "UEFA.UEL": ("champions-league", "{season}/el.txt"),
}

#: Qualifying-round files, present only for recent seasons.
QUALIFIER_SOURCES: dict[str, tuple[str, str]] = {
    "UEFA.UCL": ("champions-league", "{season}/clq.txt"),
    "UEFA.UEL": ("champions-league", "{season}/elq.txt"),
}

STAGE_MARKER = "▪"

#: Canonical stage names, matching the keys in competitions.json `stage_formats`.
#: The German entries are not defensive padding — single files really do mix
#: languages ("Gruppe G" alongside "Group F", "10. Runde" alongside "Round 9").
_STAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^final$", "final"),
    (r"^semi[\s-]?finals?$|^halbfinale$", "semi_final"),
    (r"^quarter[\s-]?finals?$|^viertelfinale$", "quarter_final"),
    (r"^round of 16$|^achtelfinale$|^last 16$", "round_of_16"),
    # "Sechzehntelfinale" is German for the round of 32 — it appears untranslated
    # in UEL 2020-21 and, before this entry existed, fell through to the
    # competition default and recorded 32 two-legged ties as Swiss league fixtures.
    (r"^round of 32$|^sechzehntelfinale$|^last 32$", "round_of_32"),
    (r"^play[\s-]?offs?$", "knockout_playoff"),
    (r"^league$", "league_phase"),
    (r"^group$|^gruppe$", "group_stage"),
    (r"^group [a-l]$|^gruppe [a-l]$", "group_stage"),
    (r"^preliminary round$|^vorrunde$", "preliminary_round"),
    (r"^qualifying|^qualifikation", "qualifying"),
    (r"^round (\d+)$|^(\d+)\.? ?runde$", "round_{}"),
)

_TIME_RE = re.compile(r"^\s*(\d{1,2}[:.]\d{2})\s+")
#: The " v " separator used by continental files. Requires surrounding whitespace
#: so it cannot fire inside a club name.
_VS_RE = re.compile(r"\s+v\.?\s+")
_COUNTRY_SUFFIX_RE = re.compile(r"\s*\(([A-Z]{3})\)\s*$")

#: The score block, in the one place it can appear: between the two team names.
#: Ordered alternatives matter — the penalty form must be tried before the plain
#: one, or "4-2 pen. 3-3" parses as a 4-2 final score.
_SCORE_RE = re.compile(
    r"""
    (?:(?P<pen_h>\d{1,2})-(?P<pen_a>\d{1,2})\s+pen\.\s+)?      # shootout, if any
    (?P<a_h>\d{1,2})-(?P<a_a>\d{1,2})                          # headline score
    (?P<aet>\s+a\.e\.t\.)?                                     # after extra time?
    (?:\s*\(
        (?P<p1_h>\d{1,2})-(?P<p1_a>\d{1,2})
        (?:\s*,\s*(?P<p2_h>\d{1,2})-(?P<p2_a>\d{1,2}))?
    \))?
    """,
    re.VERBOSE,
)

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"], start=1
    )
}
_DATE_RE = re.compile(
    r"^\s*(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?,?\s+"
    r"(?:(?P<d1>\d{1,2})\s+(?P<m1>[a-z]{3})|(?P<m2>[a-z]{3})[a-z]*\.?\s+(?P<d2>\d{1,2}))"
    r"(?:\s+(?P<year>\d{4}))?\s*$",
    re.IGNORECASE,
)


class OpenFootballError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Match:
    stage: str
    date: pd.Timestamp | None
    home_team: str
    away_team: str
    home_country: str | None
    away_country: str | None
    home_goals: int | None       # regulation (90 minutes)
    away_goals: int | None
    home_goals_ht: int | None
    away_goals_ht: int | None
    home_goals_aet: int | None
    away_goals_aet: int | None
    home_pens: int | None
    away_pens: int | None
    went_to_extra_time: bool
    went_to_penalties: bool


def season_label(season: str) -> str:
    """'2023-24' -> '2023-2024', the canonical project form."""
    start = int(str(season).split("-")[0])
    return f"{start}-{start + 1}"


def season_dir(start_year: int) -> str:
    """2023 -> '2023-24', the directory name openfootball uses."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def normalise_stage(label: str) -> str:
    """Map a file's stage label onto a canonical taxonomy stage name.

    Continental files prefix the phase ("Finals, Round of 16", "Group, Matchday
    1"); the trailing component is the one that determines format, so it wins.
    """
    text = str(label).replace(STAGE_MARKER, "").strip()
    if not text:
        return "unknown"

    parts = [p.strip() for p in text.split(",") if p.strip()]
    # "Group, Matchday 3" -> the phase is "Group"; "Finals, Round of 16" -> the
    # round is "Round of 16". Try the most specific component first.
    candidates = list(reversed(parts)) + [text]

    for candidate in candidates:
        cleaned = re.sub(r"\s+", " ", candidate).strip().lower()
        cleaned = re.sub(r"\s*matchday\s*\d+$", "", cleaned).strip()
        if not cleaned:
            continue
        for pattern, target in _STAGE_PATTERNS:
            m = re.match(pattern, cleaned)
            if m:
                if "{}" in target:
                    number = next((g for g in m.groups() if g), "")
                    return target.format(number)
                return target

    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unknown"


def _parse_date(line: str, season_start: int, last: pd.Timestamp | None) -> pd.Timestamp | None:
    m = _DATE_RE.match(line)
    if not m:
        return None
    day = int(m.group("d1") or m.group("d2"))
    month_name = (m.group("m1") or m.group("m2")).lower()[:3]
    month = _MONTHS.get(month_name)
    if month is None:
        return None

    year = m.group("year")
    if year:
        return pd.Timestamp(int(year), month, day)

    # No year on the line. A season spans two calendar years, so months from July
    # belong to the opening year and the rest to the following one.
    inferred = season_start if month >= 7 else season_start + 1
    stamp = pd.Timestamp(inferred, month, day)
    # Guard the July boundary: dates only move forward within a file.
    if last is not None and stamp < last and (last - stamp).days > 200:
        stamp = pd.Timestamp(inferred + 1, month, day)
    return stamp


def _strip_country(name: str) -> tuple[str, str | None]:
    m = _COUNTRY_SUFFIX_RE.search(name)
    if not m:
        return name.strip(), None
    return _COUNTRY_SUFFIX_RE.sub("", name).strip(), m.group(1)


def _parse_scores(block: re.Match) -> dict[str, int | bool | None]:
    """Turn a matched score block into regulation / a.e.t. / shootout columns.

    The parenthesised pairs disambiguate: two pairs mean (90 minutes, half-time),
    one pair means half-time only. So a headline score annotated `a.e.t.` is the
    extra-time result, and the 90-minute score — the one a goals model must train
    on — is the first pair inside the brackets.
    """
    g = block.groupdict()
    to_int = lambda k: None if g.get(k) is None else int(g[k])  # noqa: E731

    head_h, head_a = to_int("a_h"), to_int("a_a")
    p1 = (to_int("p1_h"), to_int("p1_a"))
    p2 = (to_int("p2_h"), to_int("p2_a"))
    aet = g.get("aet") is not None
    pens = g.get("pen_h") is not None

    if p2[0] is not None:
        regulation, half_time = p1, p2
    else:
        regulation, half_time = (None, None), p1

    if aet:
        aet_score = (head_h, head_a)
        if regulation[0] is None:
            # Annotated a.e.t. with only a half-time bracket; regulation unknown.
            regulation = (None, None)
    else:
        aet_score = (None, None)
        if regulation[0] is None:
            regulation = (head_h, head_a)

    return {
        "home_goals": regulation[0],
        "away_goals": regulation[1],
        "home_goals_ht": half_time[0],
        "away_goals_ht": half_time[1],
        "home_goals_aet": aet_score[0],
        "away_goals_aet": aet_score[1],
        "home_pens": to_int("pen_h"),
        "away_pens": to_int("pen_a"),
        "went_to_extra_time": aet,
        "went_to_penalties": pens,
    }


def parse_football_txt(text: str, season_start: int) -> list[Match]:
    """Parse one football.txt file into matches."""
    matches: list[Match] = []
    stage = "unknown"
    current_date: pd.Timestamp | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith(("#", "=")):
            continue

        if line.lstrip().startswith(STAGE_MARKER):
            stage = normalise_stage(line)
            continue

        parsed_date = _parse_date(line, season_start, current_date)
        if parsed_date is not None:
            current_date = parsed_date
            continue

        body = _TIME_RE.sub("", line).strip()
        if not body:
            continue

        # Two layouts, and they place the score differently:
        #   cups:        "Home  2-1 (0-0)  Away"
        #   continental: "Home (ENG)  v  Away (ESP)  1-3 (1-2)"
        # Assuming the cup layout for a continental line silently yields an empty
        # away side and drops the match, so the separator is detected first.
        separator = _VS_RE.search(body)
        if separator is not None:
            home_raw = body[: separator.start()].strip()
            remainder = body[separator.end():].strip()
            score = _SCORE_RE.search(remainder)
            if score is None:
                continue
            away_raw = remainder[: score.start()].strip()
        else:
            score = _SCORE_RE.search(body)
            if score is None:
                continue
            home_raw = body[: score.start()].strip()
            away_raw = body[score.end():].strip()

        if not home_raw or not away_raw:
            continue

        home, home_country = _strip_country(home_raw)
        away, away_country = _strip_country(away_raw)

        matches.append(
            Match(
                stage=stage,
                date=current_date,
                home_team=home,
                away_team=away,
                home_country=home_country,
                away_country=away_country,
                **_parse_scores(score),  # type: ignore[arg-type]
            )
        )

    return matches


# --- fetching and assembly ----------------------------------------------------

def fetch_file(
    repo: str, path: str, *, session: PoliteSession | None = None, force: bool = False
) -> str | None:
    """Fetch one file, returning None when it does not exist upstream."""
    session = session or PoliteSession()
    url = f"{RAW_BASE}/{repo}/master/{path}"
    try:
        return session.get_bytes(url, suffix=".txt", force=force).decode(
            "utf-8", errors="replace"
        )
    except FetchError as exc:
        if "404" in str(exc):
            return None
        raise


def _resolve_leg_numbers(frame: pd.DataFrame, competition_id: str) -> pd.DataFrame:
    """Number the legs of two-legged ties (FR-7).

    Within a stage the same pair meets twice with home advantage reversed, so the
    legs are ordered by date per unordered pair. Ties settled in one match — a
    single-leg round, or a final — stay null rather than being labelled leg 1.
    """
    frame = frame.copy()
    frame["leg_number"] = pd.NA

    two_leg = frame["format"] == "two_leg_knockout"
    if not two_leg.any():
        return frame

    subset = frame[two_leg]
    pair_key = subset.apply(
        lambda r: (r["stage"], *sorted((str(r["home_team"]), str(r["away_team"])))), axis=1
    )
    for _, idx in subset.groupby(pair_key).groups.items():
        rows = frame.loc[idx].sort_values("date")
        if len(rows) != 2:
            # A one-off in a nominally two-legged stage (a neutral-venue final, or
            # incomplete data). Leaving it null is honest; guessing is not.
            continue
        frame.loc[rows.index[0], "leg_number"] = 1
        frame.loc[rows.index[1], "leg_number"] = 2

    return frame


def build_competition(
    competition_id: str,
    seasons: list[int],
    *,
    session: PoliteSession | None = None,
    include_qualifiers: bool = True,
) -> pd.DataFrame:
    """Fetch and parse every available season of one competition."""
    if competition_id not in SOURCES:
        raise OpenFootballError(f"no openfootball source mapped for {competition_id!r}")

    comp = taxonomy.get(competition_id)
    session = session or PoliteSession()
    rows: list[dict] = []

    targets = [(*SOURCES[competition_id], False)]
    if include_qualifiers and competition_id in QUALIFIER_SOURCES:
        targets.append((*QUALIFIER_SOURCES[competition_id], True))

    unknown_stages: set[str] = set()

    for start_year in seasons:
        directory = season_dir(start_year)
        season = season_label(directory)
        for repo, template, is_qualifier in targets:
            text = fetch_file(repo, template.format(season=directory), session=session)
            if text is None:
                continue
            for m in parse_football_txt(text, start_year):
                # Qualifier files label their stages "Round 1", "Playoffs" —
                # indistinguishable from a domestic cup's rounds by label alone.
                # Which file it came from is the only disambiguator, and it is the
                # caller that knows. Without this every UCL qualifying round falls
                # through to the competition default and is recorded as a Swiss
                # league-phase fixture.
                stage = "qualifying" if is_qualifier else m.stage
                if not comp.knows_stage(stage):
                    unknown_stages.add(stage)
                stage_format = comp.resolve_format(stage=stage, season=season)
                rows.append(
                    {
                        "competition_id": competition_id,
                        "season": season,
                        "season_start_year": start_year,
                        "stage": stage,
                        "stage_detail": m.stage,
                        "format": stage_format,
                        "neutral_venue": comp.is_neutral_venue(m.stage),
                        "date": m.date,
                        "home_team": m.home_team,
                        "away_team": m.away_team,
                        "home_country": m.home_country,
                        "away_country": m.away_country,
                        "home_goals": m.home_goals,
                        "away_goals": m.away_goals,
                        "home_goals_ht": m.home_goals_ht,
                        "away_goals_ht": m.away_goals_ht,
                        "home_goals_aet": m.home_goals_aet,
                        "away_goals_aet": m.away_goals_aet,
                        "home_pens": m.home_pens,
                        "away_pens": m.away_pens,
                        "went_to_extra_time": m.went_to_extra_time,
                        "went_to_penalties": m.went_to_penalties,
                        "source": "openfootball",
                        # No free odds source covers cups; this flag is what stops
                        # the Decision Layer touching them (Requirements §9).
                        "odds_coverage": comp.odds_coverage,
                    }
                )

    if unknown_stages:
        # Loud, because the fallback is dangerous: an unrecognised stage takes the
        # competition's DEFAULT format, and for UCL/UEL that default is
        # swiss_league_phase. A silent fallback turns a two-legged qualifier into
        # a Swiss league fixture and nothing raises.
        log.warning(
            "openfootball: %s — %d unrecognised stage(s), each falling back to the "
            "competition default format %r: %s",
            competition_id, len(unknown_stages), comp.format, sorted(unknown_stages),
        )

    if not rows:
        log.warning("openfootball: no data for %s", competition_id)
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    for col in ("home_goals", "away_goals", "home_goals_ht", "away_goals_ht",
                "home_goals_aet", "away_goals_aet", "home_pens", "away_pens"):
        frame[col] = frame[col].astype("Int64")

    frame = frame[frame["home_goals"].notna() | frame["home_goals_aet"].notna()]
    frame = _resolve_leg_numbers(frame, competition_id)
    frame["match_id"] = (
        frame["competition_id"]
        + "|" + frame["date"].dt.strftime("%Y-%m-%d").fillna("undated")
        + "|" + frame["stage"].astype(str)
        + "|" + frame["home_team"].str.replace(r"\s+", "", regex=True)
        + "|" + frame["away_team"].str.replace(r"\s+", "", regex=True)
    )
    frame = frame.drop_duplicates(subset="match_id", keep="first")
    return frame.sort_values(["date", "stage"]).reset_index(drop=True)


def build_all(
    seasons: list[int] | None = None,
    *,
    session: PoliteSession | None = None,
) -> pd.DataFrame:
    """Every mapped competition, across every season openfootball publishes."""
    if seasons is None:
        seasons = list(range(2011, 2027))
    session = session or PoliteSession()
    frames = []
    for competition_id in SOURCES:
        try:
            frame = build_competition(competition_id, seasons, session=session)
        except Exception:
            log.exception("openfootball: failed to build %s", competition_id)
            continue
        if not frame.empty:
            log.info(
                "openfootball: %s — %d matches, %d seasons",
                competition_id, len(frame), frame["season"].nunique(),
            )
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
