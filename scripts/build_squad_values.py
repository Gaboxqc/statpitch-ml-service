"""Fetch squad market values and map them onto match-log clubs (Roadmap §4.1).

    python scripts/build_squad_values.py [--from 2005] [--to 2025]

Writes `data/processed/squad_values.parquet`: one row per club-season, keyed on
the club names the match log uses, so `build_features` can join it directly.

The valuation is lagged by a season, deliberately
=================================================

Transfermarkt's page for a past season does not state *when* within that season
its figures were taken, and an end-of-season valuation reflects the season it is
supposed to predict. Rather than guess, each match gets the club's value from the
**previous** season — unambiguously known before a ball is kicked.

That costs freshness: a club that spent heavily in July is valued as it was in
May. The alternative is a feature that might quietly contain its own target,
which is the failure NFR-10 exists to prevent and the one this project has spent
the most effort avoiding elsewhere. A slightly stale feature is a weaker feature;
a leaky one is a wrong answer that looks like a strong one.

Name resolution
===============

Transfermarkt writes formal names ("Manchester City", "Bayern Munich"),
football-data writes abbreviations ("Man City", "Bayern Munich"). Matching is
constrained to clubs that actually appear in the same competition, which is what
stops "Valencia" in Spain reaching a club of the same name elsewhere. Coverage is
reported per competition and the run fails below a floor rather than writing a
mapping that looks complete.
"""

from __future__ import annotations

import argparse
import logging
import unicodedata

import pandas as pd

from statpitch import paths
from statpitch.data import transfermarkt as tm

log = logging.getLogger("squad_values")

#: Below this share of club-seasons resolved, the join is not worth making.
#:
#: Enforced PER COMPETITION as well as overall, and the per-competition half is
#: the one that does the work. A pooled average is dominated by whichever
#: competitions are already clean: on the first eight-league run the Süper Lig
#: resolved 88.2% — under this floor — and the run passed at 97.4% overall
#: because five leagues sat at 100%. The league whose feature was most degraded
#: was precisely the one the guard could not see.
MIN_COVERAGE = 0.90

#: Transfermarkt name -> match-log name, for the cases normalisation cannot
#: reach. Every entry was confirmed against the club list of the same
#: competition, never accepted from a fuzzy match.
ALIASES: dict[str, str] = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR",
    "Athletic Bilbao": "Ath Bilbao",
    "Atlético de Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Real Sociedad": "Sociedad",
    "Celta de Vigo": "Celta",
    "RCD Espanyol Barcelona": "Espanol",
    "Espanyol Barcelona": "Espanol",
    "Deportivo de La Coruña": "La Coruna",
    "Rayo Vallecano": "Vallecano",
    "Sporting Gijón": "Sp Gijon",
    "Bayern Munich": "Bayern Munich",
    "Borussia Mönchengladbach": "M'gladbach",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "1.FC Köln": "FC Koln",
    "1.FSV Mainz 05": "Mainz",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "SV Werder Bremen": "Werder Bremen",
    "VfB Stuttgart": "Stuttgart",
    "Hertha BSC": "Hertha",
    "Fortuna Düsseldorf": "Fortuna Dusseldorf",
    "Internazionale": "Inter",
    "AC Milan": "Milan",
    "Hellas Verona": "Verona",
    "Paris Saint-Germain": "Paris SG",
    "Olympique Lyon": "Lyon",
    "Olympique Marseille": "Marseille",
    "AS Saint-Étienne": "St Etienne",
    "Stade Rennais FC": "Rennes",
    "Stade Brestois 29": "Brest",
    "1.FC Nuremberg": "Nurnberg",
    "Hamburger SV": "Hamburg",
    # Same city, different clubs — the cases token matching correctly refuses to
    # resolve, and the reason it requires a unique candidate rather than taking
    # the first. Naming them is the only safe way through.
    "Inter Milan": "Inter",
    "Chievo Verona": "Chievo",
    "AC Ajaccio": "Ajaccio",
    "GFC Ajaccio": "Ajaccio GFCO",
    # Netherlands. The match log renamed this club mid-archive: "Roda JC" up to
    # 2009/10, "Roda" from 2010/11. Only the later spelling falls inside the
    # 2010-2025 valuation window, and the seven unresolved club-seasons matched
    # its seven log seasons exactly.
    "Roda JC Kerkrade": "Roda",
    # Portugal
    "Sporting CP": "Sp Lisbon",
    "CF União Madeira (-2021)": "Uniao Madeira",
    # Belenenses SAD, the entity that played the top flight 2018-2024 after the
    # split from CF Os Belenenses. The match log calls it "Belenenses"
    # throughout, so this is a mapping onto the log's spelling rather than a
    # claim that the two are one club. Four unresolved club-seasons, and the log
    # carries exactly four (2018/19-2021/22).
    "B SAD (2018-2024)": "Belenenses",
    # Turkey. Three Transfermarkt spellings for İstanbul Başakşehir across its
    # renames, all one club in the log.
    "Basaksehir FK": "Buyuksehyr",
    "Istanbul Büyüksehir Belediyespor": "Buyuksehyr",
    "Büyüksehir Belediyespor": "Buyuksehyr",
    "Göztepe": "Goztep",
    "Adana Demirspor": "Ad. Demirspor",       # NOT Adanaspor
    "Kayseri Erciyesspor (1966-2018)": "Erciyesspor",  # NOT Kayserispor
    "Mersin Idmanyurdu": "Mersin Idman Yurdu",
    "Akhisarspor": "Akhisar Belediyespor",
    "Bodrum FK": "Bodrumspor",
}


#: Club-name noise: legal forms, sponsor initials and founding years. Both
#: sources sprinkle these differently — "AS Roma" against "Roma", "LOSC Lille"
#: against "Lille" — and they carry no information about which club is meant.
_NOISE = frozenset(
    {
        "fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "us", "uc", "ud", "cd",
        "sd", "ca", "rc", "rcd", "losc", "ogc", "ea", "estac", "gfc", "spvgg",
        "bsc", "vfb", "vfl", "tsg", "sv", "sp", "calcio", "club", "de", "del",
        "athletic", "atletico", "real", "borussia", "eintracht", "arminia",
        "fortuna", "hellas", "olympique", "stade", "delfino", "deportivo",
        "balompie", "lorraine", "1", "04", "05", "07", "98", "1846", "1909",
        "1913", "1936",
    }
)


def normalise(name: str) -> str:
    """Accent-, case- and punctuation-insensitive key."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = decomposed.encode("ascii", "ignore").decode().lower()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in ascii_only)
    return " ".join(cleaned.split())


def tokens(name: str) -> frozenset[str]:
    """Distinctive words only, so a legal form cannot decide a match.

    "Real" and "Borussia" are noise here rather than identity: several clubs in
    the same league carry each, and the competition constraint plus the remaining
    tokens are what separate them. Dropping them is only safe *because* matching
    requires a unique candidate — an ambiguous strip resolves to nothing rather
    than to the first guess.
    """
    return frozenset(w for w in normalise(name).split() if w not in _NOISE)


def resolve(values: pd.DataFrame, log_clubs: dict[str, set[str]]) -> pd.DataFrame:
    """Map Transfermarkt club names onto the match log's, per competition."""
    resolved: list[str | None] = []
    unmatched: dict[str, set[str]] = {}

    indexes = {
        competition: {normalise(c): c for c in clubs}
        for competition, clubs in log_clubs.items()
    }

    for row in values.itertuples():
        competition = str(row.competition_id)
        index = indexes.get(competition, {})
        club = str(row.club)

        target = ALIASES.get(club)
        if target is not None and target in log_clubs.get(competition, set()):
            resolved.append(target)
            continue

        hit = index.get(normalise(club))
        if hit is None and target is not None:
            hit = index.get(normalise(target))
        if hit is None:
            # Token containment, either direction: "AS Roma" against "Roma",
            # "Bournemouth" against "AFC Bournemouth". Constrained to clubs in
            # the same competition, and accepted only when exactly one candidate
            # survives — an ambiguous name resolves to nothing rather than to a
            # guess, which is the mistake `NAME_ALIASES` records for "Ath Madrid".
            key = tokens(club)
            candidates = {
                value
                for name, value in index.items()
                if key and (tokens(name) <= key or key <= tokens(name))
            }
            hit = candidates.pop() if len(candidates) == 1 else None
        if hit is None:
            unmatched.setdefault(competition, set()).add(club)
        resolved.append(hit)

    out = values.copy()
    out["club_resolved"] = resolved
    for competition, names in sorted(unmatched.items()):
        log.warning(
            "%s — %d unmatched club name(s): %s",
            competition, len(names), ", ".join(sorted(names)[:12]),
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="first", type=int, default=2005)
    parser.add_argument("--to", dest="last", type=int, default=2025)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    matches = pd.read_parquet(paths.matches_file())
    log_clubs = {
        str(competition): set(group["home_team"]) | set(group["away_team"])
        for competition, group in matches.groupby("competition_id")
    }

    seasons = list(range(args.first, args.last + 1))
    values = tm.build(seasons)
    if values.empty:
        log.error("no squad values fetched")
        return 1

    values = resolve(values, log_clubs)
    coverage = values["club_resolved"].notna().mean()
    log.info(
        "%d club-seasons, %.1f%% resolved onto match-log names",
        len(values), 100 * coverage,
    )
    for competition, group in values.groupby("competition_id"):
        log.info(
            "  %-16s %5d rows, %.1f%% resolved",
            competition, len(group), 100 * group["club_resolved"].notna().mean(),
        )

    thin = {
        str(competition): group["club_resolved"].notna().mean()
        for competition, group in values.groupby("competition_id")
        if group["club_resolved"].notna().mean() < MIN_COVERAGE
    }
    if coverage < MIN_COVERAGE or thin:
        if thin:
            log.error(
                "below the %.0f%% floor per competition: %s",
                100 * MIN_COVERAGE,
                ", ".join(f"{c} {100 * v:.1f}%" for c, v in sorted(thin.items())),
            )
        if coverage < MIN_COVERAGE:
            log.error("overall coverage %.1f%% is below the floor", 100 * coverage)
        log.error(
            "add the missing clubs to ALIASES rather than shipping this mapping"
        )
        return 1

    destination = paths.processed_dir() / "squad_values.parquet"
    values.dropna(subset=["club_resolved"]).to_parquet(destination, index=False)
    log.info("wrote %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
