"""Squad market values from Transfermarkt (Roadmap §4.1).

`cloudscraper` has been a declared dependency since the first commit, commented
"Transfermarkt (Phase 2)", and imported nowhere. This is that ingester.

Why this source and not another feature
=======================================

MODEL_CARD §6 names squad values as a gap, and §4 explains why everything else
tried has failed: xG moved the gap 0.0007 because "bookmakers use the same public
shot data", and the momentum features moved nothing because Club Elo already
integrates recent results. Both were **derivatives of information the model
already held**.

A squad's market value is not. It is a forward-looking assessment of playing
staff — transfer activity, contract length, age profile, injuries priced in by a
market of people who watch clubs rather than results. A rating says what a club
has done; a valuation says what it has to work with. That is the one remaining
free source that is a genuinely different measurement rather than a restatement.

The honest prior is still poor. Bookmakers read Transfermarkt too.

Politeness
==========

Every request goes through `PoliteSession` for the delay, retry and on-disk cache
(NFR-5), with `cloudscraper` supplying the transport because Transfermarkt sits
behind a challenge that a bare `requests` session does not clear. One page per
competition-season, ~110 pages for the whole archive, fetched once and cached.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd
from bs4 import BeautifulSoup

from statpitch.data.http import PoliteSession

log = logging.getLogger(__name__)

BASE = "https://www.transfermarkt.com"

#: competition_id -> Transfermarkt's competition code. Only the five leagues:
#: cup squads are the same clubs, and Transfermarkt has no per-cup valuation.
COMPETITIONS: dict[str, str] = {
    "ENG.PL": "GB1",
    "ESP.LALIGA": "ES1",
    "GER.BUNDESLIGA": "L1",
    "ITA.SERIEA": "IT1",
    "FRA.LIGUE1": "FR1",
}

#: Transfermarkt writes amounts as "€1.46bn", "€955.65m", "€500k".
_AMOUNT = re.compile(r"([\d.,]+)\s*(bn|m|k)?", re.IGNORECASE)
_MULTIPLIER = {"bn": 1e9, "m": 1e6, "k": 1e3, None: 1.0}


class TransfermarktError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ClubValue:
    club: str
    squad_size: int | None
    average_age: float | None
    total_value_eur: float | None
    average_value_eur: float | None


def parse_amount(text: str) -> float | None:
    """'€955.65m' -> 955650000.0, and '-' -> None.

    A missing value is None rather than 0.0. Zero would read as "this squad is
    worthless", which for a promoted club with an unlisted valuation is both
    wrong and exactly the direction that biases a model.
    """
    cleaned = str(text).replace("€", "").replace("\xa0", " ").strip()
    if not cleaned or cleaned in {"-", "?"}:
        return None
    match = _AMOUNT.search(cleaned)
    if not match:
        return None
    number = match.group(1).replace(",", "")
    try:
        value = float(number)
    except ValueError:
        return None
    suffix = (match.group(2) or "").lower() or None
    return value * _MULTIPLIER.get(suffix, 1.0)


def season_url(code: str, season_start: int) -> str:
    return (
        f"{BASE}/x/startseite/wettbewerb/{code}/plus/?saison_id={season_start}"
    )


def parse_season(html: str) -> list[ClubValue]:
    """Parse one competition-season page into per-club valuations.

    The table is positional — club, squad size, average age, foreigners, average
    value, total value — so the row is read by index and every cell is allowed to
    be missing. Transfermarkt reorders columns between layouts, and a parser that
    assumed a header would fail loudly on a redesign; this one degrades to nulls,
    which `build` then reports as a coverage figure rather than a crash.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.items")
    if table is None:
        raise TransfermarktError("no table.items on the page — layout changed?")

    out: list[ClubValue] = []
    for row in table.select("tbody > tr"):
        cells = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cells) < 7:
            continue
        club = cells[1]
        if not club:
            continue

        def _int(text: str) -> int | None:
            try:
                return int(text)
            except (TypeError, ValueError):
                return None

        def _float(text: str) -> float | None:
            try:
                return float(text.replace(",", "."))
            except (TypeError, ValueError):
                return None

        out.append(
            ClubValue(
                club=club,
                squad_size=_int(cells[2]),
                average_age=_float(cells[3]),
                average_value_eur=parse_amount(cells[5]),
                total_value_eur=parse_amount(cells[6]),
            )
        )
    return out


def polite_scraper(min_interval: float = 2.0) -> PoliteSession:
    """A PoliteSession whose transport can clear Transfermarkt's challenge.

    The delay is deliberately slower than the project default. This is a site
    being scraped rather than a published data file being downloaded, and one
    page per competition-season is not a workload worth hurrying.
    """
    import cloudscraper

    return PoliteSession(min_interval=min_interval, _session=cloudscraper.create_scraper())


def fetch_season(
    competition_id: str, season_start: int, *, session: PoliteSession | None = None
) -> list[ClubValue]:
    if competition_id not in COMPETITIONS:
        raise TransfermarktError(f"no Transfermarkt code for {competition_id!r}")
    session = session or polite_scraper()
    url = season_url(COMPETITIONS[competition_id], season_start)
    html = session.get_bytes(url, suffix=".html").decode("utf-8", errors="replace")
    return parse_season(html)


def build(
    seasons: list[int], *, session: PoliteSession | None = None
) -> pd.DataFrame:
    """Every competition-season, as one long frame."""
    session = session or polite_scraper()
    rows: list[dict] = []
    for competition_id in COMPETITIONS:
        for season_start in seasons:
            try:
                values = fetch_season(competition_id, season_start, session=session)
            except Exception:
                log.exception(
                    "transfermarkt: %s %s failed", competition_id, season_start
                )
                continue
            for value in values:
                rows.append(
                    {
                        "competition_id": competition_id,
                        "season": f"{season_start}-{season_start + 1}",
                        "season_start_year": season_start,
                        "club": value.club,
                        "squad_size": value.squad_size,
                        "average_age": value.average_age,
                        "total_value_eur": value.total_value_eur,
                        "average_value_eur": value.average_value_eur,
                    }
                )
            log.info(
                "transfermarkt: %s %d — %d clubs",
                competition_id, season_start, len(values),
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
