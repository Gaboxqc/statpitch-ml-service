"""Competition taxonomy (Design §2) — the layer underneath everything else.

The one non-obvious piece here is `resolve_format`. Design §2 models `format` as a
single field on a competition, but real competitions are not that tidy: Copa del Rey
semi-finals are two-legged while its other rounds are not, the Champions League
replaced its group stage with a Swiss league phase in 2024-2025, and every
continental final is a single leg at a neutral venue. Since Design §5.3 branches
inference on `format`, resolving it wrongly means running the wrong sub-model —
so format resolution is (stage, season)-aware and covered by tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from statpitch import paths

CompetitionType = Literal["league", "domestic_cup", "continental_cup"]
Format = Literal[
    "round_robin",
    "single_leg_knockout",
    "two_leg_knockout",
    "swiss_league_phase",
]

VALID_TYPES: frozenset[str] = frozenset(("league", "domestic_cup", "continental_cup"))
VALID_FORMATS: frozenset[str] = frozenset(
    ("round_robin", "single_leg_knockout", "two_leg_knockout", "swiss_league_phase")
)


class TaxonomyError(ValueError):
    """Raised when the taxonomy file is malformed or a lookup misses."""


def season_start_year(season: str) -> int:
    """`"2024-2025"` -> 2024. Accepts `"2024-25"` and `"2024"` too.

    Seasons are compared by start year throughout the project; a season string is
    never compared lexically, because "2024-25" and "2024-2025" would then differ.
    """
    head = str(season).strip().split("-")[0]
    if not head.isdigit():
        raise TaxonomyError(f"unparseable season: {season!r}")
    year = int(head)
    if not 1880 <= year <= 2100:
        raise TaxonomyError(f"implausible season start year in {season!r}")
    return year


@dataclass(frozen=True, slots=True)
class Competition:
    competition_id: str
    name: str
    country: str
    competition_type: CompetitionType
    format: Format
    tier: int | None
    odds_coverage: bool
    admits_lower_tiers: bool
    extra_time: bool
    stage_formats: dict[str, str] = field(default_factory=dict)
    neutral_venue_stages: tuple[str, ...] = ()
    sources: dict[str, Any] = field(default_factory=dict)
    format_history: tuple[dict[str, Any], ...] = ()
    teams: int | None = None
    cross_league_bridge: bool = False

    # --- derived helpers -------------------------------------------------

    @property
    def football_data_code(self) -> str | None:
        """football-data.co.uk division code (E0, SP1, ...) or None if uncovered."""
        return self.sources.get("football_data_code")

    @property
    def understat_code(self) -> str | None:
        return self.sources.get("understat_code")

    @property
    def is_knockout(self) -> bool:
        return self.competition_type in ("domestic_cup", "continental_cup")

    def resolve_format(self, stage: str | None = None, season: str | int | None = None) -> Format:
        """The format actually in force for a given stage and season.

        Resolution order, most specific first:
          1. a `format_history` entry that covers this season (and stage, if it names one)
          2. `stage_formats[stage]`
          3. the competition default `format`
        """
        stage_key = _normalise_stage(stage)

        if season is not None:
            year = season_start_year(str(season))
            # Most recent matching historical rule wins, so sort by cutoff ascending
            # and take the first whose window still contains this season.
            for entry in sorted(
                self.format_history, key=lambda e: season_start_year(e["until_season"])
            ):
                if year > season_start_year(entry["until_season"]):
                    continue
                entry_stage = _normalise_stage(entry.get("stage"))
                if entry_stage is None or entry_stage == stage_key:
                    return _check_format(entry["format"], self.competition_id)

        if stage_key is not None and stage_key in self.stage_formats:
            return _check_format(self.stage_formats[stage_key], self.competition_id)

        return self.format

    def is_neutral_venue(self, stage: str | None) -> bool:
        return _normalise_stage(stage) in self.neutral_venue_stages

    def away_goals_rule_applies(self, season: str | int) -> bool:
        """UEFA abolished the away-goals rule from 2021-2022 (Design §4, two-leg state)."""
        return season_start_year(str(season)) < season_start_year(
            _registry().away_goals_abolished_from
        )


def _normalise_stage(stage: str | None) -> str | None:
    if stage is None:
        return None
    return str(stage).strip().lower().replace(" ", "_").replace("-", "_")


def _check_format(value: str, competition_id: str) -> Format:
    if value not in VALID_FORMATS:
        raise TaxonomyError(f"{competition_id}: unknown format {value!r}")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Registry:
    competitions: dict[str, Competition]
    away_goals_abolished_from: str

    def __getitem__(self, competition_id: str) -> Competition:
        try:
            return self.competitions[competition_id]
        except KeyError:
            raise TaxonomyError(
                f"unknown competition_id {competition_id!r}; "
                f"known: {sorted(self.competitions)}"
            ) from None

    def __contains__(self, competition_id: object) -> bool:
        return competition_id in self.competitions

    def __iter__(self):
        return iter(self.competitions.values())

    def __len__(self) -> int:
        return len(self.competitions)

    def of_type(self, competition_type: CompetitionType) -> list[Competition]:
        return [c for c in self if c.competition_type == competition_type]

    def with_odds_coverage(self) -> list[Competition]:
        """The competitions the Decision Layer is allowed to operate on."""
        return [c for c in self if c.odds_coverage]

    def by_football_data_code(self, code: str) -> Competition:
        for c in self:
            if c.football_data_code == code:
                return c
        raise TaxonomyError(f"no competition maps to football-data code {code!r}")


def load_registry(path: Path | None = None) -> Registry:
    """Parse and validate competitions.json. Raises TaxonomyError on any defect."""
    path = path or paths.competitions_file()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise TaxonomyError(f"taxonomy file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise TaxonomyError(f"taxonomy file is not valid JSON: {path}: {exc}") from None

    competitions: dict[str, Competition] = {}
    for row in raw.get("competitions", []):
        comp = _parse_competition(row)
        if comp.competition_id in competitions:
            raise TaxonomyError(f"duplicate competition_id {comp.competition_id!r}")
        competitions[comp.competition_id] = comp

    if not competitions:
        raise TaxonomyError(f"taxonomy file {path} declares no competitions")

    return Registry(
        competitions=competitions,
        away_goals_abolished_from=raw.get(
            "away_goals_rule_abolished_from_season", "2021-2022"
        ),
    )


def _parse_competition(row: dict[str, Any]) -> Competition:
    cid = row.get("competition_id")
    if not cid:
        raise TaxonomyError(f"competition row missing competition_id: {row}")

    ctype = row.get("competition_type")
    if ctype not in VALID_TYPES:
        raise TaxonomyError(f"{cid}: unknown competition_type {ctype!r}")

    fmt = _check_format(row.get("format", ""), cid)

    stage_formats = {
        _normalise_stage(k): _check_format(v, cid)
        for k, v in (row.get("stage_formats") or {}).items()
    }

    tier = row.get("tier")
    if tier is not None and (not isinstance(tier, int) or tier < 1):
        raise TaxonomyError(f"{cid}: tier must be a positive integer or null, got {tier!r}")

    odds_coverage = row.get("odds_coverage")
    if not isinstance(odds_coverage, bool):
        raise TaxonomyError(
            f"{cid}: odds_coverage must be an explicit boolean — it gates the whole "
            "Decision Layer, so it is never allowed to default"
        )

    for entry in row.get("format_history") or ():
        if "until_season" not in entry or "format" not in entry:
            raise TaxonomyError(f"{cid}: format_history entry needs until_season and format")
        season_start_year(entry["until_season"])
        _check_format(entry["format"], cid)

    return Competition(
        competition_id=cid,
        name=row.get("name", cid),
        country=row.get("country", ""),
        competition_type=ctype,  # type: ignore[arg-type]
        format=fmt,
        tier=tier,
        odds_coverage=odds_coverage,
        admits_lower_tiers=bool(row.get("admits_lower_tiers", False)),
        extra_time=bool(row.get("extra_time", False)),
        stage_formats=stage_formats,  # type: ignore[arg-type]
        neutral_venue_stages=tuple(
            _normalise_stage(s) for s in row.get("neutral_venue_stages") or ()
        ),  # type: ignore[arg-type]
        sources=row.get("sources") or {},
        format_history=tuple(row.get("format_history") or ()),
        teams=row.get("teams"),
        cross_league_bridge=bool(row.get("cross_league_bridge", False)),
    )


@lru_cache(maxsize=1)
def _registry() -> Registry:
    return load_registry()


def registry() -> Registry:
    """Process-wide cached registry (loaded once at API startup, Design §7)."""
    return _registry()


def get(competition_id: str) -> Competition:
    return registry()[competition_id]


def reset_cache() -> None:
    _registry.cache_clear()
