"""Live pre-match odds ingestion tests (Plan §4 Phase A).

Offline, like the rest of the suite. The CSV below reproduces the real
`fixtures.csv` header — verified against the live file during development —
trimmed to the columns this module actually reads.
"""

from __future__ import annotations

import pandas as pd
import pytest

from statpitch import decision_config
from statpitch.data import football_data_live as live
from statpitch.decision import market_engine as me
from statpitch.models.dixon_coles import score_matrix

# --- fixtures -----------------------------------------------------------------

HEADER = (
    "Div,Date,Time,HomeTeam,AwayTeam,"
    "B365H,B365D,B365A,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,BFEH,BFED,BFEA,"
    "B365>2.5,B365<2.5,Max>2.5,Max<2.5,Avg>2.5,Avg<2.5,BFE>2.5,BFE<2.5,"
    "AHh,MaxAHH,MaxAHA,AvgAHH,AvgAHA,BFEAHH,BFEAHA,"
    "MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,BFECH,BFECD,BFECA"
)

#: Two real fixtures from the 21-24/08/2026 feed, plus one row in a division
#: outside the taxonomy (B1, Belgian first division) that must be dropped.
LIVE_CSV = f"""{HEADER}
E0,21/08/2026,20:00,Arsenal,Coventry,1.19,7.50,15.0,1.23,8.00,17.0,1.19,7.30,14.5,1.24,8.20,17.5,1.53,2.50,1.58,2.60,1.55,2.52,1.60,2.65,-2,1.99,1.76,1.95,1.72,2.02,1.80,,,,,,,,,
SP1,22/08/2026,20:30,Espanol,Real Madrid,7.00,4.50,1.40,8.20,4.80,1.44,7.36,4.59,1.41,8.40,4.90,1.46,1.65,2.10,1.73,2.20,1.68,2.11,1.78,2.24,1.25,1.90,1.98,1.84,1.91,1.92,2.02,,,,,,,,,
B1,22/08/2026,15:00,Waregem,Beveren,2.10,3.40,3.50,2.20,3.50,3.60,2.12,3.42,3.52,2.25,3.55,3.65,1.80,2.00,1.85,2.05,1.82,2.02,1.88,2.08,-0.25,1.95,1.90,1.92,1.88,1.97,1.93,,,,,,,,,
"""

#: A level handicap and a winter (GMT) kickoff, for the two edge cases that
#: cannot be exercised by the August rows above.
WINTER_CSV = f"""{HEADER}
E0,05/12/2026,15:00,Everton,Fulham,2.40,3.30,3.00,2.50,3.40,3.10,2.42,3.32,3.02,2.55,3.45,3.15,1.90,1.95,1.95,2.00,1.92,1.96,1.98,2.04,0,1.98,1.92,1.95,1.90,2.00,1.94,,,,,,,,,
"""

#: A row whose CLOSING columns are populated. Pre-match the feed leaves these
#: empty; this exists to pin what happens when it does not.
CLOSED_CSV = f"""{HEADER}
E0,21/08/2026,20:00,Arsenal,Coventry,1.19,7.50,15.0,1.23,8.00,17.0,1.19,7.30,14.5,1.24,8.20,17.5,1.53,2.50,1.58,2.60,1.55,2.52,1.60,2.65,-2,1.99,1.76,1.95,1.72,2.02,1.80,1.25,8.10,17.2,1.20,7.40,14.8,1.26,8.30,17.6
"""

#: Impossible prices: at or below the stake, which the archive carries as noise.
JUNK_CSV = f"""{HEADER}
E0,21/08/2026,20:00,Arsenal,Coventry,1.00,0.50,1.00,1.00,0.90,1.00,1.00,0.80,1.00,1.00,0.95,1.00,,,,,,,,,,,,,,,,,,,,,,,,
"""


def _csv(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def parsed(tmp_path):
    return live.parse(_csv(tmp_path, "fixtures.csv", LIVE_CSV), cid="20260823T1930Z")


# --- division filtering -------------------------------------------------------

def test_keeps_only_taxonomy_divisions(parsed):
    """B1 is priced by the publisher and is not in the taxonomy."""
    assert set(parsed["competition_id"]) == {"ENG.PL", "ESP.LALIGA"}
    assert "Waregem" not in set(parsed["fd_home"])


def test_unknown_divisions_only_yields_empty_frame(tmp_path):
    only_belgium = "\n".join([HEADER, LIVE_CSV.splitlines()[3]]) + "\n"
    frame = live.parse(_csv(tmp_path, "b1.csv", only_belgium))
    assert frame.empty
    assert list(frame.columns) == list(live.TIDY_COLUMNS)


# --- the time zone the column is actually in ----------------------------------

def test_kickoff_converted_from_uk_local_to_utc(parsed):
    """20:30 BST is 19:30 UTC. Reading it as UTC would be an hour early."""
    row = parsed[parsed["fd_home"] == "Espanol"].iloc[0]
    assert row["kickoff_utc"] == pd.Timestamp("2026-08-22 19:30:00")


def test_kickoff_in_winter_is_not_shifted(tmp_path):
    """The offset is seasonal, so a fixed -1h would be wrong in December."""
    frame = live.parse(_csv(tmp_path, "winter.csv", WINTER_CSV))
    assert frame["kickoff_utc"].iloc[0] == pd.Timestamp("2026-12-05 15:00:00")


def test_missing_time_yields_nat_not_midnight(tmp_path):
    text = LIVE_CSV.replace(",20:00,Arsenal", ",,Arsenal")
    frame = live.parse(_csv(tmp_path, "notime.csv", text))
    arsenal = frame[frame["fd_home"] == "Arsenal"]
    assert arsenal["kickoff_utc"].isna().all()
    assert arsenal["date"].iloc[0] == pd.Timestamp("2026-08-21")


# --- the market_engine contract -----------------------------------------------

def _engine_selections():
    config = decision_config.config().market_engine
    return me.derive(
        score_matrix(1.4, 1.1, rho=0.0, max_goals=config.matrix_max_goals),
        handicap_lines=tuple(config.asian_handicap_lines),
        totals_lines=tuple(config.total_goals_lines),
    )


def test_every_selection_key_exists_in_the_market_engine(parsed):
    """The join Phase B performs, asserted here rather than discovered there."""
    engine_keys = {s.key for s in _engine_selections()}
    live_keys = set(parsed["selection_key"].dropna())
    assert live_keys
    assert live_keys <= engine_keys, sorted(live_keys - engine_keys)


def test_line_agrees_with_the_market_engine_selection(parsed):
    lines = {s.key: s.line for s in _engine_selections()}
    for key, line in zip(parsed["selection_key"], parsed["line"], strict=True):
        if key in lines and lines[key] is not None and not pd.isna(line):
            assert float(line) == pytest.approx(lines[key]), key


def test_away_handicap_line_is_negated(parsed):
    """`AHh` is the home handicap; the away side is quoted at its negation."""
    row = parsed[
        (parsed["fd_home"] == "Espanol") & (parsed["selection"] == "ah_away")
    ].iloc[0]
    assert float(row["line"]) == pytest.approx(-1.25)
    assert row["selection_key"] == "ah_away_-1.25"
    assert row["odds_avg"] == pytest.approx(1.91)


def test_home_handicap_keeps_the_published_line(parsed):
    row = parsed[
        (parsed["fd_home"] == "Espanol") & (parsed["selection"] == "ah_home")
    ].iloc[0]
    assert float(row["line"]) == pytest.approx(1.25)
    assert row["selection_key"] == "ah_home_1.25"


def test_level_handicap_away_key_matches_the_engine_spelling(tmp_path):
    """`-0.0` is what `market_engine` emits, so it is what must be joined on."""
    frame = live.parse(_csv(tmp_path, "winter.csv", WINTER_CSV))
    away = frame[frame["selection"] == "ah_away"].iloc[0]
    assert away["selection_key"] == "ah_away_-0.0"
    assert away["selection_key"] in {s.key for s in _engine_selections()}


# --- price separation ---------------------------------------------------------

def test_avg_and_max_stay_in_separate_columns(parsed):
    """FR-16a: fair probability comes from consensus, the bet from the best price."""
    row = parsed[
        (parsed["fd_home"] == "Espanol") & (parsed["selection_key"] == "1x2_home")
    ].iloc[0]
    assert row["odds_avg"] == pytest.approx(7.36)
    assert row["odds_max"] == pytest.approx(8.20)
    assert row["odds_bfe"] == pytest.approx(8.40)


def test_pinnacle_is_absent_from_this_feed(parsed):
    """The sharp reference the +0.51% rule was defined on is not published here."""
    assert parsed["odds_pinnacle"].isna().all()


def test_impossible_prices_are_dropped(tmp_path):
    frame = live.parse(_csv(tmp_path, "junk.csv", JUNK_CSV))
    assert frame.empty


# --- snapshot separation ------------------------------------------------------

def test_empty_closing_columns_produce_no_close_rows(parsed):
    """Pre-match the C columns are blank, so there is no close to record."""
    assert set(parsed["snapshot"]) == {"preclose"}


def test_close_snapshot_does_not_borrow_the_preclose_exchange_price(tmp_path):
    """The bug this pins: a `close` row carrying `BFEH` is a fabricated close.

    Both ends of a CLV measurement would then come from one capture, and the
    number would look healthy while measuring nothing.
    """
    frame = live.parse(_csv(tmp_path, "closed.csv", CLOSED_CSV))
    close = frame[(frame["snapshot"] == "close") & (frame["selection_key"] == "1x2_home")]
    assert len(close) == 1
    assert close["odds_bfe"].iloc[0] == pytest.approx(1.26)   # BFECH, not BFEH (1.24)
    assert close["odds_avg"].iloc[0] == pytest.approx(1.20)   # AvgCH, not AvgH (1.19)


# --- club reconciliation ------------------------------------------------------

LIGUE1_OF = [
    "Olympique Lyonnais", "Paris FC", "Paris Saint-Germain FC",
    "Stade Brestois 29", "Stade Rennais FC 1901", "Toulouse FC",
]


LIGUE1_FD = ["Paris SG", "Paris FC", "Lyon", "Toulouse"]


def test_one_directional_matching_picks_the_wrong_paris_club():
    """The hazard, pinned so the mitigation below has something to mitigate.

    `Paris SG`'s only distinctive token is "paris", which it shares with a
    different club in the same division. Scored one way it resolves to `Paris FC`
    at full confidence.
    """
    assert live._unique_best("Paris SG", LIGUE1_OF) == "Paris FC"


def test_mutual_best_rejects_the_paris_pair(monkeypatch):
    """The reverse direction ties, and a tie is refused rather than broken.

    Checked with the alias table emptied, so the rejection is attributable to the
    matching rule rather than to curation covering for it.
    """
    assert live._unique_best("Paris FC", LIGUE1_FD) is None

    monkeypatch.setitem(live.CLUB_ALIASES, "FRA.LIGUE1", {})
    resolution = live.resolve_clubs(LIGUE1_FD, LIGUE1_OF, "FRA.LIGUE1")

    assert "Paris SG" in resolution.unmatched
    assert "Paris FC" in resolution.unmatched
    # The unambiguous clubs in the same call are unaffected.
    assert resolution.mapping["Toulouse"] == "Toulouse FC"


def test_curated_alias_answers_what_the_matcher_declined():
    resolution = live.resolve_clubs(
        ["Paris SG", "Paris FC", "Lyon", "Toulouse"], LIGUE1_OF, "FRA.LIGUE1"
    )
    assert resolution.mapping["Paris SG"] == "Paris Saint-Germain FC"
    assert resolution.mapping["Paris FC"] == "Paris FC"
    assert resolution.mapping["Lyon"] == "Olympique Lyonnais"
    # Unambiguous on tokens alone, so it is not in the alias table.
    assert resolution.mapping["Toulouse"] == "Toulouse FC"
    assert not resolution.unmatched


def test_curated_alias_wins_over_an_automatic_match():
    """Curation exists where the automatic rule was wrong; it must not be re-derived."""
    resolution = live.resolve_clubs(
        ["Barcelona"], ["FC Barcelona", "RCD Espanyol de Barcelona"], "ESP.LALIGA"
    )
    assert resolution.mapping["Barcelona"] == "FC Barcelona"
    assert "Barcelona" in resolution.curated


def test_unmatched_clubs_are_reported_not_guessed():
    resolution = live.resolve_clubs(
        ["Wolves"], ["Arsenal FC", "Chelsea FC"], "ENG.PL"
    )
    assert resolution.mapping == {}
    assert resolution.unmatched == ["Wolves"]
    assert resolution.coverage == 0.0


def test_matching_is_symmetric():
    """`Le Havre` and `Le Mans` share their only surviving token pattern."""
    pool = ["Le Havre AC", "Le Mans FC"]
    assert live._unique_best("Le Havre", pool) == "Le Havre AC"
    assert live._unique_best("Le Mans", pool) == "Le Mans FC"


# --- keying onto the fixture list ---------------------------------------------

@pytest.fixture
def fixtures():
    return pd.DataFrame(
        {
            "competition_id": ["ENG.PL", "ESP.LALIGA"],
            "fixture_id": ["ENG.PL|2026-27|Arsenal FC|Coventry City FC",
                           "ESP.LALIGA|2026-27|RCD Espanyol de Barcelona|Real Madrid CF"],
            "home_team": ["Arsenal FC", "RCD Espanyol de Barcelona"],
            "away_team": ["Coventry City FC", "Real Madrid CF"],
            "date": [pd.Timestamp("2026-08-21"), pd.Timestamp("2026-08-23")],
        }
    )


def test_fixture_ids_are_joined_not_rebuilt(parsed, fixtures):
    mapping = {
        competition: resolution.mapping
        for competition, resolution in live.resolve_all(parsed, fixtures).items()
    }
    keyed, stats = live.attach_fixture_ids(parsed, fixtures, mapping)
    assert stats["keyed"] == stats["priced"]
    assert set(keyed["fixture_id"]) == set(fixtures["fixture_id"])


def test_date_shift_is_reported_rather_than_rejected(parsed, fixtures):
    """The list holds a provisional matchday; the odds carry the confirmed day."""
    mapping = {
        competition: resolution.mapping
        for competition, resolution in live.resolve_all(parsed, fixtures).items()
    }
    keyed, _ = live.attach_fixture_ids(parsed, fixtures, mapping)
    espanyol = keyed[keyed["fd_home"] == "Espanol"]
    assert (espanyol["date_shift_days"] == -1).all()


def test_prices_for_an_unlisted_fixture_are_dropped_and_counted(parsed, fixtures):
    listed = fixtures.iloc[:1]
    mapping = {
        competition: resolution.mapping
        for competition, resolution in live.resolve_all(parsed, listed).items()
    }
    keyed, stats = live.attach_fixture_ids(parsed, listed, mapping)
    assert set(keyed["competition_id"]) == {"ENG.PL"}
    dropped = stats["unmapped_club"] + stats["unlisted"] + stats["already_played"]
    assert dropped == stats["priced"] - stats["keyed"]
    assert stats["keyed"] < stats["priced"]


def test_an_already_played_fixture_is_not_counted_as_a_coverage_gap(parsed, fixtures):
    """The feed's window is rolling; the fixture list holds only unplayed games.

    Counting the overlap as "unlisted" made the coverage floor fire at 47.4% on a
    healthy matchday capture, because openfootball had published results for the
    earlier half of the round.
    """
    # The club map is built against the full list, so a drop below is
    # attributable to the horizon rather than to an unmapped name.
    mapping = {
        competition: resolution.mapping
        for competition, resolution in live.resolve_all(parsed, fixtures).items()
    }
    # A list whose horizon starts after the Arsenal fixture on the 21st.
    later = fixtures[fixtures["date"] >= pd.Timestamp("2026-08-23")]
    _, stats = live.attach_fixture_ids(parsed, later, mapping)

    assert stats["already_played"] > 0
    assert stats["unlisted"] == 0
    # The floor's denominator excludes them, so coverage reads as complete.
    assert stats["keyed"] == stats["listable"]


def test_a_genuine_gap_inside_the_horizon_still_counts_as_unlisted(parsed, fixtures):
    """An unaliased club must not be excused by the already-played carve-out."""
    mapping = {
        competition: resolution.mapping
        for competition, resolution in live.resolve_all(parsed, fixtures).items()
    }
    # Keep the EARLIEST fixture, so the horizon covers everything priced, but
    # drop the Espanyol row so its prices have nothing to key onto. The clubs
    # still map, so the only possible attribution is `unlisted`.
    partial = fixtures[fixtures["competition_id"] == "ENG.PL"]
    _, stats = live.attach_fixture_ids(parsed, partial, mapping)

    assert stats["already_played"] == 0
    assert stats["unlisted"] > 0
    assert stats["keyed"] < stats["listable"]


# --- append-only discipline ---------------------------------------------------

def test_append_keeps_the_earlier_capture(tmp_path, parsed, fixtures):
    """Overwriting Friday's snapshot deletes the half of CLV that cannot be refetched."""
    destination = tmp_path / "live_odds.parquet"
    mapping = {
        competition: resolution.mapping
        for competition, resolution in live.resolve_all(parsed, fixtures).items()
    }
    first, _ = live.attach_fixture_ids(parsed, fixtures, mapping)
    live.append_snapshot(first, destination)

    second = first.copy()
    second["capture_id"] = "20260824T1200Z"
    second["odds_avg"] = second["odds_avg"] + 0.1
    _, appended = live.append_snapshot(second, destination)

    stored = pd.read_parquet(destination)
    assert appended == len(second)
    assert set(stored["capture_id"]) == {"20260823T1930Z", "20260824T1200Z"}
    assert len(stored) == 2 * len(first)


def test_reappending_the_same_capture_is_a_no_op(tmp_path, parsed, fixtures):
    """A collector re-run after a partial failure must not duplicate the ledger."""
    destination = tmp_path / "live_odds.parquet"
    mapping = {
        competition: resolution.mapping
        for competition, resolution in live.resolve_all(parsed, fixtures).items()
    }
    frame, _ = live.attach_fixture_ids(parsed, fixtures, mapping)
    live.append_snapshot(frame, destination)
    _, appended = live.append_snapshot(frame, destination)

    assert appended == 0
    assert len(pd.read_parquet(destination)) == len(frame)


def test_a_second_source_in_the_same_minute_is_not_treated_as_a_repeat(
    tmp_path, parsed, fixtures
):
    """The regression this granularity exists for.

    `capture_id` is the UTC minute, and `collect_odds_api.py` runs straight after
    `collect_live_odds.py` in one workflow step, so the two routinely share one.
    A capture-level check read the second as a re-run and dropped every row of
    it — which meant discarding every Pinnacle price, the keyless feed carrying
    none and the paid one being the only source of the reference the selection
    rule is defined on.
    """
    destination = tmp_path / "live_odds.parquet"
    mapping = {
        competition: resolution.mapping
        for competition, resolution in live.resolve_all(parsed, fixtures).items()
    }
    free, _ = live.attach_fixture_ids(parsed, fixtures, mapping)
    live.append_snapshot(free, destination)

    # Same minute, different selections — as measured, the two sources overlap
    # on nothing: the free feed publishes one league block, the paid one reaches
    # the cups and midweek.
    paid = free.copy()
    paid["selection_key"] = paid["selection_key"] + "_paid"
    paid["odds_pinnacle"] = 1.91
    _, appended = live.append_snapshot(paid, destination)

    stored = pd.read_parquet(destination)
    assert appended == len(paid)
    assert stored["odds_pinnacle"].notna().sum() == len(paid)


def test_a_duplicate_row_keeps_the_one_carrying_prices(tmp_path, parsed, fixtures):
    """A feed can list one match twice — once at a confirmed kickoff, once at a
    placeholder date — and both key to the same fixture, a fixture id being
    deliberately date-independent. Keeping the *first* would decide that on the
    API's row order.
    """
    destination = tmp_path / "live_odds.parquet"
    mapping = {
        competition: resolution.mapping
        for competition, resolution in live.resolve_all(parsed, fixtures).items()
    }
    frame, _ = live.attach_fixture_ids(parsed, fixtures, mapping)

    blank = frame.head(3).copy()
    for column in blank.columns:
        if column.startswith("odds_"):
            blank[column] = pd.NA
    # The empty copy first, which is the ordering the naive version got wrong.
    _, appended = live.append_snapshot(
        pd.concat([blank, frame.head(3)], ignore_index=True), destination
    )

    stored = pd.read_parquet(destination)
    assert appended == 3
    assert stored["odds_avg"].notna().all()


def test_a_row_already_on_file_is_not_appended_twice(tmp_path, parsed, fixtures):
    """Re-run safety survives the finer key: identity moved from the capture to
    the row, it did not go away."""
    destination = tmp_path / "live_odds.parquet"
    mapping = {
        competition: resolution.mapping
        for competition, resolution in live.resolve_all(parsed, fixtures).items()
    }
    frame, _ = live.attach_fixture_ids(parsed, fixtures, mapping)
    live.append_snapshot(frame, destination)

    half = frame.head(3).copy()
    half["odds_avg"] = half["odds_avg"] + 5.0  # a changed price is still the same row
    _, appended = live.append_snapshot(half, destination)
    assert appended == 0


# --- capture identity ---------------------------------------------------------

def test_capture_id_is_filename_safe_and_utc():
    from datetime import UTC, datetime

    cid = live.capture_id(datetime(2026, 8, 23, 19, 30, 45, tzinfo=UTC))
    assert cid == "20260823T1930Z"
    assert live.raw_path(cid).name == "fixtures_20260823T1930Z.csv"
