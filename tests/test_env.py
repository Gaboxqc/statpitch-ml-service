"""Reading .env, which `.env.example` has always claimed happens."""
from __future__ import annotations

import os

from statpitch import env


def test_key_value_lines_are_parsed():
    assert env.parse("A=1\nB=two\n") == {"A": "1", "B": "two"}


def test_comments_and_blanks_are_skipped():
    assert env.parse("# a note\n\nA=1\n   \n# another\n") == {"A": "1"}


def test_surrounding_quotes_are_stripped():
    """A key pasted with quotes still works."""
    assert env.parse('A="secret"\nB=\'other\'\n') == {"A": "secret", "B": "other"}


def test_an_exported_line_is_accepted():
    """`export FOO=bar` is what people paste out of shell instructions."""
    assert env.parse("export STATPITCH_ODDS_API_KEY=abc\n") == {
        "STATPITCH_ODDS_API_KEY": "abc"
    }


def test_a_line_without_an_equals_costs_one_variable_not_the_run():
    assert env.parse("nonsense\nA=1\n") == {"A": "1"}


def test_a_value_containing_equals_survives():
    assert env.parse("A=b=c\n") == {"A": "b=c"}


def test_a_missing_file_is_a_no_op(tmp_path):
    """The file is optional by design."""
    assert env.load_dotenv(tmp_path / "nope") == {}


def test_values_reach_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("STATPITCH_TEST_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("STATPITCH_TEST_KEY=abc\n", encoding="utf-8")

    applied = env.load_dotenv(path)
    assert applied == {"STATPITCH_TEST_KEY": "abc"}
    assert os.environ["STATPITCH_TEST_KEY"] == "abc"


def test_the_real_environment_wins(tmp_path, monkeypatch):
    """CI and Render set secrets as real environment variables.

    A stale .env left in a working copy must not shadow them — that is how a
    deployment ends up using a developer's expired key.
    """
    monkeypatch.setenv("STATPITCH_TEST_KEY", "from-the-environment")
    path = tmp_path / ".env"
    path.write_text("STATPITCH_TEST_KEY=from-the-file\n", encoding="utf-8")

    assert env.load_dotenv(path) == {}
    assert os.environ["STATPITCH_TEST_KEY"] == "from-the-environment"


def test_override_is_available_and_says_what_it_does(tmp_path, monkeypatch):
    monkeypatch.setenv("STATPITCH_TEST_KEY", "from-the-environment")
    path = tmp_path / ".env"
    path.write_text("STATPITCH_TEST_KEY=from-the-file\n", encoding="utf-8")

    env.load_dotenv(path, override=True)
    assert os.environ["STATPITCH_TEST_KEY"] == "from-the-file"


def test_a_credential_never_reaches_the_log(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("STATPITCH_TEST_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("STATPITCH_TEST_KEY=super-secret-value\n", encoding="utf-8")

    with caplog.at_level("INFO"):
        env.load_dotenv(path)
    assert "STATPITCH_TEST_KEY" in caplog.text
    assert "super-secret-value" not in caplog.text


def test_the_example_file_documents_every_key_the_code_reads():
    """A credential nobody can discover is a credential nobody sets."""
    from statpitch import paths
    from statpitch.data import api_football, football_data_org, odds_api

    example = (paths.REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for module in (api_football, football_data_org, odds_api):
        assert module.ENV_KEY in example, module.__name__
