import json

from iid_relative_value.cli import main


def test_cli_json(capsys):
    main(["analyze", "examples/analysis.json", "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert data["anchor"]["status"] == "ready"
    assert data["kelly"]["fractional_fractions"]
