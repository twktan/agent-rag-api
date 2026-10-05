"""Cross-platform behaviour (CI runs these on Windows too): UTF-8 I/O, .env with a BOM, eval tools."""

import csv
import json
import shutil

import pytest

from app.config import Settings
from eval import judge_calibration, update_readme
from eval.common import write_json
from eval.inspect_failures import failures
from eval.metrics import cohens_kappa
from eval.review_golden import grep_docs, show_items

NON_ASCII = "Leave: 20 days · κ ≥ 0.6 → ✅ “quoted” café"


def test_write_json_roundtrips_non_ascii(tmp_path):
    path = tmp_path / "out.json"
    write_json(path, {"answer": NON_ASCII})
    assert json.loads(path.read_text(encoding="utf-8"))["answer"] == NON_ASCII


def test_update_readme_preserves_unicode(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    shutil.copy(update_readme.README, readme)
    readme.write_text(readme.read_text(encoding="utf-8") + f"\n{NON_ASCII}\n", encoding="utf-8")
    monkeypatch.setattr(update_readme, "README", readme)
    update_readme.main()
    text = readme.read_text(encoding="utf-8")
    assert NON_ASCII in text and update_readme.START in text and "Retrieval MRR" in text


def test_env_file_with_bom_is_read(tmp_path, monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    env = tmp_path / ".env"
    env.write_bytes("API_KEYS=k1,k2\n".encode("utf-8-sig"))
    assert Settings(_env_file=env).api_key_set == {"k1", "k2"}


def test_cohens_kappa():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohens_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cohens_kappa([True], [True, False])


def test_review_tools_run(capsys):
    assert show_items("test", "adversarial", None) == 10
    assert grep_docs("halal") == 1
    assert "workplace_food_dining_wellness_program" in capsys.readouterr().out


def test_inspect_failures_flags_misroutes_and_wrong_answers():
    ok, wrong = {"correct": True, "faithful": True}, {"correct": False, "faithful": True}
    records = [
        {"id": "a", "expected_route": "rag", "route_decision": "rag", "judge_final": ok},
        {"id": "b", "expected_route": "rag", "route_decision": "direct"},
        {"id": "c", "expected_route": "rag", "route_decision": "rag", "judge_final": wrong},
    ]
    assert [(why, r["id"]) for why, r in failures(records)] == [("misrouted (rag -> direct)", "b"),
                                                               ("incorrect", "c")]


def test_judge_calibration_export_then_score(tmp_path, monkeypatch, capsys):
    results = tmp_path / "eval_test.json"
    records = [{"id": f"company-{i}", "question": f"q{i} · κ", "answer": f"a{i}",
                "judge_final": {"correct": i % 2 == 0, "faithful": True, "reason": ""}} for i in range(6)]
    results.write_text(json.dumps({"records": records}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(judge_calibration, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(judge_calibration, "CSV_PATH", tmp_path / "judge_calibration.csv")

    judge_calibration.export(results, n=6, seed=0)
    with open(tmp_path / "judge_calibration.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert "judge_correct" not in rows[0]  # blind labelling
    for row in rows:  # label exactly like the judge, but in Excel-style upper case, ;-separated
        row["human_correct"] = "TRUE" if int(row["id"].split("-")[1]) % 2 == 0 else "FALSE"
    with open(tmp_path / "judge_calibration.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=judge_calibration.FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    judge_calibration.score(results)
    assert "agreement=1.00  cohen_kappa=1.00" in capsys.readouterr().out
    saved = json.loads((tmp_path / "judge_calibration.json").read_text(encoding="utf-8"))
    assert saved["n"] == 6 and saved["disagreements"] == []
