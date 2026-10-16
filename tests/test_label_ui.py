"""The labelling page: it saves labels, keeps the judge's scores away from you, and refuses anything but its own page."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest

from eval.label_ui import LabelError, apply_label, make_server, make_server_preferring, progress, public_items, read_sheet


def worksheet(n=3) -> dict:
    return {
        "_instructions": "label them",
        "source_metrics": "metrics_n100.json",
        "binary_threshold": 0.8,
        "seed": 1,
        "items": [
            {
                "id": 10 + i,
                "question": f"question {i}",
                "answer": f"answer {i}",
                "expected_behavior": "answer",
                "evidence": [{"source": "a.pdf", "page": 1, "text": "passage"}],
                "human_grounded": None,
                "human_correct": None,
                "human_note": "",
                "_judge_scores_hidden": True,
                # something that must never reach the page
                "judge_groundedness": 0.97,
            }
            for i in range(n)
        ],
    }


@pytest.fixture
def sheet_path(tmp_path) -> Path:
    path = tmp_path / "human_labels.json"
    path.write_text(json.dumps(worksheet(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def running(sheet_path):
    server, token = make_server(sheet_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, token, sheet_path
    server.shutdown()
    server.server_close()


def post(base, token, body, **headers):
    return httpx.post(base + "/api/label", json=body, headers={"X-Label-Token": token, **headers})


# ---- what the page is allowed to see ----


def test_the_page_never_receives_the_judges_scores():
    shown = public_items(worksheet())
    assert all("judge_groundedness" not in item and "_judge_scores_hidden" not in item for item in shown)
    assert {"id", "question", "answer", "evidence", "human_grounded", "human_correct", "human_note"} <= set(shown[0])


def test_the_items_endpoint_hides_them_over_http_too(running):
    base, _token, _ = running
    text = httpx.get(base + "/api/items").text
    assert "0.97" not in text and "judge" not in text.lower()
    assert json.loads(text)["total"] == 3


def test_the_page_itself_loads_and_carries_a_token(running):
    base, token, _ = running
    page = httpx.get(base + "/")
    assert page.status_code == 200 and token in page.text
    assert "suggest" not in page.text.lower(), "the page must not offer labels"


# ---- saving ----


def test_a_label_is_written_to_the_file_and_counted(running):
    base, token, path = running
    assert post(base, token, {"id": 11, "human_grounded": 1, "human_correct": 0}).json() == {"total": 3, "labelled": 1}
    item = [i for i in read_sheet(path)["items"] if i["id"] == 11][0]
    assert (item["human_grounded"], item["human_correct"]) == (1, 0)


def test_an_item_counts_as_labelled_only_when_both_are_set(running):
    base, token, path = running
    assert post(base, token, {"id": 10, "human_grounded": 1}).json()["labelled"] == 0
    assert post(base, token, {"id": 10, "human_correct": 1}).json()["labelled"] == 1


def test_a_label_can_be_changed_and_cleared(running):
    base, token, path = running
    post(base, token, {"id": 10, "human_grounded": 1, "human_correct": 1})
    post(base, token, {"id": 10, "human_grounded": 0})
    post(base, token, {"id": 10, "human_correct": None})
    item = read_sheet(path)["items"][0]
    assert (item["human_grounded"], item["human_correct"]) == (0, None)


def test_a_note_is_saved(running):
    base, token, path = running
    post(base, token, {"id": 12, "human_note": "the page number is wrong"})
    assert read_sheet(path)["items"][2]["human_note"] == "the page number is wrong"


def test_saving_one_item_leaves_every_other_field_and_item_alone(running):
    base, token, path = running
    before = read_sheet(path)
    post(base, token, {"id": 11, "human_grounded": 1})
    after = read_sheet(path)
    after["items"][1]["human_grounded"] = None
    assert after == before, "a save changed something other than the label"
    assert after["items"][0]["judge_groundedness"] == 0.97, "the hidden score must survive a save"


def test_the_file_is_written_in_the_same_format_as_the_scoring_script_writes(running):
    base, token, path = running
    post(base, token, {"id": 10, "human_grounded": 1, "human_correct": 1})
    text = path.read_text(encoding="utf-8")
    assert text == json.dumps(read_sheet(path), indent=2, ensure_ascii=False)


def test_a_stale_page_cannot_overwrite_labels_made_since(sheet_path):
    # two saves for different items in turn, each reading the file fresh, keep both
    apply_label(sheet_path, {"id": 10, "human_grounded": 1, "human_correct": 1})
    apply_label(sheet_path, {"id": 11, "human_grounded": 0, "human_correct": 0})
    assert progress(read_sheet(sheet_path))["labelled"] == 2


def test_many_saves_at_once_do_not_corrupt_the_file(sheet_path):
    errors = []

    def work(i):
        try:
            for _ in range(15):
                apply_label(sheet_path, {"id": 10 + (i % 3), "human_grounded": i % 2, "human_correct": 1})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert progress(read_sheet(sheet_path))["total"] == 3  # still valid JSON with all three items


# ---- what it refuses ----


@pytest.mark.parametrize(
    "body",
    [
        {"id": 10, "human_grounded": 2},
        {"id": 10, "human_grounded": "yes"},
        {"id": 10, "human_correct": True},
        {"id": 999, "human_grounded": 1},
        {"human_grounded": 1},
        {"id": 10, "human_note": "x" * 2001},
        {"id": 10, "human_note": 5},
    ],
)
def test_bad_labels_are_refused_and_change_nothing(running, body):
    base, token, path = running
    before = read_sheet(path)
    assert post(base, token, body).status_code == 400
    assert read_sheet(path) == before


def test_a_save_with_no_token_or_a_wrong_one_is_refused(running):
    base, token, path = running
    before = read_sheet(path)
    assert httpx.post(base + "/api/label", json={"id": 10, "human_grounded": 1}).status_code == 403
    assert post(base, "not-the-token", {"id": 10, "human_grounded": 1}).status_code == 403
    assert read_sheet(path) == before


def test_a_save_from_another_websites_page_is_refused_even_with_the_token(running):
    base, token, path = running
    before = read_sheet(path)
    assert post(base, token, {"id": 10, "human_grounded": 1}, Origin="https://evil.example").status_code == 403
    assert read_sheet(path) == before


def test_a_request_with_another_host_name_is_refused(running):
    base, token, _ = running
    assert httpx.get(base + "/api/items", headers={"Host": "evil.example"}).status_code == 403


def test_an_oversized_body_is_refused(running):
    base, token, _ = running
    assert post(base, token, {"id": 10, "human_note": "x" * 20000}).status_code == 400


def test_unknown_paths_are_404(running):
    base, token, _ = running
    assert httpx.get(base + "/nope").status_code == 404
    assert httpx.post(base + "/other", json={}, headers={"X-Label-Token": token}).status_code == 404


def test_only_one_mistake_type_is_raised_for_bad_input(sheet_path):
    with pytest.raises(LabelError):
        apply_label(sheet_path, [1, 2, 3])


def test_when_the_preferred_port_is_taken_a_free_one_is_used_instead(sheet_path):
    import socket

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    taken = blocker.getsockname()[1]
    try:
        server, token, fell_back = make_server_preferring(sheet_path, taken)
        try:
            assert fell_back and server.server_address[1] != taken
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            assert httpx.get(base + "/api/items").json()["total"] == 3
            # the origin check follows the port that was really used, so saving still works
            ok = httpx.post(base + "/api/label", json={"id": 10, "human_grounded": 1},
                            headers={"X-Label-Token": token, "Origin": base})
            assert ok.status_code == 200
        finally:
            server.shutdown()
            server.server_close()
    finally:
        blocker.close()


def test_a_free_preferred_port_is_used_as_asked(sheet_path):
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    server, _token, fell_back = make_server_preferring(sheet_path, free)
    try:
        assert not fell_back and server.server_address[1] == free
    finally:
        server.server_close()
