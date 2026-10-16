"""A page for labelling the judge worksheet by hand, one item at a time.

    python -m eval.label_ui            # opens http://127.0.0.1:8765

eval/human_labels.json has 30 answers, each with the passages it should be supported by. Reading that as JSON is
the reason labelling gets put off, so this shows one item at a time with the rubric beside it and writes each
choice straight back into the file. Then `python -m eval.label_for_kappa --score` compares you with the judge.

The judge's scores are never sent to the page, because seeing them first would pull your labels toward them and
the comparison would then say nothing. There are no suggested labels either. The labels have to be yours.

It uses only the standard library and listens on 127.0.0.1 alone. Because it writes a file in the repository, a
save is accepted only with a token the page was given when it loaded and from the page's own origin, so another web
page open in the same browser cannot post labels to it.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "human_labels.json"

# what the page may see of an item. Anything else, including any judge score, stays on the server
VISIBLE = ("id", "question", "answer", "expected_behavior", "evidence", "human_grounded", "human_correct", "human_note")
MAX_NOTE = 2000
_write_lock = threading.Lock()


def read_sheet(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_sheet(path: Path, sheet: dict) -> None:
    """Same format label_for_kappa writes, through a temporary file so a crash cannot leave half a worksheet."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sheet, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def progress(sheet: dict) -> dict:
    items = sheet["items"]
    both = sum(1 for i in items if i.get("human_grounded") is not None and i.get("human_correct") is not None)
    return {"total": len(items), "labelled": both}


def public_items(sheet: dict) -> list[dict]:
    return [{key: item.get(key) for key in VISIBLE} for item in sheet["items"]]


class LabelError(ValueError):
    pass


def apply_label(path: Path, payload: dict) -> dict:
    """Validate one save and write it. Reads the file again each time, so a save never overwrites newer work."""
    if not isinstance(payload, dict):
        raise LabelError("expected an object")
    for field in ("human_grounded", "human_correct"):
        if field in payload:
            value = payload[field]
            # True and False count as 1 and 0 in Python, so they are refused by name
            if isinstance(value, bool) or value not in (0, 1, None):
                raise LabelError(f"{field} must be 1, 0 or null")
    note = payload.get("human_note")
    if note is not None and (not isinstance(note, str) or len(note) > MAX_NOTE):
        raise LabelError(f"human_note must be text of at most {MAX_NOTE} characters")
    with _write_lock:
        sheet = read_sheet(path)
        match = [item for item in sheet["items"] if item["id"] == payload.get("id")]
        if len(match) != 1:
            raise LabelError("no such item")
        item = match[0]
        for field in ("human_grounded", "human_correct", "human_note"):
            if field in payload:
                item[field] = payload[field]
        write_sheet(path, sheet)
        return progress(sheet)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label the judge worksheet</title>
<style>
  :root { --bg:#fafaf8; --fg:#1d1d1b; --mute:#6b6b66; --card:#fff; --line:#dcdcd5; --accent:#2a5bd7; --yes:#1f7a3e; --no:#b3261e; --mark:#ffe9a8; }
  @media (prefers-color-scheme: dark) { :root { --bg:#161616; --fg:#ececea; --mute:#9a9a94; --card:#202020; --line:#3a3a36; --accent:#7aa2ff; --yes:#5ec283; --no:#ff8a80; --mark:#5a4a12; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
  main { max-width: 920px; margin: 0 auto; padding: 16px; }
  header { display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between; position:sticky; top:0; background:var(--bg); padding:10px 0; border-bottom:1px solid var(--line); z-index:2; }
  .bar { flex:1 1 200px; height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
  .bar > div { height:100%; background:var(--accent); width:0; transition:width .2s; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:14px 0; }
  .label { font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--mute); margin-bottom:6px; }
  .question { font-size:19px; font-weight:600; }
  .text { white-space:pre-wrap; overflow-wrap:anywhere; }
  .answer { max-height: 46vh; overflow:auto; }
  .passage { border-top:1px solid var(--line); padding-top:10px; margin-top:10px; }
  .passage:first-of-type { border-top:0; margin-top:0; padding-top:0; }
  .src { font-size:13px; color:var(--mute); margin-bottom:4px; }
  mark { background:var(--mark); color:inherit; padding:0 1px; border-radius:2px; }
  .rubric { font-size:14px; color:var(--mute); }
  .rubric b { color:var(--fg); }
  .row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:8px 0; }
  .row .name { width:100%; font-weight:600; }
  button { font:inherit; border:1px solid var(--line); background:var(--card); color:var(--fg); padding:9px 18px; border-radius:8px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button.on.yes { background:var(--yes); color:#fff; border-color:var(--yes); }
  button.on.no { background:var(--no); color:#fff; border-color:var(--no); }
  button.nav { padding:7px 14px; }
  input[type=search], textarea { width:100%; font:inherit; padding:8px 10px; border:1px solid var(--line); border-radius:8px; background:var(--card); color:var(--fg); }
  textarea { min-height:64px; }
  .keys { font-size:13px; color:var(--mute); }
  kbd { border:1px solid var(--line); border-radius:4px; padding:0 5px; font-size:12px; }
  .toast { font-size:13px; color:var(--yes); min-height:18px; }
  .done { border-color:var(--yes); }
  details > summary { cursor:pointer; color:var(--mute); font-size:14px; }
</style></head>
<body><main>
<header>
  <h1 id="where">Loading...</h1>
  <div class="bar" title="labelled"><div id="fill"></div></div>
  <div class="keys" id="count"></div>
</header>

<div class="card rubric">
  <details>
    <summary>The rubric (from the worksheet)</summary>
    <p><b>Grounded</b> = every claim in the answer is supported by the passages shown under "Evidence". One unsupported claim makes it a 0.</p>
    <p><b>Correct</b> = the answer actually answers the question. (If the expected behaviour is "refuse", correct means it refuses.)</p>
    <p>The judge's own scores are hidden on purpose, so they cannot pull your labels toward them. Judge only from what is on this page.</p>
  </details>
</div>

<div class="card"><div class="label">Question <span id="expected"></span></div><div class="question text" id="question"></div></div>
<div class="card"><div class="label">Answer</div><div class="answer text" id="answer"></div></div>

<div class="card">
  <div class="label">Evidence (the passages the answer should be supported by)</div>
  <input type="search" id="find" placeholder="Highlight a word or phrase in the evidence and the answer...">
  <div id="evidence"></div>
</div>

<div class="card" id="decide">
  <div class="row"><span class="name">Is every claim supported by the evidence? (grounded)</span>
    <button id="g1" class="yes">Yes, 1 <kbd>q</kbd></button><button id="g0" class="no">No, 0 <kbd>w</kbd></button></div>
  <div class="row"><span class="name">Does it answer the question? (correct)</span>
    <button id="c1" class="yes">Yes, 1 <kbd>a</kbd></button><button id="c0" class="no">No, 0 <kbd>s</kbd></button></div>
  <div class="label" style="margin-top:12px">Note (optional)</div>
  <textarea id="note" placeholder="Why, if it was a close call"></textarea>
  <div class="toast" id="toast"></div>
</div>

<div class="row">
  <button class="nav" id="prev">Back <kbd>&larr;</kbd></button>
  <button class="nav" id="next">Next <kbd>&rarr;</kbd></button>
  <button class="nav" id="todo">Next unlabelled <kbd>n</kbd></button>
</div>
<p class="keys">Each choice saves to <code>eval/human_labels.json</code> as you click. When both are set, it moves on to the next unlabelled item. Use Back to change one.</p>
<div class="card done" id="finished" hidden><b>All labelled.</b> Close this page, press Ctrl+C in the terminal, and run
  <code>python -m eval.label_for_kappa --score</code></div>
</main>
<script>
const TOKEN = "__TOKEN__";
let items = [], idx = 0, advanceTimer = null;
const $ = id => document.getElementById(id);

function done(it) { return it.human_grounded !== null && it.human_correct !== null; }

function highlight(el, text, term) {
  el.textContent = "";
  if (!term) { el.textContent = text; return; }
  const lower = text.toLowerCase(), t = term.toLowerCase();
  let at = 0, i;
  while ((i = lower.indexOf(t, at)) !== -1) {
    el.appendChild(document.createTextNode(text.slice(at, i)));
    const m = document.createElement("mark"); m.textContent = text.slice(i, i + t.length); el.appendChild(m);
    at = i + t.length;
  }
  el.appendChild(document.createTextNode(text.slice(at)));
}

function paintChoice(prefix, value) {
  $(prefix + "1").classList.toggle("on", value === 1);
  $(prefix + "0").classList.toggle("on", value === 0);
}

function render() {
  const it = items[idx], term = $("find").value.trim();
  const n = items.filter(done).length;
  $("where").textContent = `Item ${idx + 1} of ${items.length}  (id ${it.id})`;
  $("count").textContent = `${n} of ${items.length} labelled`;
  $("fill").style.width = (100 * n / items.length) + "%";
  $("expected").textContent = it.expected_behavior ? `(expected: ${it.expected_behavior})` : "";
  $("question").textContent = it.question;
  highlight($("answer"), it.answer, term);
  const ev = $("evidence"); ev.textContent = "";
  it.evidence.forEach((p, k) => {
    const box = document.createElement("div"); box.className = "passage";
    const src = document.createElement("div"); src.className = "src"; src.textContent = `[${k + 1}] ${p.source}, page ${p.page}`;
    const body = document.createElement("div"); body.className = "text"; highlight(body, p.text, term);
    box.append(src, body); ev.appendChild(box);
  });
  paintChoice("g", it.human_grounded); paintChoice("c", it.human_correct);
  $("note").value = it.human_note || "";
  $("finished").hidden = n < items.length;
  $("toast").textContent = "";
  window.scrollTo(0, 0);
}

async function save(fields) {
  const it = items[idx];
  const r = await fetch("/api/label", { method: "POST", headers: { "Content-Type": "application/json", "X-Label-Token": TOKEN },
                                        body: JSON.stringify({ id: it.id, ...fields }) });
  if (!r.ok) { $("toast").style.color = "var(--no)"; $("toast").textContent = "Not saved: " + (await r.text()); return false; }
  Object.assign(it, fields);
  $("toast").style.color = "var(--yes)"; $("toast").textContent = "Saved";
  return true;
}

async function choose(field, value) {
  clearTimeout(advanceTimer);
  if (!(await save({ [field]: value }))) return;
  paintChoice(field === "human_grounded" ? "g" : "c", value);
  $("count").textContent = `${items.filter(done).length} of ${items.length} labelled`;
  $("fill").style.width = (100 * items.filter(done).length / items.length) + "%";
  $("finished").hidden = !items.every(done);
  if (done(items[idx]) && !items.every(done)) advanceTimer = setTimeout(goTodo, 600);
}

function go(k) { clearTimeout(advanceTimer); idx = Math.max(0, Math.min(items.length - 1, k)); render(); }
function goTodo() {
  const order = [...items.keys()].filter(k => k > idx).concat([...items.keys()].filter(k => k <= idx));
  const k = order.find(k => !done(items[k])); if (k !== undefined) go(k);
}

$("g1").onclick = () => choose("human_grounded", 1); $("g0").onclick = () => choose("human_grounded", 0);
$("c1").onclick = () => choose("human_correct", 1);  $("c0").onclick = () => choose("human_correct", 0);
$("prev").onclick = () => go(idx - 1); $("next").onclick = () => go(idx + 1); $("todo").onclick = goTodo;
$("find").oninput = render;
let noteTimer; $("note").oninput = () => { clearTimeout(noteTimer); noteTimer = setTimeout(() => save({ human_note: $("note").value }), 500); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.ctrlKey || e.metaKey || e.altKey) return;
  const map = { q: () => choose("human_grounded", 1), w: () => choose("human_grounded", 0), a: () => choose("human_correct", 1),
                s: () => choose("human_correct", 0), ArrowLeft: () => go(idx - 1), ArrowRight: () => go(idx + 1), n: goTodo };
  if (map[e.key]) { e.preventDefault(); map[e.key](); }
});

fetch("/api/items").then(r => r.json()).then(data => {
  items = data.items; const first = items.findIndex(i => !done(i)); idx = first === -1 ? 0 : first; render();
});
</script></body></html>
"""


def make_handler(path: Path, token: str, port_holder: list[int]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "label-ui"

        def log_message(self, fmt, *args):  # nothing that could hold item text goes to the terminal
            return

        def _local(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if not self._local():
                return self._send(403, b"forbidden", "text/plain")
            if self.path == "/":
                return self._send(200, PAGE.replace("__TOKEN__", token).encode("utf-8"), "text/html; charset=utf-8")
            if self.path == "/api/items":
                sheet = read_sheet(path)
                body = json.dumps({"items": public_items(sheet), **progress(sheet)}).encode("utf-8")
                return self._send(200, body, "application/json")
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            origin_ok = self.headers.get("Origin") in (None, f"http://127.0.0.1:{port_holder[0]}", f"http://localhost:{port_holder[0]}")
            if not self._local() or not origin_ok or not secrets.compare_digest(self.headers.get("X-Label-Token", ""), token):
                return self._send(403, b"forbidden", "text/plain")
            if self.path != "/api/label":
                return self._send(404, b"not found", "text/plain")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 10_000:
                    raise LabelError("too large")
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = apply_label(path, payload)
            except (LabelError, json.JSONDecodeError, ValueError) as exc:
                return self._send(400, str(exc).encode("utf-8"), "text/plain")
            self._send(200, json.dumps(result).encode("utf-8"), "application/json")

    return Handler


def make_server(path: Path = DEFAULT_PATH, port: int = 8765) -> tuple[ThreadingHTTPServer, str]:
    """A server bound to 127.0.0.1 and the token it expects. Port 0 picks a free port."""
    token = secrets.token_urlsafe(24)
    holder = [port]
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(path, token, holder))
    holder[0] = server.server_address[1]
    return server, token


def main() -> None:
    parser = argparse.ArgumentParser(description="Label the judge worksheet in a browser")
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"No worksheet at {path}. Make one with: python -m eval.label_for_kappa --sample 30")
    server, _token = make_server(path, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    state = progress(read_sheet(path))
    print(f"{state['labelled']} of {state['total']} labelled. Open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state = progress(read_sheet(path))
        print(f"\n{state['labelled']} of {state['total']} labelled.")
        if state["labelled"] == state["total"]:
            print("All done. Next: python -m eval.label_for_kappa --score")


if __name__ == "__main__":
    main()
