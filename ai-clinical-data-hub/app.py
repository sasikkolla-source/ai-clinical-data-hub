"""
AI Clinical Data Hub - simple base version
------------------------------------------
- Stores patients and their vital-sign records (SQLite)
- REST API + a small dashboard (served at "/")
- "AI" insights: rule-based flagging of abnormal vitals, with an optional
  LLM summary if ANTHROPIC_API_KEY is set.

Run:
    pip install -r requirements.txt
    python app.py
Open http://127.0.0.1:5000

NOTE: Demo/educational use only. Use synthetic data, not real patient data.
A real clinical system needs authentication, audit logs, encryption and
regulatory compliance (HIPAA/GDPR/DPDP etc.).
"""
import os
import sqlite3
from datetime import datetime

from flask import Flask, jsonify, request, Response, g

DB_PATH = os.environ.get("HUB_DB", "clinical_hub.db")
app = Flask(__name__)


# ---------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            sex TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL REFERENCES patients(id),
            recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            heart_rate INTEGER,
            systolic INTEGER,
            diastolic INTEGER,
            temp_c REAL,
            spo2 INTEGER,
            glucose REAL,
            notes TEXT
        );
        """
    )
    db.commit()
    db.close()


# ---------------------------------------------------------------- insights
# (field, low, high, label, unit)
RULES = [
    ("heart_rate", 50, 110, "Heart rate", "bpm"),
    ("systolic", 90, 140, "Systolic BP", "mmHg"),
    ("diastolic", 60, 90, "Diastolic BP", "mmHg"),
    ("temp_c", 35.5, 37.8, "Temperature", "°C"),
    ("spo2", 94, 100, "SpO2", "%"),
    ("glucose", 70, 140, "Glucose", "mg/dL"),
]


def analyze_record(rec):
    """Return a list of flags for one record."""
    flags = []
    for field, low, high, label, unit in RULES:
        val = rec[field]
        if val is None:
            continue
        if val < low:
            flags.append({"field": field, "level": "low",
                          "message": f"{label} low: {val} {unit} (expected {low}-{high})"})
        elif val > high:
            flags.append({"field": field, "level": "high",
                          "message": f"{label} high: {val} {unit} (expected {low}-{high})"})
    return flags


def build_insights(patient, records):
    latest = records[0] if records else None
    flags = analyze_record(latest) if latest else []
    trend = None
    if len(records) >= 2:
        a, b = records[0]["heart_rate"], records[1]["heart_rate"]
        if a is not None and b is not None and abs(a - b) >= 20:
            trend = f"Heart rate changed by {a - b:+d} bpm since the previous reading."
    summary = f"{patient['name']}: {len(records)} record(s). "
    summary += "No abnormal values in the latest reading." if not flags else \
        f"{len(flags)} abnormal value(s) in the latest reading."
    result = {"summary": summary, "flags": flags, "trend": trend, "llm_summary": None}

    # Optional LLM summary (only if key + package are available)
    if os.environ.get("ANTHROPIC_API_KEY") and records:
        try:
            import anthropic
            client = anthropic.Anthropic()
            lines = [dict(r) for r in records[:5]]
            msg = client.messages.create(
                model=os.environ.get("HUB_MODEL", "claude-sonnet-5"),
                max_tokens=400,
                messages=[{"role": "user", "content":
                           "Summarize these recent vitals for a clinician in 3 short "
                           "sentences. Do not diagnose.\n" + str(lines)}],
            )
            result["llm_summary"] = msg.content[0].text
        except Exception as exc:  # keep the app working without the LLM
            result["llm_summary"] = f"(LLM summary unavailable: {exc})"
    return result


# --------------------------------------------------------------------- API
@app.get("/api/patients")
def list_patients():
    rows = get_db().execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.post("/api/patients")
def add_patient():
    data = request.get_json(force=True)
    if not data.get("name"):
        return jsonify({"error": "name is required"}), 400
    db = get_db()
    cur = db.execute("INSERT INTO patients (name, age, sex) VALUES (?, ?, ?)",
                     (data["name"], data.get("age"), data.get("sex")))
    db.commit()
    return jsonify({"id": cur.lastrowid}), 201


@app.get("/api/patients/<int:pid>/records")
def list_records(pid):
    rows = get_db().execute(
        "SELECT * FROM records WHERE patient_id=? ORDER BY id DESC", (pid,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["flags"] = analyze_record(r)
        out.append(d)
    return jsonify(out)


@app.post("/api/patients/<int:pid>/records")
def add_record(pid):
    data = request.get_json(force=True)
    fields = ["heart_rate", "systolic", "diastolic", "temp_c", "spo2", "glucose", "notes"]
    vals = [data.get(f) or None for f in fields]
    db = get_db()
    if not db.execute("SELECT 1 FROM patients WHERE id=?", (pid,)).fetchone():
        return jsonify({"error": "patient not found"}), 404
    db.execute(
        "INSERT INTO records (patient_id, recorded_at, heart_rate, systolic, diastolic,"
        " temp_c, spo2, glucose, notes) VALUES (?,?,?,?,?,?,?,?,?)",
        (pid, datetime.now().strftime("%Y-%m-%d %H:%M"), *vals))
    db.commit()
    return jsonify({"ok": True}), 201


@app.get("/api/patients/<int:pid>/insights")
def insights(pid):
    db = get_db()
    patient = db.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    if not patient:
        return jsonify({"error": "patient not found"}), 404
    records = db.execute(
        "SELECT * FROM records WHERE patient_id=? ORDER BY id DESC", (pid,)).fetchall()
    return jsonify(build_insights(patient, records))


# --------------------------------------------------------------- dashboard
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Clinical Data Hub</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f4f6f9;color:#1c2330}
 header{background:#0f4c81;color:#fff;padding:14px 24px;font-size:20px;font-weight:600}
 main{display:grid;grid-template-columns:300px 1fr;gap:16px;padding:16px;max-width:1100px;margin:auto}
 @media(max-width:800px){main{grid-template-columns:1fr}}
 .card{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 3px #0001;margin-bottom:16px}
 h3{margin:0 0 10px;font-size:15px}
 input,select,textarea,button{font:inherit;padding:7px;margin:3px 0;width:100%;box-sizing:border-box;
   border:1px solid #c9d1dc;border-radius:6px}
 button{background:#0f4c81;color:#fff;border:0;cursor:pointer}
 .row{display:grid;grid-template-columns:1fr 1fr;gap:6px}
 .patient{padding:8px;border-radius:6px;cursor:pointer;border:1px solid transparent}
 .patient:hover,.patient.active{background:#e8f0f9;border-color:#0f4c81}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{padding:6px;border-bottom:1px solid #e3e8ef;text-align:left}
 .flag{background:#fdecea;color:#b3261e;border-radius:4px;padding:3px 7px;margin:2px 0;display:block;font-size:13px}
 .ok{color:#1b7f3b}
 .muted{color:#6b7686;font-size:13px}
</style></head><body>
<header>AI Clinical Data Hub</header>
<main>
 <div>
  <div class="card"><h3>Add patient</h3>
   <input id="pname" placeholder="Name">
   <div class="row"><input id="page" type="number" placeholder="Age">
    <select id="psex"><option>F</option><option>M</option><option>Other</option></select></div>
   <button onclick="addPatient()">Add</button></div>
  <div class="card"><h3>Patients</h3><div id="plist" class="muted">None yet</div></div>
 </div>
 <div>
  <div class="card"><h3 id="title">Select a patient</h3>
   <div id="insights" class="muted">AI insights appear here.</div></div>
  <div class="card" id="formcard" style="display:none"><h3>New vitals record</h3>
   <div class="row"><input id="hr" type="number" placeholder="Heart rate (bpm)">
    <input id="temp" type="number" step="0.1" placeholder="Temp (°C)"></div>
   <div class="row"><input id="sys" type="number" placeholder="Systolic BP">
    <input id="dia" type="number" placeholder="Diastolic BP"></div>
   <div class="row"><input id="spo2" type="number" placeholder="SpO2 (%)">
    <input id="glu" type="number" placeholder="Glucose (mg/dL)"></div>
   <textarea id="notes" placeholder="Notes" rows="2"></textarea>
   <button onclick="addRecord()">Save record</button></div>
  <div class="card"><h3>History</h3><div id="records" class="muted">-</div></div>
 </div>
</main>
<script>
let current = null;
const $ = id => document.getElementById(id);
const api = (url, opts) => fetch(url, opts).then(r => r.json());
const post = (url, body) => api(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function loadPatients() {
  const ps = await api('/api/patients');
  $('plist').innerHTML = ps.length ? ps.map(p =>
    `<div class="patient ${current===p.id?'active':''}" onclick="select(${p.id}, '${esc(p.name).replace(/'/g,"")}')">
      <b>${esc(p.name)}</b><br><span class="muted">${esc(p.age ?? '?')} y · ${esc(p.sex ?? '')}</span></div>`).join('') : 'None yet';
}
async function addPatient() {
  if (!$('pname').value.trim()) return;
  await post('/api/patients', {name: $('pname').value.trim(), age: +$('page').value || null, sex: $('psex').value});
  $('pname').value = ''; $('page').value = '';
  loadPatients();
}
async function select(id, name) {
  current = id; $('title').textContent = name; $('formcard').style.display = 'block';
  loadPatients(); refresh();
}
async function refresh() {
  const [recs, ins] = await Promise.all([api(`/api/patients/${current}/records`), api(`/api/patients/${current}/insights`)]);
  let html = `<div>${esc(ins.summary)}</div>`;
  html += ins.flags.length ? ins.flags.map(f => `<span class="flag">⚠ ${esc(f.message)}</span>`).join('')
                           : '<div class="ok">✓ Latest vitals within expected ranges</div>';
  if (ins.trend) html += `<div>${esc(ins.trend)}</div>`;
  if (ins.llm_summary) html += `<hr><div>${esc(ins.llm_summary)}</div>`;
  $('insights').innerHTML = html;
  $('records').innerHTML = recs.length ? `<table><tr><th>Time</th><th>HR</th><th>BP</th><th>Temp</th><th>SpO2</th><th>Glu</th><th>Flags</th></tr>` +
    recs.map(r => `<tr><td>${esc(r.recorded_at)}</td><td>${esc(r.heart_rate)}</td><td>${esc(r.systolic)}/${esc(r.diastolic)}</td>
      <td>${esc(r.temp_c)}</td><td>${esc(r.spo2)}</td><td>${esc(r.glucose)}</td><td>${r.flags.length || '-'}</td></tr>`).join('') + '</table>' : 'No records yet';
}
async function addRecord() {
  await post(`/api/patients/${current}/records`, {
    heart_rate: +$('hr').value, temp_c: +$('temp').value, systolic: +$('sys').value,
    diastolic: +$('dia').value, spo2: +$('spo2').value, glucose: +$('glu').value, notes: $('notes').value});
  ['hr','temp','sys','dia','spo2','glu','notes'].forEach(i => $(i).value = '');
  refresh();
}
loadPatients();
</script></body></html>"""


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
