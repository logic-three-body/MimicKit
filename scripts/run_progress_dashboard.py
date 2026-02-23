#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import socketserver
import subprocess
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "output" / "train"

HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>MimicKit 训练进度看板</title>
  <style>
    :root{
      --bg-1:#f4efe6;
      --bg-2:#e3ece7;
      --ink:#1f2b2a;
      --muted:#4b5e5c;
      --ok:#1f7a52;
      --warn:#b45309;
      --bad:#b91c1c;
      --card:#ffffffd9;
      --line:#c5d3d0;
      --gpu0:#0f766e;
      --gpu1:#c2410c;
      --accent:#244b6b;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      color:var(--ink);
      font-family:"IBM Plex Sans","Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif;
      background:linear-gradient(130deg,var(--bg-1),var(--bg-2));
      min-height:100vh;
    }
    .wrap{
      max-width:1320px;
      margin:0 auto;
      padding:18px 18px 28px;
    }
    .head{
      display:flex;
      flex-wrap:wrap;
      gap:12px;
      align-items:center;
      justify-content:space-between;
      margin-bottom:14px;
    }
    h1{
      margin:0;
      font-size:24px;
      letter-spacing:.2px;
    }
    .sub{
      color:var(--muted);
      font-size:13px;
      margin-top:2px;
    }
    .toolbar{
      display:flex;
      flex-wrap:wrap;
      gap:8px;
      align-items:center;
    }
    input,button{
      border:1px solid var(--line);
      background:#fff;
      color:var(--ink);
      padding:8px 10px;
      border-radius:10px;
      font-size:13px;
    }
    button{
      cursor:pointer;
      background:var(--accent);
      color:#fff;
      border:none;
    }
    .grid{
      display:grid;
      grid-template-columns:repeat(12,minmax(0,1fr));
      gap:12px;
    }
    .card{
      background:var(--card);
      border:1px solid #ffffffaa;
      border-radius:14px;
      padding:12px 12px 10px;
      backdrop-filter: blur(2px);
      box-shadow: 0 2px 9px #1f2b2a17;
    }
    .kpi{
      grid-column:span 3;
    }
    .kpi .k{
      color:var(--muted);
      font-size:12px;
      margin-bottom:4px;
    }
    .kpi .v{
      font-size:24px;
      font-weight:700;
      line-height:1.1;
      letter-spacing:.2px;
    }
    .wide{
      grid-column:span 6;
    }
    .full{
      grid-column:1 / -1;
    }
    .mono{
      font-family:"IBM Plex Mono","Menlo","Consolas",monospace;
      font-size:12px;
      white-space:pre-wrap;
      line-height:1.45;
    }
    .label-ok{color:var(--ok);font-weight:700}
    .label-warn{color:var(--warn);font-weight:700}
    .label-bad{color:var(--bad);font-weight:700}
    .gpu-row{
      display:grid;
      grid-template-columns: 72px 1fr 95px;
      gap:10px;
      align-items:center;
      margin:6px 0;
    }
    .bar{
      height:12px;
      border-radius:999px;
      background:#e2ecea;
      overflow:hidden;
      border:1px solid #cad7d4;
    }
    .fill{
      height:100%;
      transition:width .25s linear;
    }
    .fill.g0{background:linear-gradient(90deg,#0ea5a0,var(--gpu0))}
    .fill.g1{background:linear-gradient(90deg,#f97316,var(--gpu1))}
    table{
      width:100%;
      border-collapse:collapse;
      font-size:12px;
    }
    th,td{
      border-bottom:1px solid #d8e3e0;
      text-align:left;
      padding:6px 4px;
      vertical-align:top;
    }
    th{
      color:#36514e;
      font-weight:700;
      background:#f4f9f8;
      position:sticky;
      top:0;
    }
    .table-wrap{
      max-height:320px;
      overflow:auto;
      border:1px solid #d7e4e1;
      border-radius:10px;
      background:#fff;
    }
    .chart-wrap{
      height:220px;
      border:1px solid #d7e4e1;
      border-radius:10px;
      background:#fff;
      padding:8px;
    }
    .foot{
      margin-top:12px;
      color:var(--muted);
      font-size:12px;
    }
    @media (max-width:1000px){
      .kpi{grid-column:span 6}
      .wide{grid-column:span 12}
    }
    @media (max-width:700px){
      .kpi{grid-column:span 12}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div>
        <h1>MimicKit 训练进度看板</h1>
        <div class="sub" id="subtitle">加载中...</div>
      </div>
      <div class="toolbar">
        <input id="rootInput" placeholder="root-out（可空）" />
        <button id="applyBtn">切换 Root</button>
        <button id="refreshBtn">立即刷新</button>
      </div>
    </div>

    <div class="grid">
      <div class="card kpi">
        <div class="k">流程进度</div>
        <div class="v" id="progressV">-</div>
        <div class="sub" id="progressSub">-</div>
      </div>
      <div class="card kpi">
        <div class="k">当前 Case</div>
        <div class="v" id="caseV">-</div>
        <div class="sub" id="caseSub">-</div>
      </div>
      <div class="card kpi">
        <div class="k">双卡健康</div>
        <div class="v" id="healthV">-</div>
        <div class="sub" id="healthSub">阈值: 双卡持续 >60%</div>
      </div>
      <div class="card kpi">
        <div class="k">运行进程</div>
        <div class="v" id="procV">-</div>
        <div class="sub" id="procSub">-</div>
      </div>

      <div class="card wide">
        <div class="k">GPU 实时利用率</div>
        <div id="gpuPanel"></div>
      </div>
      <div class="card wide">
        <div class="k">GPU 利用率时间线（最近样本）</div>
        <div class="chart-wrap"><canvas id="gpuChart" width="700" height="190"></canvas></div>
      </div>

      <div class="card full">
        <div class="k">当前 Case 尝试明细</div>
        <div class="table-wrap"><table id="attemptTable"></table></div>
      </div>

      <div class="card full">
        <div class="k">最近事件</div>
        <div class="mono" id="eventsBox">-</div>
      </div>
    </div>

    <div class="foot" id="foot">-</div>
  </div>

  <script>
    const refreshMs = 5000;
    const rootInput = document.getElementById("rootInput");
    const subtitle = document.getElementById("subtitle");
    const foot = document.getElementById("foot");
    let activeRoot = "";
    let chartHistory = [];

    function q(id){ return document.getElementById(id); }

    function clsByHealth(s){
      if (s === "good") return "label-ok";
      if (s === "warning") return "label-warn";
      return "label-bad";
    }

    function setText(id, val){ q(id).textContent = val; }

    function parseRootFromUrl(){
      const u = new URL(window.location.href);
      return u.searchParams.get("root") || "";
    }

    function applyRootToUrl(root){
      const u = new URL(window.location.href);
      if (root) u.searchParams.set("root", root);
      else u.searchParams.delete("root");
      history.replaceState({}, "", u.toString());
    }

    async function fetchStatus(){
      const root = parseRootFromUrl();
      const url = root ? `/api/status?root=${encodeURIComponent(root)}` : "/api/status";
      const r = await fetch(url, {cache:"no-store"});
      if (!r.ok){
        throw new Error(`HTTP ${r.status}`);
      }
      return await r.json();
    }

    function renderGpuPanel(gpus){
      const panel = q("gpuPanel");
      if (!gpus || !gpus.length){
        panel.innerHTML = "<div class='sub'>GPU 不可用</div>";
        return;
      }
      panel.innerHTML = "";
      gpus.forEach((g, idx) => {
        const row = document.createElement("div");
        row.className = "gpu-row";
        const pct = Math.max(0, Math.min(100, Number(g.util || 0)));
        const cls = idx === 0 ? "g0" : "g1";
        row.innerHTML = `
          <div><b>GPU${g.index}</b></div>
          <div class="bar"><div class="fill ${cls}" style="width:${pct}%"></div></div>
          <div>${pct}% | ${g.mem_used}MiB</div>
        `;
        panel.appendChild(row);
      });
    }

    function drawChart(points){
      const canvas = q("gpuChart");
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#e2ebe9";
      ctx.lineWidth = 1;
      for (let y=0; y<=100; y+=20){
        const py = h - (y / 100.0) * (h - 20) - 10;
        ctx.beginPath();
        ctx.moveTo(30, py);
        ctx.lineTo(w - 6, py);
        ctx.stroke();
        ctx.fillStyle = "#607775";
        ctx.font = "11px IBM Plex Mono, monospace";
        ctx.fillText(String(y), 6, py + 3);
      }

      const usable = points.slice(-120);
      if (usable.length < 2){ return; }

      const mapX = (i) => 30 + (i / (usable.length - 1)) * (w - 42);
      const mapY = (v) => h - (Math.max(0, Math.min(100, v)) / 100.0) * (h - 20) - 10;

      function drawLine(key, color){
        ctx.beginPath();
        usable.forEach((p, i) => {
          const x = mapX(i);
          const y = mapY(Number(p[key] || 0));
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      drawLine("u0", "#0f766e");
      drawLine("u1", "#c2410c");
    }

    function renderAttempts(attempts){
      const table = q("attemptTable");
      if (!attempts || !attempts.length){
        table.innerHTML = "<tr><td>暂无 attempts</td></tr>";
        return;
      }
      const cols = ["variant","num_envs","note","probe_ok","long_ok","test_ok","viz_ok","long_budget_elapsed_sec","final_ok"];
      let h = "<thead><tr>";
      cols.forEach(c => h += `<th>${c}</th>`);
      h += "</tr></thead><tbody>";
      attempts.slice().reverse().forEach(a => {
        h += "<tr>";
        cols.forEach(c => h += `<td>${a[c] ?? ""}</td>`);
        h += "</tr>";
      });
      h += "</tbody>";
      table.innerHTML = h;
    }

    function renderEvents(events){
      q("eventsBox").textContent = (events || []).join("\\n");
    }

    function appendRealtimePoint(status){
      const g = status.gpus || [];
      if (g.length >= 2){
        chartHistory.push({
          t: status.server_time || "",
          u0: Number(g[0].util || 0),
          u1: Number(g[1].util || 0),
        });
      }
      chartHistory = chartHistory.slice(-240);
    }

    function render(status){
      activeRoot = status.root_name || status.root_path || "";
      rootInput.value = parseRootFromUrl() || activeRoot;
      subtitle.textContent = `Root: ${activeRoot || "-"} | 最近刷新: ${status.server_time || "-"}`;

      const done = status.progress?.done ?? 0;
      const total = status.progress?.total ?? 0;
      const pct = total > 0 ? ((done / total) * 100).toFixed(1) : "0.0";
      setText("progressV", `${done}/${total} (${pct}%)`);
      setText("progressSub", `状态: ${status.progress?.status || "-"}`);

      setText("caseV", status.progress?.last_case || status.current_attempt?.case || "-");
      setText("caseSub", `attempt: ${status.current_attempt?.variant || "-"} e${status.current_attempt?.num_envs || "-" } | note: ${status.current_attempt?.note || "-"}`);

      const h = status.health || {};
      const hv = q("healthV");
      hv.className = clsByHealth(h.level || "warning");
      hv.textContent = h.label || "-";
      setText("healthSub", h.detail || "-");

      const p = status.processes || {};
      setText("procV", `${p.longcycle ? 1 : 0} / ${p.worker_count || 0}`);
      setText("procSub", `longcycle_pid=${p.longcycle_pid || "-"} worker_pid=${(p.worker_pids || []).join(",") || "-"}`);

      renderGpuPanel(status.gpus || []);
      appendRealtimePoint(status);
      const merged = (status.monitor_samples || []).concat(chartHistory);
      drawChart(merged);

      renderAttempts(status.current_case_attempts || []);
      renderEvents(status.events || []);

      foot.textContent = `monitor_log=${status.monitor_log || "-"} | runner_log=${status.runner_log || "-"} | history_points=${(status.monitor_samples || []).length}`;
    }

    async function tick(){
      try{
        const status = await fetchStatus();
        render(status);
      }catch(err){
        subtitle.textContent = `刷新失败: ${err.message}`;
      }
    }

    q("refreshBtn").addEventListener("click", tick);
    q("applyBtn").addEventListener("click", () => {
      const v = rootInput.value.trim();
      applyRootToUrl(v);
      tick();
    });

    setInterval(tick, refreshMs);
    tick();
  </script>
</body>
</html>
"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--root-out", default="", help="run root name or path; empty means auto-detect latest")
    ap.add_argument("--monitor-log", default="", help="optional monitor log path; empty means auto-detect latest")
    ap.add_argument("--history-size", type=int, default=180, help="max monitor samples returned to web")
    return ap.parse_args()


def resolve_root(root_arg: str):
    if root_arg:
        raw = Path(root_arg)
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(TRAIN_ROOT / root_arg)
            candidates.append(ROOT / root_arg)
            candidates.append(Path(root_arg))

        for p in candidates:
            if p.exists() and p.is_dir():
                return p.resolve()
        return None

    patterns = [
        "case_ultralong_alloc_*",
        "case_ultralong_*",
        "case_longcycle_*",
        "case_e2e_*",
        "case_gpu_alloc_*",
        "case_gpu_bench_*",
    ]
    roots = []
    for pat in patterns:
        roots.extend(TRAIN_ROOT.glob(pat))
    roots = [x for x in roots if x.is_dir()]
    if not roots:
        return None
    return sorted(roots, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def parse_tsv(path: Path):
    if not path.exists():
        return []
    try:
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    except Exception:
        return []


def read_gpu_snapshot():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return []

    rows = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "index": int(float(parts[0])),
                "util": int(float(parts[1])),
                "mem_used": int(float(parts[2])),
                "mem_total": int(float(parts[3])),
                "power": float(parts[4]),
            }
        )
    return rows


def read_process_state():
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except Exception:
        return {
            "longcycle": False,
            "longcycle_pid": None,
            "worker_count": 0,
            "worker_pids": [],
        }

    longcycle_pid = None
    worker_pids = []
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except Exception:
            continue
        cmd = parts[1] if len(parts) > 1 else ""

        if "scripts/run_case_longcycle.py" in cmd:
            longcycle_pid = pid
        if "mimickit/run.py" in cmd and "--mode train" in cmd:
            worker_pids.append(pid)

    return {
        "longcycle": longcycle_pid is not None,
        "longcycle_pid": longcycle_pid,
        "worker_count": len(worker_pids),
        "worker_pids": worker_pids[:8],
    }


def detect_monitor_log(root_path: Path, explicit: str):
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p if p.exists() else None

    if root_path is not None:
        name = root_path.name
        ts_match = re.search(r"(20\\d{6}_\\d{6})", name)
        if ts_match:
            ts = ts_match.group(1)
            for pat in [f"/tmp/*{ts}*.log"]:
                hits = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
                if hits:
                    return Path(hits[0])

    patterns = [
        "/tmp/mk_dualgpu_follow_until_high_*.log",
        "/tmp/mk_ultralong_alloc_monitor_*.log",
        "/tmp/mk_longcycle_monitor_*.log",
    ]
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat))
    if not hits:
        return None
    hits = sorted(hits, key=os.path.getmtime, reverse=True)
    return Path(hits[0])


def parse_monitor_samples(path: Path, limit: int):
    if not path or not path.exists():
        return []
    samples = []
    rx = re.compile(
        r"^(?P<ts>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) \\| .*?GPU0=(?P<u0>\\d+)%.*?GPU1=(?P<u1>\\d+)%",
        re.M,
    )
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return []
    for m in rx.finditer(text):
        samples.append(
            {
                "t": m.group("ts"),
                "u0": int(m.group("u0")),
                "u1": int(m.group("u1")),
            }
        )
    return samples[-limit:]


def tail_lines(path: Path, n=30):
    if not path or not path.exists():
        return []
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return []
    return lines[-n:]


def detect_health(gpus, attempt, monitor_samples):
    if len(gpus) < 2:
        return {
            "level": "critical",
            "label": "GPU 不可用",
            "detail": "nvidia-smi 无数据",
        }

    u0 = int(gpus[0].get("util", 0))
    u1 = int(gpus[1].get("util", 0))
    note = str((attempt or {}).get("note", ""))

    if u0 >= 60 and u1 >= 60:
        return {
            "level": "good",
            "label": "双卡高利用率",
            "detail": f"GPU0={u0}% GPU1={u1}%",
        }

    if note.endswith("nccl") or note == "long_nccl":
        return {
            "level": "critical",
            "label": "疑似通信异常",
            "detail": f"当前 note={note}",
        }

    if monitor_samples:
        recent = monitor_samples[-12:]
        gap = [abs(x["u0"] - x["u1"]) for x in recent]
        if gap and (sum(gap) / len(gap)) > 45:
            return {
                "level": "warning",
                "label": "双卡负载不均",
                "detail": "最近窗口负载差较大",
            }

    return {
        "level": "warning",
        "label": "利用率偏低",
        "detail": f"GPU0={u0}% GPU1={u1}%",
    }


def collect_status(config, root_arg_override=""):
    root_path = resolve_root(root_arg_override or config["root_out"])
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if root_path is None:
        return {
            "server_time": now,
            "root_name": "",
            "root_path": "",
            "progress": {"done": 0, "total": 0, "status": "no_root", "last_case": ""},
            "current_attempt": {},
            "current_case_attempts": [],
            "gpus": read_gpu_snapshot(),
            "processes": read_process_state(),
            "health": {"level": "critical", "label": "未发现运行目录", "detail": "请指定 --root-out 或等待目录创建"},
            "events": [],
            "monitor_log": "",
            "runner_log": "",
            "monitor_samples": [],
        }

    progress = read_json(root_path / "progress.json")
    progress = {
        "done": int(progress.get("done", 0) or 0),
        "total": int(progress.get("total", 0) or 0),
        "status": str(progress.get("status", "")),
        "last_case": str(progress.get("last_case", "")),
    }

    runs_dir = root_path / "runs"
    current_case = progress["last_case"]
    current_case_key = current_case.replace(".txt", "")
    attempts_path = runs_dir / current_case_key / "attempts.json"
    attempts = read_json(attempts_path)
    if not isinstance(attempts, list):
        attempts = []
    current_attempt = attempts[-1] if attempts else {}

    runner_log = Path(f"/tmp/{root_path.name}.log")
    if not runner_log.exists():
        candidates = sorted(glob.glob("/tmp/case_*longcycle*.log"), key=os.path.getmtime, reverse=True)
        runner_log = Path(candidates[0]) if candidates else None

    monitor_log = detect_monitor_log(root_path, config["monitor_log"])

    gpus = read_gpu_snapshot()
    monitor_samples = parse_monitor_samples(monitor_log, limit=config["history_size"])
    health = detect_health(gpus, current_attempt, monitor_samples)

    events = []
    events.extend([f"[RUNNER] {x}" for x in tail_lines(runner_log, n=12)])
    events.extend([f"[MONITOR] {x}" for x in tail_lines(monitor_log, n=12)])
    events = events[-24:]

    return {
        "server_time": now,
        "root_name": root_path.name,
        "root_path": str(root_path),
        "progress": progress,
        "current_attempt": current_attempt,
        "current_case_attempts": attempts,
        "best_rows": len(parse_tsv(root_path / "best_by_case.tsv")),
        "gpus": gpus,
        "processes": read_process_state(),
        "health": health,
        "events": events,
        "monitor_log": str(monitor_log) if monitor_log else "",
        "runner_log": str(runner_log) if runner_log else "",
        "monitor_samples": monitor_samples,
    }


def make_handler(config):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stdout.write(
                "%s - - [%s] %s\n"
                % (self.address_string(), self.log_date_time_string(), fmt % args)
            )

        def _send_json(self, obj, code=200):
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, text, code=200):
            raw = text.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send_html(HTML_PAGE)
                return
            if u.path == "/api/status":
                q = parse_qs(u.query)
                root_override = (q.get("root", [""])[0] or "").strip()
                status = collect_status(config, root_arg_override=root_override)
                self._send_json(status)
                return
            self._send_json({"error": "not found"}, code=404)

    return Handler


def main():
    args = parse_args()
    config = {
        "root_out": args.root_out.strip(),
        "monitor_log": args.monitor_log.strip(),
        "history_size": max(30, int(args.history_size)),
    }
    handler = make_handler(config)

    class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True

    with ThreadingTCPServer((args.host, args.port), handler) as httpd:
        base = f"http://{args.host}:{args.port}/"
        print(f"[dashboard] listen={base}")
        print(f"[dashboard] root_out={config['root_out'] or '(auto)'}")
        if config["monitor_log"]:
            print(f"[dashboard] monitor_log={config['monitor_log']}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
