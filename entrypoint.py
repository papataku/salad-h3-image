#!/usr/bin/env python3
"""Entrypoint for the salad-h3-trial container.

One process, three phases (design: ~/workspace/salad-h3/README.md v2):
  1. A status server on :3000 answers from the first seconds:
     /health always 200 (probes never couple to the weight download),
     /ready 503 until phase 3 completes, /preflight diagnostic JSON,
     /log tail of the ComfyUI log for remote diagnosis.
  2. The three MiniMax-H3-Image weight files are downloaded directly into
     the ComfyUI models directories (resumable), then SHA-256 verified.
     A mismatch on a complete file deletes it and exits (fail closed);
     a truncated file keeps its .part so a restart resumes it. Verified
     files carry a .ok sidecar and are skipped on same-instance restarts.
  3. ComfyUI (pinned v0.45.0, /opt/comfyui-h3) starts on 127.0.0.1:8188.
     Once it answers, the status server proxies the native ComfyUI API
     and /ready flips to 200. If ComfyUI dies, this process exits so
     Salad restarts the container (weights persist on the instance).

The CUDA probe runs in a SUBPROCESS: importing torch in this long-lived
process would hold a CUDA context (hundreds of MiB) against the 32GB
card for the whole run and corrupt the smoke's VRAM measurements.

Only the Python standard library is used here, so the entrypoint does not
depend on the base image's package layout beyond python3 + the ComfyUI
install baked at build time.
"""
import hashlib, json, os, shutil, subprocess, sys, threading, time
import urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY = "/opt/comfyui-h3"
MODELS = os.path.join(COMFY, "models")
COMFY_PORT = 8188
SCRATCH_BYTES = 5 * 1000**3
LOG_PATH = "/tmp/comfyui.log"

STATE = {
    "phase": "boot",            # boot|downloading|starting|ready|error
    "error": None,
    "started_epoch": time.time(),
    "probe": {},
    "disk_free_bytes": None,
    "downloads": {},            # path -> {bytes_done, bytes_total, sha_ok}
}
LOCK = threading.Lock()

PROBE_SNIPPET = r"""
import json
try:
    import torch
    info = {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
    if info["cuda_available"]:
        info["device"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["capability"] = f"{cap[0]}.{cap[1]}"
    print(json.dumps(info))
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
"""


def set_phase(p, error=None):
    with LOCK:
        STATE["phase"] = p
        if error:
            STATE["error"] = error


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, part_path, expected, rec):
    """Resumable download of url into part_path. Returns when the .part
    file holds `expected` bytes (or raises after retries)."""
    done = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    if done > expected:            # corrupt overshoot: start clean
        os.remove(part_path)
        done = 0
    rec["bytes_done"] = done
    while done < expected:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "curl/8.5.0")
        if done:
            req.add_header("Range", f"bytes={done}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if done and r.status != 206:   # server ignored Range: restart
                    done = 0
                mode = "ab" if done else "wb"
                with open(part_path, mode) as f:
                    while done < expected:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        rec["bytes_done"] = done
        except Exception:
            if rec.get("retries", 0) >= 5:
                raise
            rec["retries"] = rec.get("retries", 0) + 1
            time.sleep(10)
            continue
        # Clean early EOF (no exception, still short): count it as a
        # retry too, otherwise this loop would spin without sleeping.
        if rec.get("retries", 0) >= 5:
            raise RuntimeError(f"download stalled at {done}/{expected}")
        rec["retries"] = rec.get("retries", 0) + 1
        time.sleep(10)


def verified(dest, w):
    sidecar = dest + ".ok"
    if not (os.path.exists(dest) and os.path.exists(sidecar)):
        return False
    try:
        return json.load(open(sidecar)).get("sha256") == w["sha256"] and \
            os.path.getsize(dest) == w["bytes"]
    except Exception:
        return False


def ensure_weights():
    weights = json.load(open(os.path.join(HERE, "weights.json")))
    extra = json.loads(os.environ.get("EXTRA_FILES_JSON") or "[]")
    for e in extra:
        assert set(e) == {"url", "path", "bytes", "sha256"} \
            and e["path"].startswith("loras/") and ".." not in e["path"] \
            and isinstance(e["bytes"], int) and len(e["sha256"]) == 64, \
            "bad EXTRA_FILES_JSON"
        weights = extra + weights
    # Free-space requirement counts only what still has to arrive:
    # unverified files minus the bytes already sitting in their .part.
    remaining = 0
    for w in weights:
        dest = os.path.join(MODELS, w["path"])
        rec = {"bytes_done": 0, "bytes_total": w["bytes"], "sha_ok": None}
        STATE["downloads"][w["path"]] = rec
        if verified(dest, w):
            rec.update(bytes_done=w["bytes"], sha_ok=True)
        else:
            part = dest + ".part"
            have = os.path.getsize(part) if os.path.exists(part) else 0
            remaining += max(0, w["bytes"] - have)
    free = shutil.disk_usage(MODELS).free
    STATE["disk_free_bytes"] = free
    if free < remaining + SCRATCH_BYTES:
        set_phase("error", f"free disk {free} < remaining {remaining} + scratch")
        print("FATAL insufficient disk for remaining downloads", flush=True)
        threading.Event().wait()   # stay up for /preflight diagnostics
    for w in weights:
        dest = os.path.join(MODELS, w["path"])
        rec = STATE["downloads"][w["path"]]
        if rec["sha_ok"]:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"
        download(w["url"], part, w["bytes"], rec)
        if sha256_file(part) != w["sha256"]:
            # Complete-size but wrong content = corruption: drop it so the
            # next boot re-downloads cleanly instead of resuming bad bytes.
            # (A truncated .part never reaches this branch: download()
            # only returns at full size.)
            os.remove(part)
            set_phase("error", f"sha256 mismatch for {w['path']}")
            print(f"FATAL sha256 mismatch {w['path']}", flush=True)
            os._exit(1)
        os.replace(part, dest)
        with open(dest + ".ok", "w") as f:
            json.dump({"sha256": w["sha256"], "bytes": w["bytes"]}, f)
        rec["sha_ok"] = True


def probe_torch():
    """Fail closed BEFORE the expensive download if this is not the
    expected Blackwell environment (round-3 delta review: the driver
    cannot read /preflight until the gateway's readiness gate opens, so
    the entrypoint itself must refuse a wrong environment)."""
    try:
        out = subprocess.run([sys.executable, "-c", PROBE_SNIPPET],
                             capture_output=True, text=True, timeout=180)
        STATE["probe"] = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:
        STATE["probe"] = {"error": f"{type(e).__name__}: {e}"}
    p = STATE["probe"]
    if not p.get("cuda_available") or p.get("capability") != "12.0" \
            or "5090" not in str(p.get("device", "")):
        set_phase("error", f"unexpected environment: {p}")
        print(f"FATAL unexpected environment {p}", flush=True)
        os._exit(1)


def start_comfy():
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT)],
        cwd=COMFY, stdout=open(LOG_PATH, "w"), stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            set_phase("error", f"comfyui exited rc={proc.returncode}")
            os._exit(1)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=5) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(3)
    set_phase("error", "comfyui did not answer in 900s")
    os._exit(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _phase(self):
        with LOCK:
            return STATE["phase"]

    def _route(self, method):
        path = self.path
        if path == "/health":
            return self._json(200, {"status": "ok", "phase": self._phase()})
        if path == "/ready":
            if self._phase() == "ready":
                return self._json(200, {"status": "ready"})
            return self._json(503, {"status": self._phase()})
        if path == "/preflight":
            with LOCK:
                snapshot = json.loads(json.dumps(STATE))
            for cand in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
                try:
                    snapshot["host_ram_peak_bytes"] = int(open(cand).read().strip())
                    break
                except Exception:
                    snapshot["host_ram_peak_bytes"] = None
            return self._json(200, snapshot)
        if path == "/log":
            try:
                with open(LOG_PATH, "rb") as f:
                    f.seek(max(0, os.path.getsize(LOG_PATH) - 65536))
                    tail = f.read()
            except Exception as e:
                tail = f"(no log yet: {e})".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(tail)))
            self.end_headers()
            return self.wfile.write(tail)
        if self._phase() != "ready":
            return self._json(503, {"status": self._phase()})
        return self._proxy(method)

    # Only the native endpoints the driver uses are proxied; everything
    # else (arbitrary paths, /ws, settings, model management) is refused.
    ALLOWED = ("/prompt", "/history", "/object_info", "/system_stats",
               "/view", "/upload/image", "/interrupt", "/queue")
    MAX_BODY = {"/upload/image": 32 * 1024 * 1024}
    DEFAULT_MAX_BODY = 2 * 1024 * 1024

    def _proxy(self, method):
        base = self.path.split("?", 1)[0]
        if not any(base == a or base.startswith(a + "/") for a in self.ALLOWED):
            return self._json(403, {"error": "path not allowed"})
        if method not in ("GET", "POST"):
            return self._json(405, {"error": "method not allowed"})
        length = int(self.headers.get("Content-Length") or 0)
        cap = self.MAX_BODY.get(base, self.DEFAULT_MAX_BODY)
        if length > cap:
            return self._json(413, {"error": f"body over {cap} bytes"})
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(f"http://127.0.0.1:{COMFY_PORT}{self.path}",
                                     data=body, method=method)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "connection"):
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
                self.send_response(r.status)
                ct = r.headers.get("Content-Type")
                if ct:
                    self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._json(502, {"error": f"proxy: {type(e).__name__}: {e}"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")


def main():
    os.makedirs(MODELS, exist_ok=True)
    STATE["disk_free_bytes"] = shutil.disk_usage(MODELS).free
    server = ThreadingHTTPServer(("0.0.0.0", 3000), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("status server on :3000", flush=True)
    probe_torch()
    set_phase("downloading")
    try:
        ensure_weights()
    except Exception as e:
        set_phase("error", f"download: {type(e).__name__}: {e}")
        print(f"FATAL download {e}", flush=True)
        os._exit(1)
    set_phase("starting")
    proc = start_comfy()
    set_phase("ready")
    print("READY: proxying native ComfyUI API", flush=True)
    rc = proc.wait()
    print(f"comfyui exited rc={rc}; exiting for restart", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
