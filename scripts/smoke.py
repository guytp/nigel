"""Post-install smoke test for Nigel.

Runs on the Pi right after bootstrap. Exercises each hardware subsystem
directly (no MCP, no OpenAI, no network) and prints clear pass/fail.

If this passes, the hardware side is wired correctly. If it fails, the
specific line tells you what's wrong before you waste time wondering why
Claude Code / the voice agent can't talk to Nigel.

    sudo systemctl stop picrawler-mcp  # only one process can open the camera
    cd /home/pi/nigel
    .venv/bin/python -m scripts.smoke
    sudo systemctl start picrawler-mcp
"""

from __future__ import annotations

import sys
import time
import traceback


GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def bad(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def section(msg: str) -> None:
    print(f"\n{msg}")


def main() -> int:
    failures = 0

    section("Loading hardware backend")
    try:
        from mcp_picrawler.hardware import BUILTIN_ACTIONS, get_hardware
    except Exception as e:
        bad(f"failed to import mcp_picrawler.hardware: {e}")
        traceback.print_exc()
        return 1

    hw = get_hardware()
    if hw.kind == "mock":
        bad("hardware backend is 'mock' — SunFounder libs not importable")
        warn("likely causes: picrawler/robot_hat/vilib not installed, OR a reboot")
        warn("                is needed after SunFounder's installers enabled I2C/SPI")
        return 1
    ok(f"hardware backend: {hw.kind}")

    section("Ultrasonic sensor")
    try:
        for _ in range(3):
            cm = hw.read_distance_cm()
            time.sleep(0.1)
        if cm < 0:
            bad(f"ultrasonic returned sentinel {cm} — check D2/D3 wiring")
            failures += 1
        elif cm > 400:
            warn(f"ultrasonic {cm}cm — suspicious (nothing in front?), but responsive")
        else:
            ok(f"ultrasonic: {cm}cm")
    except Exception as e:
        bad(f"ultrasonic read failed: {e}")
        failures += 1

    section("Camera")
    try:
        frame = hw.latest_frame_bgr(retry_budget_s=5.0)
        ok(f"camera frame shape {frame.shape} dtype {frame.dtype}")
        jpeg = hw.snapshot_jpeg()
        ok(f"JPEG encode ok ({len(jpeg)} bytes)")
    except Exception as e:
        bad(f"camera failed: {e}")
        failures += 1

    section("MJPEG web stream")
    try:
        import urllib.request

        url = "http://127.0.0.1:9000/mjpg.jpg"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = r.read(4096)
            if data[:2] == b"\xff\xd8":
                ok(f"mjpg.jpg served on :9000")
            else:
                warn(f"mjpg.jpg responded but content doesn't look like JPEG")
    except Exception as e:
        warn(f"could not reach :9000/mjpg.jpg ({e}) — vilib display server may not be up")

    section("Servos (tiny test move)")
    try:
        print("  asking Nigel to 'ready' (small, safe stance)...")
        hw.do_action("ready", steps=1, speed=60)
        ok("servos responded")
        time.sleep(0.3)
        hw.do_action("sit", steps=1, speed=60)
        ok("sit ok")
    except Exception as e:
        bad(f"servo command failed: {e}")
        failures += 1

    section("Vision toggles")
    from mcp_picrawler.hardware import DETECTIONS

    for feat in DETECTIONS:
        try:
            hw.set_vision(feat, True)
            hw.set_vision(feat, False)
            ok(f"vision.{feat} toggle ok")
        except Exception as e:
            bad(f"vision.{feat}: {e}")
            failures += 1

    section("TTS (listen for a beep/voice)")
    try:
        hw.speak("smoke test complete")
        ok("TTS invoked — did you hear it? If silent, check speaker amp and volume.")
    except Exception as e:
        bad(f"TTS failed: {e}")
        failures += 1

    section("Vision stack (YOLO warm-up — first run downloads ~6MB)")
    try:
        from mcp_picrawler.vision import VisionStack

        vs = VisionStack()
        r = vs.scan(hw.latest_frame_bgr(retry_budget_s=2.0), include_objects=True)
        ok(f"scan: motion={r.motion:.3f} objects={len(r.objects)} elapsed={r.elapsed_ms:.0f}ms")
    except Exception as e:
        bad(f"vision scan failed: {e}")
        failures += 1

    print()
    if failures == 0:
        print(f"{GREEN}ALL GOOD — Nigel is ready.{RESET}")
        print("  Restart the service:  sudo systemctl start picrawler-mcp")
        return 0

    print(f"{RED}{failures} test(s) failed{RESET} — fix these before connecting Claude Code.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
