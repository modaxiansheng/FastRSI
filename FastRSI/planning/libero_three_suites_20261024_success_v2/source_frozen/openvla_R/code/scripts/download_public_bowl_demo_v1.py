"""Download one pinned public demo file, preserving partials and verifying SHA."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    candidate = json.loads(args.candidate.read_text())
    item = candidate["matches"][0]
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / Path(item["path"]).name
    partial = destination.with_suffix(destination.suffix + ".partial")
    receipt_path = args.output / "DOWNLOAD.json"
    start = time.perf_counter()
    with (args.output / "DOWNLOAD.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = dict(status="RUNNING", pid=os.getpid(), source=candidate["pinned_url"],
                       revision=candidate["revision"], expected_bytes=item["size"],
                       expected_sha256=item["lfs"]["oid"], attempts=[],
                       model_calls=0, robot_actions=0, training_steps=0)
        def save():
            receipt["elapsed_seconds"] = time.perf_counter() - start
            tmp = receipt_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(receipt, indent=2) + "\n")
            os.replace(tmp, receipt_path)
        save()
        try:
            if destination.exists():
                if destination.stat().st_size != item["size"] or sha(destination) != item["lfs"]["oid"]:
                    raise ValueError("Existing destination differs; preserve it")
                receipt.update(status="REUSED_SHA_VERIFIED", bytes=destination.stat().st_size)
                return
            available = os.statvfs(args.output)
            if available.f_bavail * available.f_frsize < 2 * item["size"]:
                raise ValueError("Insufficient fresh filesystem capacity")
            for attempt in range(1, 4):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > item["size"]:
                    raise ValueError("Oversized partial retained")
                if offset == item["size"]:
                    break
                entry = dict(attempt=attempt, starting_byte=offset, status="RUNNING")
                receipt["attempts"].append(entry)
                save()
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                request = urllib.request.Request(candidate["pinned_url"] + "?download=true", headers=headers)
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                            raise ValueError("Resume response not bound to partial offset")
                        with partial.open("ab" if offset else "xb") as stream:
                            while True:
                                block = response.read(8 * 1024 * 1024)
                                if not block:
                                    break
                                if stream.tell() + len(block) > item["size"]:
                                    raise ValueError("Response exceeds pinned file size")
                                stream.write(block)
                                receipt["downloaded_bytes"] = stream.tell()
                                save()
                    entry["status"] = "RESPONSE_COMPLETE"
                    if partial.stat().st_size == item["size"]:
                        break
                except Exception as error:
                    entry.update(status="FAILED_PARTIAL_RETAINED", error=repr(error))
                    save()
            if not partial.exists() or partial.stat().st_size != item["size"]:
                raise ValueError("Download incomplete after bounded attempts; partial retained")
            actual_sha = sha(partial)
            receipt["actual_sha256"] = actual_sha
            if actual_sha != item["lfs"]["oid"]:
                raise ValueError("SHA mismatch; preserve partial, do not promote")
            os.replace(partial, destination)
            receipt.update(status="DOWNLOADED_SHA_VERIFIED", bytes=destination.stat().st_size,
                           destination=str(destination))
        except BaseException as error:
            receipt.update(status="FAILED_RETAINED", error=repr(error))
            raise
        finally:
            save()
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
