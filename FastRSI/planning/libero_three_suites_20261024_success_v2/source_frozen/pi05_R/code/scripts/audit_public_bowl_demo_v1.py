"""Inspect a pinned public HDF5 demo using bounded HTTP reads; no training."""
import argparse
import io
import json
import sys
from pathlib import Path

import requests


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, budget=32 * 1024 * 1024):
        self.url, self.size, self.budget = url, size, budget
        self.position = self.bytes_read = 0
        self.requests = 0
        self.cache = {}
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 1:
            offset += self.position
        elif whence == 2:
            offset += self.size
        if offset < 0:
            raise ValueError("Negative seek")
        self.position = offset
        return offset

    def block(self, start):
        end = min(start + 65536, self.size) - 1
        size = end - start + 1
        key = (start, end)
        if key not in self.cache:
            if self.bytes_read + size > self.budget:
                raise ValueError("Metadata/sample audit byte budget exceeded")
            # Separate cache entries for different ranges of the same pinned file.
            response = self.session.get(self.url + f"?audit_range={start}-{end}",
                                        headers={"Range": f"bytes={start}-{end}"},
                                        stream=True, timeout=25)
            try:
                expected = f"bytes {start}-{end}/{self.size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Server did not honor exact bounded range")
                data = response.raw.read(size + 1)
                if len(data) != size:
                    raise ValueError("Truncated or oversized byte range")
            finally:
                response.close()
            self.bytes_read += size
            self.requests += 1
            self.cache[key] = data
        return self.cache[key]

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b""
        remaining, chunks = size, []
        while remaining:
            start = self.position // 65536 * 65536
            block = self.block(start)
            offset = self.position - start
            count = min(remaining, len(block) - offset)
            chunks.append(block[offset:offset + count])
            self.position += count
            remaining -= count
        return b"".join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def plain(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "audit_deps"))
    import h5py
    candidate = json.loads((root / "PUBLIC_DEMO_CANDIDATE.json").read_text())
    assert candidate["range_supported"]
    match = candidate["matches"][0]
    file = RangeFile(candidate["pinned_url"], match["size"])
    with h5py.File(file, "r") as h5:
        data = h5["data"]
        episodes = []
        names = sorted(data, key=lambda n: int(n.split("_")[-1]))
        inspected = [names[0], names[len(names) // 2], names[-1]]
        for name in inspected:
            demo = data[name]
            episodes.append(dict(name=name, attrs={k: plain(v) for k, v in demo.attrs.items()
                                                   if k != "model_file"},
                                 actions_shape=list(demo["actions"].shape),
                                 observation_shapes={k: list(v.shape) for k, v in demo["obs"].items()},
                                 states_shape=list(demo["states"].shape) if "states" in demo else None))
        result = dict(pinned_url=candidate["pinned_url"], revision=candidate["revision"],
                      file_size=match["size"], declared_lfs_sha256=match["lfs"]["oid"],
                      full_file_downloaded=False, full_file_hash_verified=False,
                      data_attrs={k: plain(v) for k, v in data.attrs.items()},
                      episodes=episodes, episode_count=len(names), inspected_episode_count=len(episodes),
                      sampling_rule="first, middle, last in numeric demo order; metadata only",
                      min_actions_inspected=min(e["actions_shape"][0] for e in episodes),
                      max_actions_inspected=max(e["actions_shape"][0] for e in episodes),
                      verified_success_in_current_environment=False,
                      better_than_current_policy_proven=False,
                      new_model_calls=0, new_training_steps=0)
    result.update(downloaded_bytes=file.bytes_read, range_requests=file.requests)
    (root / "PUBLIC_DEMO_METADATA_AUDIT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
