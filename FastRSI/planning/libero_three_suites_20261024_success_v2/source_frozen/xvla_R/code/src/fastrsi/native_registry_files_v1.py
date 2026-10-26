"""Read the complete native registry without repeated ancestor walks.

Every call traverses every configs directory and rereads all file bytes. There
is no metadata or digest cache. The workspace root is resolved as before; links
inside it are rejected. Checks and reads remain separate operations, so this
does not strengthen the previous concurrent filesystem/TOCTOU guarantees.
"""
import os
from pathlib import Path
import stat


def _regular_bytes(path):
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise ValueError('Symlink registry/configuration path is not supported')
    if not stat.S_ISREG(mode):
        raise ValueError('Registry/configuration path must be a regular file')
    return path.read_bytes()


def read_config_files(workspace):
    """Return all configs bytes, checking each tree entry once per call."""
    root = Path(workspace).resolve()
    folder = root/'configs'
    try:
        mode = folder.lstat().st_mode
    except FileNotFoundError:
        raise ValueError('Inherited configs directory required') from None
    if stat.S_ISLNK(mode):
        raise ValueError('Symlink registry/configuration path is not supported')
    if not stat.S_ISDIR(mode):
        raise ValueError('Inherited configs directory required')
    pending = [folder]
    result = {}
    while pending:
        directory = pending.pop()
        with os.scandir(str(directory)) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        children = []
        for entry in ordered:
            if entry.is_symlink():
                raise ValueError('Symlink registry/configuration path is not supported')
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                children.append(path)
            elif entry.is_file(follow_symlinks=False):
                result[path.relative_to(root).as_posix()] = path.read_bytes()
            else:
                raise ValueError('Configuration tree contains a nonregular entry')
        pending.extend(reversed(children))
    # Match the previous sorted Path inventory order, including nested entries.
    return {name: result[name] for name in sorted(result, key=Path)}


def read_registry_files(workspace):
    """Return TARGETS, SKILLRUNTIME and the complete configs byte inventory."""
    root = Path(workspace).resolve()
    result = {name: _regular_bytes(root/name)
              for name in ('TARGETS.md', 'SKILLRUNTIME.md')}
    result.update(read_config_files(root))
    return result
