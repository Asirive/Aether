from __future__ import annotations

import os
import sys
import time
from pathlib import Path


class TeeLogger:
    _instances: list["TeeLogger"] = []

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8", buffering=1)
        TeeLogger._instances.append(self)

    def log(self, msg: str, *, flush: bool = True) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except Exception:
            pass
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
                if flush:
                    self._fh.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        if self in TeeLogger._instances:
            TeeLogger._instances.remove(self)


_DEFAULT: TeeLogger | None = None


def get_logger(path: str | Path | None = None) -> TeeLogger:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TeeLogger(path)
    return _DEFAULT


def log(msg: str, *, path: str | Path | None = None) -> None:
    if _DEFAULT is not None:
        _DEFAULT.log(msg)
        return
    if path is not None:
        TeeLogger(path).log(msg)
        return
    try:
        sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stdout.flush()
    except Exception:
        pass