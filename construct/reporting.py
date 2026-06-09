from __future__ import annotations

import sys
import time
from dataclasses import dataclass

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


class _PlainProgress:
    def __init__(self, total: int, desc: str, enabled: bool) -> None:
        self.total = total
        self.desc = desc
        self.enabled = enabled
        self.current = 0
        self.last_percent = -1
        if self.enabled:
            print(f"{self.desc}: 0/{self.total}", flush=True)

    def update(self, step: int = 1) -> None:
        if not self.enabled:
            return
        self.current += step
        percent = int((self.current / max(1, self.total)) * 100)
        should_print = self.total <= 20 or self.current == self.total or percent >= self.last_percent + 10
        if should_print:
            self.last_percent = percent
            print(f"{self.desc}: {self.current}/{self.total} ({percent}%)", flush=True)

    def close(self) -> None:
        return


@dataclass
class ConsoleReporter:
    enabled: bool = True
    log_timestamps: bool = True
    progress_bar_width: int = 28

    def section(self, title: str, detail: str = "") -> None:
        if not self.enabled:
            return
        line = title if not detail else f"{title} | {detail}"
        self._write("")
        self._write(f"{self._prefix()} {line}")

    def info(self, message: str) -> None:
        if not self.enabled:
            return
        self._write(f"{self._prefix()}   {message}")

    def warn(self, message: str) -> None:
        if not self.enabled:
            return
        self._write(f"{self._prefix()}   WARNING: {message}")

    def progress(self, total: int, description: str) -> tqdm:
        if _tqdm is None:
            return _PlainProgress(total=total, desc=description, enabled=self.enabled)
        return _tqdm(
            total=total,
            desc=description,
            leave=False,
            ascii=True,
            dynamic_ncols=True,
            ncols=max(80, self.progress_bar_width + 40),
            file=sys.stdout,
            disable=not self.enabled,
        )

    def _prefix(self) -> str:
        if not self.log_timestamps:
            return "[Harness]"
        return f"[{time.strftime('%H:%M:%S')}]"

    def _write(self, message: str) -> None:
        print(message, flush=True)
