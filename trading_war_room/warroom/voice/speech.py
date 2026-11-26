"""J.A.R.V.I.S. voice output: an async, non-overlapping speech queue.

Backend chain (first that works wins, re-probed down the chain on failure):
1. edge-tts (``en-GB-RyanNeural``) synthesised to a temp mp3, played with
   ``pygame.mixer`` in a worker thread, temp file deleted afterwards.
   Needs network access to Microsoft's TTS service.
2. pyttsx3 (offline, OS voices; prefers an en-GB voice if installed).
3. Text-only: speech silently disabled, briefings still appear in the TUI.

All imports are lazy. Every blocking call runs in ``asyncio.to_thread`` so
speaking never stalls market scanning. The queue is bounded and drops the
OLDEST pending line when full: a stale briefing is worse than none.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import threading
from abc import ABC, abstractmethod

from warroom.config import VoiceSection

log = logging.getLogger(__name__)


def clean_for_speech(text: str) -> str:
    text = re.sub(r"\[[A-Z .]+\]", "", text)          # [RISK] style tags
    text = re.sub(r"[*_`#>|]", "", text)               # markdown
    text = text.replace("R:R", "reward to risk").replace("TP1", "target one").replace("TP2", "target two")
    text = re.sub(r"\bSL\b", "stop", text)
    return re.sub(r"\s+", " ", text).strip()


class SpeechBackend(ABC):
    name = "base"

    @abstractmethod
    async def probe(self) -> bool:
        """Return True if this backend can speak on this machine."""

    @abstractmethod
    async def speak(self, text: str) -> None: ...

    async def stop(self) -> None:  # noqa: B027
        pass


class EdgeTTSBackend(SpeechBackend):
    name = "edge-tts"

    def __init__(self, voice: str, rate: str) -> None:
        self.voice = voice
        self.rate = rate
        self._stop = threading.Event()

    async def probe(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            import pygame  # noqa: F401
        except ImportError:
            return False

        def _init() -> bool:
            import pygame

            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                return True
            except Exception:  # noqa: BLE001 - no audio device
                return False

        return await asyncio.to_thread(_init)

    async def speak(self, text: str) -> None:
        import edge_tts

        fd, path = tempfile.mkstemp(prefix="jarvis_", suffix=".mp3")
        os.close(fd)
        try:
            await edge_tts.Communicate(text, self.voice, rate=self.rate).save(path)
            self._stop.clear()
            await asyncio.to_thread(self._play_blocking, path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _play_blocking(self, path: str) -> None:
        import pygame

        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        try:
            while pygame.mixer.music.get_busy() and not self._stop.is_set():
                pygame.time.wait(50)
        finally:
            pygame.mixer.music.stop()
            try:
                pygame.mixer.music.unload()  # release the file handle so it can be deleted (Windows)
            except Exception:  # noqa: BLE001
                pass

    async def stop(self) -> None:
        self._stop.set()

        def _quit() -> None:
            try:
                import pygame

                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()
            except Exception:  # noqa: BLE001
                pass

        await asyncio.to_thread(_quit)


class Pyttsx3Backend(SpeechBackend):
    name = "pyttsx3"

    def __init__(self) -> None:
        self._engine = None
        self._lock = threading.Lock()

    async def probe(self) -> bool:
        try:
            import pyttsx3  # noqa: F401
        except ImportError:
            return False

        def _init() -> bool:
            import pyttsx3

            try:
                eng = pyttsx3.init()
                for v in eng.getProperty("voices") or []:
                    ident = f"{getattr(v, 'id', '')} {getattr(v, 'name', '')}".lower()
                    if "en-gb" in ident or "en_gb" in ident or "british" in ident or "daniel" in ident:
                        eng.setProperty("voice", v.id)
                        break
                eng.setProperty("rate", 175)
                self._engine = eng
                return True
            except Exception:  # noqa: BLE001 - no speech driver
                return False

        return await asyncio.to_thread(_init)

    async def speak(self, text: str) -> None:
        def _say() -> None:
            with self._lock:
                self._engine.say(text)
                self._engine.runAndWait()

        await asyncio.to_thread(_say)

    async def stop(self) -> None:
        if self._engine is not None:
            try:
                await asyncio.to_thread(self._engine.stop)
            except Exception:  # noqa: BLE001
                pass


class SpeechQueue:
    def __init__(self, enabled: bool, cfg: VoiceSection | None = None,
                 backends: list[SpeechBackend] | None = None) -> None:
        cfg = cfg or VoiceSection()
        self.requested = enabled
        self.backends = backends if backends is not None else [EdgeTTSBackend(cfg.edge_voice, cfg.rate), Pyttsx3Backend()]
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=cfg.max_queue)
        self._backend: SpeechBackend | None = None
        self._task: asyncio.Task | None = None
        self.dropped = 0
        self.spoken = 0

    @property
    def active(self) -> bool:
        return self._backend is not None

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else "text-only"

    async def start(self) -> str:
        if not self.requested:
            return "text-only"
        await self._select_backend(self.backends)
        if self._backend is None:
            log.warning("voice requested but no speech backend available; continuing text-only")
            return "text-only"
        self._task = asyncio.create_task(self._worker(), name="speech-queue")
        return self._backend.name

    async def _select_backend(self, candidates: list[SpeechBackend]) -> None:
        self._backend = None
        for b in candidates:
            try:
                if await b.probe():
                    self._backend = b
                    log.info("speech backend: %s", b.name)
                    return
            except Exception:  # noqa: BLE001
                log.exception("speech backend %s probe failed", b.name)

    def say(self, text: str) -> bool:
        """Enqueue a line without blocking. Returns False if speech is inactive."""
        if not self.active:
            return False
        text = clean_for_speech(text)
        if not text:
            return False
        while True:
            try:
                self._queue.put_nowait(text)
                return True
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    pass

    async def _worker(self) -> None:
        while True:
            text = await self._queue.get()
            if self._backend is None:
                continue
            try:
                await self._backend.speak(text)
                self.spoken += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - degrade to the next backend
                failed = self._backend
                log.warning("speech backend %s failed (%s); degrading", failed.name, exc)
                remaining = self.backends[self.backends.index(failed) + 1:] if failed in self.backends else []
                await self._select_backend(remaining)
                if self._backend is None:
                    log.warning("no speech backend left; text-only from now on")
                    return

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        if self._backend:
            await self._backend.stop()
        self._backend = None
