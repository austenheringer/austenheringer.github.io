import asyncio

from warroom.voice.speech import SpeechBackend, SpeechQueue, clean_for_speech


class FakeVoice(SpeechBackend):
    name = "fake"

    def __init__(self, ok=True, fail=False, delay=0.01):
        self.ok, self.fail, self.delay = ok, fail, delay
        self.spoken: list[str] = []
        self.active = 0
        self.max_active = 0

    async def probe(self):
        return self.ok

    async def speak(self, text):
        if self.fail:
            raise OSError("audio device lost")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self.delay)
        self.spoken.append(text)
        self.active -= 1


def test_clean_for_speech():
    assert clean_for_speech("[RISK] **R:R** 2.1, TP1 then SL") == "reward to risk 2.1, target one then stop"


async def test_disabled_and_no_backend_is_text_only():
    q = SpeechQueue(False)
    assert await q.start() == "text-only" and not q.say("x")
    q = SpeechQueue(True, backends=[FakeVoice(ok=False)])
    assert await q.start() == "text-only" and not q.say("x")
    await q.stop()


async def test_real_backends_degrade_silently_without_packages():
    q = SpeechQueue(True)
    assert await q.start() in ("text-only", "edge-tts", "pyttsx3")
    await q.stop()


async def test_sequential_never_overlaps():
    fb = FakeVoice()
    q = SpeechQueue(True, backends=[fb])
    await q.start()
    for i in range(3):
        q.say(f"line {i}")
    await asyncio.sleep(0.15)
    assert fb.spoken == ["line 0", "line 1", "line 2"] and fb.max_active == 1
    await q.stop()


async def test_full_queue_drops_oldest():
    from warroom.config import VoiceSection

    fb = FakeVoice(delay=0.2)
    q = SpeechQueue(True, VoiceSection(max_queue=2), backends=[fb])
    await q.start()
    for i in range(5):
        q.say(f"l{i}")
    assert q.dropped >= 2
    await q.stop()


async def test_failing_backend_degrades_to_next():
    bad, good = FakeVoice(fail=True), FakeVoice()
    q = SpeechQueue(True, backends=[bad, good])
    await q.start()
    q.say("one")
    q.say("two")
    await asyncio.sleep(0.1)
    assert good.spoken == ["two"]
    await q.stop()
