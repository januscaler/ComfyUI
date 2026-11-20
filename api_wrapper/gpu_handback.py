"""A render takes the GPU back from the engines that sit warm beside it.

Maya and OmniVoice run behind idle gates (``docker/idle_gate.py``) that may keep a model loaded
past its idle window while nothing else uses the card. A video render (H3, LTX) loads ~30 GB of an
RTX 5090's 32; started beside a warm Maya (~12 GB) it is the one that runs out of memory. So before
the wrapper queues such a prompt it asks each gate to hand the GPU back
(``POST /__idle-gate/release?block=N&wait=M``): the gate waits for the lines it is speaking, stops
its server, and answers only once it is down; for ``N`` seconds nothing starts it warm again.

Set by the deployment, all optional — unset, nothing is asked and nothing changes:

* ``WRAPPER_RELEASE_URLS`` — comma-separated gate origins, e.g.
  ``http://voxmin-maya:8080,http://voxmin-omnivoice:3900`` (the gates must be reachable from here);
* ``WRAPPER_RELEASE_WORKFLOWS`` — the workflows that need the whole card (default ``minimaxh3,ltx25``);
* ``WRAPPER_RELEASE_BLOCK_S`` (300) and ``WRAPPER_RELEASE_WAIT_S`` (30).

A gate that is unreachable, slow or refuses is logged and skipped: a render is never failed or held
past the bounded wait because of an engine.
"""

from __future__ import annotations

import asyncio
import logging
import os

import aiohttp

DEFAULT_WORKFLOWS = ("minimaxh3", "ltx25")
RELEASE_PATH = "/__idle-gate/release"
#: Seconds past the gate's own wait that a call may take (connecting, answering).
SLACK_S = 5.0


def urls() -> list[str]:
    return [u.strip().rstrip("/") for u in os.environ.get("WRAPPER_RELEASE_URLS", "").split(",")
            if u.strip()]


def workflows() -> frozenset[str]:
    raw = os.environ.get("WRAPPER_RELEASE_WORKFLOWS")
    names = [n.strip().lower() for n in raw.split(",")] if raw is not None else DEFAULT_WORKFLOWS
    return frozenset(n for n in names if n)


def wanted(workflow: str) -> bool:
    """Whether a prompt of ``workflow`` must have the GPU to itself. Pure given the environment."""
    return bool(urls()) and workflow.lower() in workflows()


def _seconds(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, "") or default))
    except ValueError:
        return default


async def _release_one(session: aiohttp.ClientSession, base: str, block: float, wait: float) -> str:
    """One gate: ``"released"``, or why not (never raises)."""
    try:
        async with session.post(f"{base}{RELEASE_PATH}", params={"block": f"{block:g}",
                                                                "wait": f"{wait:g}"}) as answer:
            if answer.status == 200:
                return "released"
            return f"answered {answer.status}"
    except asyncio.CancelledError:
        raise
    except Exception as err:  # noqa: BLE001 — a gate must never fail a render
        return f"{type(err).__name__}: {err}"


async def release_all(*, session_factory=aiohttp.ClientSession) -> dict[str, str]:
    """Ask every configured gate to hand the GPU back, side by side, and wait until each has
    answered or the bound (the gate's wait plus a few seconds) is spent. ``{url: outcome}``;
    anything but ``released`` is logged. Never raises."""
    bases = urls()
    if not bases:
        return {}
    block, wait = _seconds("WRAPPER_RELEASE_BLOCK_S", 300.0), _seconds("WRAPPER_RELEASE_WAIT_S", 30.0)
    timeout = aiohttp.ClientTimeout(total=wait + SLACK_S)
    outcomes: dict[str, str] = {}
    try:
        async with session_factory(timeout=timeout) as session:
            done = await asyncio.wait_for(
                asyncio.gather(*(_release_one(session, base, block, wait) for base in bases)),
                timeout=wait + SLACK_S)
        outcomes = dict(zip(bases, done, strict=True))
    except Exception as err:  # noqa: BLE001 — bounded, logged, never fatal
        logging.warning("wrapper: handing the GPU back timed out or failed (%s); rendering anyway", err)
        return {base: "not confirmed" for base in bases}
    for base, outcome in outcomes.items():
        if outcome != "released":
            logging.warning("wrapper: %s did not hand the GPU back (%s); rendering anyway", base, outcome)
    return outcomes
