"""Tests for correcting the Alexa player's position from the Echo's own reports."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from music_assistant_models.enums import PlaybackState

from music_assistant.providers import alexa
from music_assistant.providers.alexa import (
    POLL_INTERVAL,
    AlexaDevice,
    AlexaPlayer,
    get_playback_state,
)

NOW = 1_000_000.0


def _setup_value(key: str) -> str | None:
    return {
        alexa.CONF_API_URL: "http://skill:5000/",
        alexa.CONF_API_BASIC_AUTH_USERNAME: "user",
        alexa.CONF_API_BASIC_AUTH_PASSWORD: "secret",
    }.get(key)


@pytest.fixture
def provider() -> MagicMock:
    """Create a mock AlexaProvider."""
    provider = MagicMock()
    provider.instance_id = "alexa"
    provider.domain = "alexa"
    provider.manifest.domain = "alexa"
    provider.mass = MagicMock()
    provider.get_setup_value = MagicMock(side_effect=_setup_value)
    config = MagicMock()
    config.name = None
    config.default_name = "Echo Dot"
    config.enabled = True
    config.player_type = None
    config.get_value = MagicMock(return_value=None)
    provider.mass.config.get_base_player_config.return_value = config
    return provider


@pytest.fixture
def player(provider: MagicMock, monkeypatch: pytest.MonkeyPatch) -> AlexaPlayer:
    """Create a playing AlexaPlayer that does not publish state."""
    player = AlexaPlayer(provider, "Echo Dot", AlexaDevice())
    player._attr_playback_state = PlaybackState.PLAYING
    player._attr_elapsed_time = 0
    player._attr_elapsed_time_last_updated = NOW - 600
    monkeypatch.setattr(player, "update_state", MagicMock())
    return player


def _report(**changes: Any) -> dict[str, Any]:
    report = {
        "current": True,
        "activity": "PLAYING",
        "offsetMs": 125_000,
        "ageSeconds": 2.0,
        "observedAt": 42.0,
    }
    report.update(changes)
    return report


async def _poll(player: AlexaPlayer, report: dict[str, Any] | None) -> None:
    with (
        patch.object(alexa, "get_playback_state", AsyncMock(return_value=report)),
        patch("music_assistant.providers.alexa.time.time", return_value=NOW),
    ):
        await player.poll()


@pytest.mark.parametrize(
    ("state", "polled"),
    [
        (PlaybackState.PLAYING, True),
        # Alexa resumes on its own after a voice request, so a paused Echo is watched too.
        (PlaybackState.PAUSED, True),
        (PlaybackState.IDLE, False),
    ],
)
def test_polls_while_playing_or_paused_but_not_when_stopped(
    player: AlexaPlayer, state: PlaybackState, polled: bool
) -> None:
    """Only an Echo that can move on its own is checked, every few seconds."""
    player._attr_playback_state = state
    assert player.needs_poll is polled
    assert player.poll_interval == POLL_INTERVAL


async def test_reported_offset_replaces_the_clock_estimate(player: AlexaPlayer) -> None:
    """The Echo's stream offset becomes the elapsed time, anchored to its age."""
    await _poll(player, _report())
    assert player._attr_elapsed_time == 125
    assert player._attr_elapsed_time_last_updated == NOW - 2.0
    assert player.playback_state == PlaybackState.PLAYING
    # MA extrapolates from the anchor while playing: 125 s plus the 2 s since the report.
    with patch("music_assistant.models.player.time.time", return_value=NOW):
        assert player.corrected_elapsed_time == pytest.approx(127)
    player.update_state.assert_called_once()  # type: ignore[attr-defined]


async def test_pause_for_a_voice_request_pauses_the_player(player: AlexaPlayer) -> None:
    """Alexa's STOPPED (paused for a voice request) holds the position."""
    await _poll(player, _report(activity="STOPPED", offsetMs=90_500))
    assert player.playback_state == PlaybackState.PAUSED
    assert player._attr_elapsed_time == 90.5
    assert player.needs_poll


@pytest.mark.parametrize(
    "report",
    [
        None,
        {},
        _report(current=False),
        _report(activity="FINISHED"),
        _report(activity="IDLE"),
        _report(offsetMs=None),
        _report(ageSeconds="soon"),
    ],
)
async def test_unusable_reports_change_nothing(
    player: AlexaPlayer, report: dict[str, Any] | None
) -> None:
    """Old streams, finished or idle Echos and malformed reports are ignored."""
    await _poll(player, report)
    assert player._attr_elapsed_time == 0
    assert player.playback_state == PlaybackState.PLAYING
    player.update_state.assert_not_called()  # type: ignore[attr-defined]


async def test_same_report_is_applied_once(player: AlexaPlayer) -> None:
    """A report already applied must not overwrite later local changes."""
    await _poll(player, _report())
    player._attr_playback_state = PlaybackState.PAUSED  # e.g. paused from the MA UI
    await _poll(player, _report())
    assert player.playback_state == PlaybackState.PAUSED
    player.update_state.assert_called_once()  # type: ignore[attr-defined]


async def test_stopped_player_is_not_revived(player: AlexaPlayer) -> None:
    """After MA stops the player, Echo reports do not change its state."""
    player._attr_playback_state = PlaybackState.IDLE
    lookup = AsyncMock(return_value=_report())
    with patch.object(alexa, "get_playback_state", lookup):
        await player.poll()
    lookup.assert_not_awaited()
    assert player.playback_state == PlaybackState.IDLE


def _session(status: int = 200, body: Any = None, error: Exception | None = None) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.json = AsyncMock(return_value=body)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get = MagicMock(side_effect=error) if error else MagicMock(return_value=context)
    return session


async def test_playback_state_request_uses_the_skill_api(provider: MagicMock) -> None:
    """The report is fetched for the player with the configured API credentials."""
    provider.mass.http_session = _session(body=_report())
    assert await get_playback_state(provider, "Echo Dot") == _report()
    args, kwargs = provider.mass.http_session.get.call_args
    assert args == ("http://skill:5000/alexa/playback-state",)
    assert kwargs["params"] == {"player_id": "Echo Dot"}
    assert kwargs["auth"] == aiohttp.BasicAuth("user", "secret")


@pytest.mark.parametrize(
    "session",
    [
        _session(status=404),
        _session(body=["not", "an", "object"]),
        _session(error=aiohttp.ClientConnectionError("down")),
        _session(error=TimeoutError()),
    ],
)
async def test_playback_state_failures_return_none(provider: MagicMock, session: MagicMock) -> None:
    """Missing endpoints, bad bodies and network errors are quiet None results."""
    provider.mass.http_session = session
    assert await get_playback_state(provider, "Echo Dot") is None
