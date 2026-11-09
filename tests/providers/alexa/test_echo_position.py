"""Tests for correcting the Alexa player's position from the Echo's own reports."""

import asyncio
import math
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

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
    provider.mass.loop.time.return_value = 100.0
    provider._playback_state_retry_at = -math.inf
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


@pytest.mark.parametrize(
    ("session", "backs_off"),
    [
        (_session(error=aiohttp.ClientConnectionError("down")), True),
        (_session(error=TimeoutError()), True),
        (_session(status=404), False),
    ],
)
async def test_unreachable_skill_api_is_left_alone_for_a_while(
    provider: MagicMock, session: MagicMock, backs_off: bool
) -> None:
    """MA polls players in turn, so a down API must not cost a timeout per Echo each round."""
    provider.mass.http_session = session
    await get_playback_state(provider, "Echo Dot")
    provider.mass.loop.time.return_value = 100.0 + alexa.PLAYBACK_STATE_RETRY_DELAY - 1
    await get_playback_state(provider, "Other Echo")
    assert session.get.call_count == (1 if backs_off else 2)
    provider.mass.loop.time.return_value = 100.0 + alexa.PLAYBACK_STATE_RETRY_DELAY
    await get_playback_state(provider, "Echo Dot")
    assert session.get.call_count == (2 if backs_off else 3)


@pytest.mark.parametrize("command", ["stop", "new stream"])
async def test_report_fetched_before_a_playback_command_is_discarded(
    player: AlexaPlayer, command: str
) -> None:
    """A stop or new stream while the request is out makes its report stale."""

    async def slow_lookup(*_args: Any) -> dict[str, Any]:
        player._playback_generation += 1  # what stop()/play_media() do first
        if command == "stop":
            player._attr_playback_state = PlaybackState.IDLE
        return _report()

    with patch.object(alexa, "get_playback_state", AsyncMock(side_effect=slow_lookup)):
        await player.poll()
    player.update_state.assert_not_called()  # type: ignore[attr-defined]
    assert player._attr_elapsed_time == 0
    expected = PlaybackState.IDLE if command == "stop" else PlaybackState.PLAYING
    assert player.playback_state == expected


@pytest.mark.parametrize("method", ["stop", "play", "pause", "play_media"])
async def test_report_requested_during_a_playback_command_is_discarded(
    player: AlexaPlayer, provider: MagicMock, method: str
) -> None:
    """A lookup started while a command waits for Amazon and answered after it is stale."""
    provider.get_intent_utterance = AsyncMock(return_value="ask music assistant to stop")
    provider.mass.streams.resolve_stream_url = AsyncMock(return_value="http://ma/flow/x.mp3")
    provider.config.get_value.return_value = "en-US"
    answer = asyncio.Event()
    lookups: list[asyncio.Task[None]] = []

    async def slow_lookup(*_args: Any) -> dict[str, Any]:
        await answer.wait()
        return _report()

    async def amazon(*_args: Any) -> None:
        # The poll asks the skill API while the command is still waiting for Amazon.
        lookups.append(asyncio.create_task(player.poll()))
        await asyncio.sleep(0)

    api = MagicMock(run_custom=AsyncMock(side_effect=amazon))
    with (
        patch.object(AlexaPlayer, "api", new_callable=PropertyMock, return_value=api),
        patch.object(alexa, "api_request", AsyncMock(return_value="{}")),
        patch.object(alexa, "get_playback_state", AsyncMock(side_effect=slow_lookup)),
        patch("music_assistant.providers.alexa.time.time", return_value=NOW),
    ):
        if method == "play_media":
            await player.play_media(MagicMock(title="Song", image_url=None))
        else:
            await getattr(player, method)()
        state = player.playback_state
        update_state = cast("MagicMock", player.update_state)
        position = player._attr_elapsed_time
        update_state.reset_mock()
        answer.set()
        await lookups[0]
    update_state.assert_not_called()
    assert player.playback_state == state
    # The command's own position stands, not the late report's 125 s.
    assert player._attr_elapsed_time == position
    assert position == (600 if method in ("play", "pause") else 0)


async def test_paused_report_moves_the_queue_without_a_playing_blip(provider: MagicMock) -> None:
    """MA's real update path sees one paused jump to the Echo's offset, vouched for."""
    # Let the player's own position count: no audio-source session or other player to mirror.
    provider.mass.players.get_audio_source_session.return_value = None
    provider.mass.players.get_player.return_value = None
    player = AlexaPlayer(provider, "Echo Dot", AlexaDevice())
    player._attr_playback_state = PlaybackState.PLAYING
    player._attr_elapsed_time = 0
    player._attr_elapsed_time_last_updated = NOW - 600
    jumps: list[tuple[PlaybackState, float | None, bool]] = []
    provider.mass.players.on_player_position_jumped.side_effect = lambda p: jumps.append(
        (p.state.playback_state, p.state.corrected_elapsed_time, p.paused_position_reliable)
    )
    published: list[PlaybackState] = []
    provider.mass.players.signal_player_state_update.side_effect = lambda p, *_args, **_kwargs: (
        published.append(p.state.playback_state)
    )
    with patch("music_assistant.models.player.time.time", return_value=NOW):
        player.update_state()  # settle the initial state (clock estimate at 600 s)
    jumps.clear()
    published.clear()
    with (
        patch.object(
            alexa,
            "get_playback_state",
            AsyncMock(return_value=_report(activity="STOPPED", offsetMs=90_500)),
        ),
        patch("music_assistant.providers.alexa.time.time", return_value=NOW),
        patch("music_assistant.models.player.time.time", return_value=NOW),
    ):
        await player.poll()
    assert len(jumps) == 1
    state, elapsed, vouched = jumps[0]
    assert state == PlaybackState.PAUSED
    assert elapsed == pytest.approx(90.5)
    assert vouched
    assert PlaybackState.PLAYING not in published
    assert player.playback_state == PlaybackState.PAUSED
    assert player.state.corrected_elapsed_time == pytest.approx(90.5)


async def test_playing_report_is_not_a_vouched_paused_position(player: AlexaPlayer) -> None:
    """Only a paused offset from the Echo lets the queue follow a paused player."""
    await _poll(player, _report(activity="STOPPED", offsetMs=90_500))
    assert player.paused_position_reliable
    await _poll(player, _report(activity="PLAYING", offsetMs=91_000, observedAt=43.0))
    assert not player.paused_position_reliable


@pytest.mark.parametrize("method", ["stop", "play", "pause", "play_media"])
async def test_playback_commands_do_not_vouch_for_the_paused_position(
    player: AlexaPlayer, provider: MagicMock, method: str
) -> None:
    """MA's own commands do not know where the Echo is, so they clear the flag."""
    await _poll(player, _report(activity="STOPPED", offsetMs=90_500))
    provider.get_intent_utterance = AsyncMock(return_value="ask music assistant to stop")
    provider.mass.streams.resolve_stream_url = AsyncMock(return_value="http://ma/flow/x.mp3")
    provider.config.get_value.return_value = "en-US"
    api = MagicMock(run_custom=AsyncMock())
    with (
        patch.object(AlexaPlayer, "api", new_callable=PropertyMock, return_value=api),
        patch.object(alexa, "api_request", AsyncMock(return_value="{}")),
    ):
        if method == "play_media":
            await player.play_media(MagicMock(title="Song", image_url=None))
        else:
            await getattr(player, method)()
    assert not player.paused_position_reliable


async def _send(player: AlexaPlayer, provider: Any, method: str, at: float) -> None:
    """Run a playback command from MA as if it finished at the given time."""
    provider.get_intent_utterance = AsyncMock(return_value="ask music assistant to resume")
    api = MagicMock(run_custom=AsyncMock())
    with (
        patch.object(AlexaPlayer, "api", new_callable=PropertyMock, return_value=api),
        patch("music_assistant.providers.alexa.time.time", return_value=at),
    ):
        await getattr(player, method)()


async def test_resume_after_a_reported_pause_continues_from_the_paused_offset(
    player: AlexaPlayer, provider: MagicMock
) -> None:
    """A long pause must not be added to the position when MA resumes the Echo."""
    await _poll(player, _report(activity="STOPPED", offsetMs=90_500))
    await _send(player, provider, "play", NOW + 600)
    assert player.playback_state == PlaybackState.PLAYING
    assert player._attr_elapsed_time == pytest.approx(90.5)
    assert player._attr_elapsed_time_last_updated == NOW + 600


async def test_resume_after_a_pause_from_ma_continues_where_it_paused(
    player: AlexaPlayer, provider: MagicMock
) -> None:
    """MA's own pause freezes the clock estimate, and the resume continues from it."""
    # The fixture's player has been playing for 600 s by MA's clock.
    await _send(player, provider, "pause", NOW)
    assert player._attr_elapsed_time == pytest.approx(600)
    await _send(player, provider, "play", NOW + 300)
    assert player._attr_elapsed_time == pytest.approx(600)
    assert player._attr_elapsed_time_last_updated == NOW + 300
