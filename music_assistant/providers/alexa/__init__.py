"""Alexa player provider support for Music Assistant."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from typing import TYPE_CHECKING, Any, cast

import aiohttp
from aiohttp import BasicAuth
from alexapy import AlexaAPI, AlexaLogin
from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigValueOption,
)
from music_assistant_models.enums import (
    ConfigEntryType,
    PlaybackState,
    PlayerFeature,
    ProviderFeature,
)
from music_assistant_models.errors import ActionUnavailable
from music_assistant_models.player import DeviceInfo, PlayerMedia

from music_assistant.constants import CONF_PASSWORD, CONF_USERNAME
from music_assistant.helpers.util import lock
from music_assistant.models.player import Player
from music_assistant.models.player_provider import PlayerProvider

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_URL = "url"
CONF_AUTH_SECRET = "secret"
CONF_API_BASIC_AUTH_USERNAME = "api_username"
CONF_API_BASIC_AUTH_PASSWORD = "api_password"
CONF_API_URL = "api_url"
CONF_ALEXA_LANGUAGE = "alexa_language"

ALEXA_LANGUAGE_COMMANDS = {
    "play_audio_de-DE": "sag music assistant spiele audio",
    "play_audio_en-CA": "ask music assistant to play audio",
    "play_audio_en-US": "ask music assistant to play audio",
    "play_audio_es-ES": "pídele a music assistant que reproduzca audio",
    "play_audio_fr-CA": "music assistant",
    "play_audio_fr-FR": "music assistant",
    "play_audio_it-IT": "chiedi a music assistant di riprodurre audio",
    "play_audio_pt-BR": "peça ao music assistant para reproduzir áudio",
    "play_audio_nl-NL": "speel music assistant",
    "play_audio_default": "ask music assistant to play audio",
}

SUPPORTED_FEATURES: set[ProviderFeature] = set()  # no special features supported (yet)

# Alexa pauses skill audio for every voice request and resumes it on its own, so the Echo
# falls behind a clock started at play_media. While playing or paused, ask the skill API
# where the Echo really is every few seconds.
POLL_INTERVAL = 5
# MA polls its players one after another, so an unreachable skill API is left alone for
# a while rather than holding up every poll round with a timeout per Echo.
PLAYBACK_STATE_RETRY_DELAY = 60

# AudioPlayer activity reported by the Echo. IDLE and FINISHED are left to MA's own
# stop and end-of-stream handling.
ECHO_PLAYBACK_STATES = {
    "PLAYING": PlaybackState.PLAYING,
    "BUFFER_UNDERRUN": PlaybackState.PLAYING,
    "STOPPED": PlaybackState.PAUSED,
    "PAUSED": PlaybackState.PAUSED,
}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return AlexaProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def save_cookie(login: AlexaLogin, username: str, mass: MusicAssistant) -> None:
    """Save the cookie file for the Alexa login."""
    if login._session is None:
        _LOGGER.error("AlexaLogin session is not initialized.")
        return

    cookie_dir = os.path.join(mass.storage_path, ".alexa")
    await asyncio.to_thread(os.makedirs, cookie_dir, exist_ok=True)
    cookie_path = os.path.join(cookie_dir, f"alexa_media.{username}.pickle")
    login._cookiefile = [login._outputpath(cookie_path)]
    if (login._cookiefile[0]) and await asyncio.to_thread(os.path.exists, login._cookiefile[0]):
        _LOGGER.debug("Removing outdated cookiefile %s", login._cookiefile[0])
        await delete_cookie(login._cookiefile[0])
    cookie_jar = login._session.cookie_jar
    assert isinstance(cookie_jar, aiohttp.CookieJar)
    if login._debug:
        _LOGGER.debug("Saving cookie to %s", login._cookiefile[0])
    try:
        await asyncio.to_thread(cookie_jar.save, login._cookiefile[0])
    except OSError, EOFError, TypeError, AttributeError:
        _LOGGER.debug("Error saving pickled cookie to %s", login._cookiefile[0])


async def delete_cookie(cookiefile: str) -> None:
    """Delete the specified cookie file."""
    if await asyncio.to_thread(os.path.exists, cookiefile):
        try:
            await asyncio.to_thread(os.remove, cookiefile)
            _LOGGER.debug("Deleted cookie file: %s", cookiefile)
        except OSError as e:
            _LOGGER.error("Failed to delete cookie file %s: %s", cookiefile, e)
    else:
        _LOGGER.debug("Cookie file %s does not exist, nothing to delete.", cookiefile)


async def load_cookie(login: AlexaLogin) -> dict[str, str] | None:
    """
    Restore a previously saved Alexa session into the login's aiohttp session.

    :param login: The AlexaLogin whose session cookie jar should be populated.
    """
    cookiefile = login._cookiefile[0] if login._cookiefile else None
    if not cookiefile or not await asyncio.to_thread(os.path.exists, cookiefile):
        return None
    if login._session is None:
        login._create_session()
    # aiohttp 3.14 saves the cookie jar as JSON, which alexapy's own load_cookie()
    # cannot parse. Load it with aiohttp's loader into the session jar (preserving
    # the cookie domains required for auth) and return the cookies for login().
    cookie_jar = login._session.cookie_jar
    assert isinstance(cookie_jar, aiohttp.CookieJar)
    try:
        await asyncio.to_thread(cookie_jar.load, cookiefile)
    except (OSError, EOFError, TypeError, ValueError, AttributeError) as ex:
        _LOGGER.debug("Error loading cookie from %s: %s", cookiefile, ex)
        return None
    cookies = login._get_cookies_from_session()
    return cast("dict[str, str]", cookies) if cookies else None


async def _request_with_session(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    json_data: dict[str, Any] | None,
    timeout: int,
    auth: BasicAuth | None,
) -> str:
    """
    Handle an API request with a provided aiohttp session.

    :param session: The aiohttp session to use.
    :param method: HTTP method to use for the request.
    :param url: Full URL for the request.
    :param json_data: Optional JSON payload or query params.
    :param timeout: Timeout in seconds for the request.
    :param auth: Optional basic auth credentials.
    """
    request_timeout = aiohttp.ClientTimeout(total=timeout)
    if method.upper() == "GET":
        async with session.get(url, params=json_data, timeout=request_timeout, auth=auth) as resp:
            resp_text = await resp.text()
            if resp.status < 200 or resp.status >= 300:
                msg = (
                    f"Failed API request to {url}: Status code: {resp.status}, "
                    f"Response: {resp_text}"
                )
                _LOGGER.error(msg)
                raise ActionUnavailable(msg)
            return resp_text

    async with session.request(
        method.upper(),
        url,
        json=json_data,
        timeout=request_timeout,
        auth=auth,
    ) as resp:
        resp_text = await resp.text()
        if resp.status < 200 or resp.status >= 300:
            msg = f"Failed API request to {url}: Status code: {resp.status}, Response: {resp_text}"
            _LOGGER.error(msg)
            raise ActionUnavailable(msg)
        return resp_text


async def api_request(
    provider: PlayerProvider,
    endpoint: str,
    method: str = "POST",
    json_data: dict[str, Any] | None = None,
    timeout: int = 10,
) -> str:
    """
    Send a request to the configured Music Assistant / Alexa API.

    Returns the response text on success or raises `ActionUnavailable` on failure.
    """
    url, auth = _api_target(provider, endpoint)
    return await _request_with_session(
        provider.mass.http_session, method, url, json_data, timeout, auth
    )


def _api_target(provider: PlayerProvider, endpoint: str) -> tuple[str, BasicAuth | None]:
    """Return the full URL and basic auth for an endpoint of the configured API."""
    username = provider.get_setup_value(CONF_API_BASIC_AUTH_USERNAME)
    password = provider.get_setup_value(CONF_API_BASIC_AUTH_PASSWORD)

    auth = None
    if username is not None and password is not None:
        auth = BasicAuth(str(username), str(password))

    api_url = str(provider.get_setup_value(CONF_API_URL) or "")
    return f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}", auth


async def get_playback_state(provider: AlexaProvider, player_id: str) -> dict[str, Any] | None:
    """
    Return what the Echo last reported to the skill API for this player.

    Returns None when the API cannot be reached or lacks the endpoint (older skill
    versions); unlike api_request this is quiet, as it runs on every poll.
    """
    if provider.mass.loop.time() < provider._playback_state_retry_at:
        return None
    url, auth = _api_target(provider, "/alexa/playback-state")
    try:
        async with provider.mass.http_session.get(
            url,
            params={"player_id": player_id},
            auth=auth,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json(content_type=None)
    except aiohttp.ClientError, TimeoutError:
        provider._playback_state_retry_at = provider.mass.loop.time() + PLAYBACK_STATE_RETRY_DELAY
        return None
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


class AlexaDevice:
    """Representation of an Alexa Device."""

    _device_type: str
    device_serial_number: str
    _device_family: str
    _cluster_members: str
    _locale: str


class AlexaPlayer(Player):
    """Implementation of an Alexa Player."""

    def __init__(
        self,
        provider: AlexaProvider,
        player_id: str,
        device: AlexaDevice,
    ) -> None:
        """Initialize AlexaPlayer."""
        super().__init__(provider, player_id)
        self.device = device
        self._attr_supported_features = {
            PlayerFeature.PLAY_MEDIA,
            PlayerFeature.VOLUME_SET,
            PlayerFeature.PAUSE,
        }
        self._attr_name = player_id
        self._attr_device_info = DeviceInfo()
        self._attr_powered = False
        self._attr_available = True
        # Keep track of the last metadata we pushed to avoid unnecessary uploads
        self._last_meta_checksum: str | None = None
        # Keep last stream url pushed (set in play_media)
        self._last_stream_url: str | None = None
        self._attr_poll_interval = POLL_INTERVAL
        self._last_observed_at: float | None = None
        # Bumped by every playback command, so a report fetched before it is discarded.
        self._playback_generation = 0

    @property
    def needs_poll(self) -> bool:
        """Return if the Echo can move on its own: while playing or paused, not stopped."""
        return self.playback_state != PlaybackState.IDLE

    async def poll(self) -> None:
        """
        Correct elapsed time and play/pause state from the Echo's own reports.

        In flow mode the elapsed time is the position in the whole flow stream, which is
        exactly the stream offset the Echo reports; MA maps it to the right queue item.
        """
        if self.playback_state == PlaybackState.IDLE:
            # Stopped by MA; the Echo has nothing newer to say about this queue.
            return
        generation = self._playback_generation
        report = await get_playback_state(cast("AlexaProvider", self.provider), self.player_id)
        if generation != self._playback_generation:
            # A playback command ran while the report was on its way; it describes the past.
            return
        if not report or not report.get("current"):
            # No report (yet), or the Echo is still on an older stream.
            return
        state = ECHO_PLAYBACK_STATES.get(str(report.get("activity")))
        offset_ms = report.get("offsetMs")
        age = report.get("ageSeconds")
        observed_at = report.get("observedAt")
        if (
            state is None
            or not isinstance(offset_ms, (int, float))
            or not isinstance(age, (int, float))
            or observed_at == self._last_observed_at
        ):
            return
        self._last_observed_at = observed_at
        if state == PlaybackState.PAUSED:
            # In flow mode the queue only follows a playing player's position (see
            # _get_flow_queue_stream_index), so move it to the paused offset as playing
            # first; otherwise it stays at the clock estimate until playback resumes.
            self._attr_playback_state = PlaybackState.PLAYING
            self._attr_elapsed_time = offset_ms / 1000
            self._attr_elapsed_time_last_updated = time.time()
            self.update_state()
        # The age keeps the anchor on MA's clock even if the skill API runs elsewhere.
        self._attr_elapsed_time = offset_ms / 1000
        self._attr_elapsed_time_last_updated = time.time() - age
        self._attr_playback_state = state
        self.update_state()

    @property
    def requires_flow_mode(self) -> bool:
        """Return if the player requires flow mode."""
        return True

    @property
    def api(self) -> AlexaAPI:
        """Get the AlexaAPI instance for this player."""
        provider = cast("AlexaProvider", self.provider)
        return AlexaAPI(self.device, provider.login)

    async def stop(self) -> None:
        """Handle STOP command on the player."""
        self._playback_generation += 1
        provider = cast("AlexaProvider", self.provider)

        utter = await provider.get_intent_utterance("AMAZON.StopIntent", "stop")
        await self.api.run_custom(utter)

        self._attr_current_media = None
        self._attr_playback_state = PlaybackState.IDLE
        self.update_state()

    async def play(self) -> None:
        """Handle PLAY command on the player."""
        self._playback_generation += 1
        provider = cast("AlexaProvider", self.provider)

        utter = await provider.get_intent_utterance("AMAZON.ResumeIntent", "resume")
        await self.api.run_custom(utter)

        self._attr_playback_state = PlaybackState.PLAYING
        self.update_state()

    async def pause(self) -> None:
        """Handle PAUSE command on the player."""
        self._playback_generation += 1
        provider = cast("AlexaProvider", self.provider)

        utter = await provider.get_intent_utterance("AMAZON.PauseIntent", "pause")
        await self.api.run_custom(utter)

        self._attr_playback_state = PlaybackState.PAUSED
        self.update_state()

    async def volume_set(self, volume_level: int) -> None:
        """Handle VOLUME_SET command on the player."""
        await self.api.set_volume(volume_level / 100)
        self._attr_volume_level = volume_level
        self.update_state()

    async def play_media(self, media: PlayerMedia) -> None:
        """Handle PLAY MEDIA on the player."""
        self._playback_generation += 1
        stream_url = await self.provider.mass.streams.resolve_stream_url(self.player_id, media)

        payload = {
            "streamUrl": stream_url,
            "title": media.title,
            "artist": media.artist,
            "album": media.album,
            "imageUrl": media.image_url,
        }

        await api_request(
            self.provider,
            "/ma/push-url",
            method="POST",
            json_data=payload,
            timeout=10,
        )

        # Save last pushed stream url so metadata updates can reuse it
        self._last_stream_url = stream_url

        alexa_locale = self.provider.config.get_value(CONF_ALEXA_LANGUAGE)

        ask_command_key = f"play_audio_{alexa_locale or 'default'}"

        if ask_command_key not in ALEXA_LANGUAGE_COMMANDS:
            _LOGGER.debug(
                "Ask command key %s not found in ALEXA_LANGUAGE_COMMANDS.",
                ask_command_key,
            )
            ask_command_key = "play_audio_default"

        _LOGGER.debug(
            "Using ask command key: %s -> %s",
            ask_command_key,
            ALEXA_LANGUAGE_COMMANDS[ask_command_key],
        )

        await self.api.run_custom(ALEXA_LANGUAGE_COMMANDS[ask_command_key])
        self._last_observed_at = None
        self._attr_elapsed_time = 0
        self._attr_elapsed_time_last_updated = time.time()
        self._attr_playback_state = PlaybackState.PLAYING
        self._attr_current_media = media
        self.update_state()

    def on_player_media_updated(self) -> None:
        """
        Handle callback when the current media of the player is updated.

        Upload the stream URL and media metadata (title/artist/album/imageUrl)
        to the configured Music Assistant / Alexa API so the Alexa side can
        display/update the playing item.
        """
        if self._last_stream_url is None:
            return

        media = self.state.current_media

        async def _upload_metadata() -> None:
            stream_url = self._last_stream_url
            if media is not None:
                title = media.title
                artist = media.artist
                album = media.album
                image_url = media.image_url
            else:
                return

            meta_checksum = f"{stream_url}-{album}-{artist}-{title}-{image_url}"
            if meta_checksum == self._last_meta_checksum:
                return

            payload = {
                "streamUrl": stream_url,
                "title": title,
                "artist": artist,
                "album": album,
                "imageUrl": image_url,
            }

            await api_request(
                self.provider, "/ma/push-url", method="POST", json_data=payload, timeout=10
            )

            # store last pushed values
            self._last_meta_checksum = meta_checksum

        self.mass.create_task(_upload_metadata())


class AlexaProvider(PlayerProvider):
    """Implementation of an Alexa Device Provider."""

    login: AlexaLogin
    devices: dict[str, AlexaDevice]

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """
        Return the config entries for the Alexa provider.

        Amazon credentials, the companion API details and the login are collected by the
        interactive setup flow (see ``setup_flow.py``); only the language option is
        configurable here.
        """
        return (
            ConfigEntry(
                key=CONF_ALEXA_LANGUAGE,
                type=ConfigEntryType.STRING,
                required=True,
                options=[
                    ConfigValueOption("en-US"),
                    ConfigValueOption("en-CA"),
                    ConfigValueOption("de-DE"),
                    ConfigValueOption("es-ES"),
                    ConfigValueOption("fr-FR"),
                    ConfigValueOption("fr-CA"),
                    ConfigValueOption("it-IT"),
                    ConfigValueOption("pt-BR"),
                    ConfigValueOption("nl-NL"),
                ],
                default_value="en-US",  # choose a sensible default
            ),
        )

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self.devices = {}
        self._playback_state_retry_at = -math.inf
        self._intents: list[dict[str, Any]] | None = None
        self._invocation_name: str | None = None

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""
        self.login = AlexaLogin(
            url=str(self.get_setup_value(CONF_URL)),
            email=str(self.get_setup_value(CONF_USERNAME)),
            password=str(self.get_setup_value(CONF_PASSWORD)),
            outputpath=lambda x: x,
        )

        cookie_dir = os.path.join(self.mass.storage_path, ".alexa")
        await asyncio.to_thread(os.makedirs, cookie_dir, exist_ok=True)
        cookie_path = os.path.join(
            cookie_dir, f"alexa_media.{self.get_setup_value(CONF_USERNAME)}.pickle"
        )
        self.login._cookiefile = [self.login._outputpath(cookie_path)]

        await self.login.login(cookies=await load_cookie(self.login))

        devices = await AlexaAPI.get_devices(self.login)

        if devices is None:
            return

        alexa_locale = str(self.config.get_value(CONF_ALEXA_LANGUAGE, "en-US"))

        for device in devices:
            if device.get("capabilities") and "MUSIC_SKILL" in device.get("capabilities"):
                dev_name = device["accountName"]
                player_id = dev_name
                # Initialize AlexaDevice
                device_object = AlexaDevice()
                device_object._device_type = device["deviceType"]
                device_object.device_serial_number = device["serialNumber"]
                device_object._device_family = device["deviceOwnerCustomerId"]
                device_object._cluster_members = device["clusterMembers"]
                device_object._locale = alexa_locale
                self.devices[player_id] = device_object

                # Create AlexaPlayer instance
                player = AlexaPlayer(self, player_id, device_object)
                await self.mass.players.register_or_update(player)

        await self._load_intents()

    @lock
    async def _load_intents(self) -> None:
        """Load intents from the configured API and cache them on the provider."""
        resp = await api_request(self, "/alexa/intents", method="GET", timeout=5)
        data = json.loads(resp)
        if isinstance(data, dict):
            # cache invocationName if present
            self._invocation_name = data.get("invocationName")
            self._intents = data.get("intents", []) or []
        else:
            self._intents = []

    async def get_intent_utterance(self, intent_name: str, default: str) -> str:
        """
        Return the first utterance for the given intent name (cached).

        If intents are not yet cached, attempt to load them.
        """
        if self._intents is None:
            await self._load_intents()

        for intent in self._intents or []:
            if intent.get("intent") == intent_name:
                utts = cast("list[str]", intent.get("utterances") or [])
                if utts:
                    utter = utts[0]
                    if self._invocation_name:
                        inv = self._invocation_name.strip()
                        return f"{inv} {utter}".strip()
                    return utter
        return default
