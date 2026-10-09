"""Type into the focused window through the XDG RemoteDesktop portal.

Mutter does not implement the virtual-keyboard protocol that wtype needs.
The portal's NotifyKeyboardKeycode path is implemented by the GNOME portal
on this machine and delivers evdev keycodes to the focused application.
The session is opened once and kept for every later insert() call.

The first successful Start() shows a permission dialog. A restore token is
saved so later launches can skip it when the desktop still honors the grant.
"""

from __future__ import annotations

import secrets
import time
from collections import deque
from pathlib import Path

from jeepney import DBusAddress, HeaderFields, MessageType, new_method_call
from jeepney.bus_messages import MatchRule, message_bus
from jeepney.io.blocking import open_dbus_connection

from whisperflow.injection.base import TextInjector
from whisperflow.injection.keymap import KEY_LEFTSHIFT, key_events

_BUS = "org.freedesktop.portal.Desktop"
_PATH = "/org/freedesktop/portal/desktop"
_REMOTE = "org.freedesktop.portal.RemoteDesktop"
_KEYBOARD = 1
_PERSIST_UNTIL_REVOKED = 2
_RESPONSE = {1: "input permission was cancelled", 2: "input permission dialog closed"}


class PortalError(RuntimeError):
    pass


def _state_dir() -> Path:
    root = Path.home() / ".local" / "state" / "whisperflow"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _vardict(results) -> dict:
    pairs = results.items() if isinstance(results, dict) else results
    parsed = {}
    for key, value in pairs:
        # jeepney represents a D-Bus variant as (signature, payload).
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
            parsed[key] = value[1]
        else:
            parsed[key] = value
    return parsed


class PortalKeyboard(TextInjector):
    def __init__(self, token_path: Path | None = None, key_gap_s: float = 0.005) -> None:
        self.token_path = token_path or (_state_dir() / "portal-restore-token")
        # A press and its release in the same instant is dropped by some text
        # fields. The gap is between keys, not a new process per utterance.
        self.key_gap_s = key_gap_s
        self._conn = None
        self._session: str | None = None
        self._sender = ""

    def open(self) -> None:
        if self._session is not None:
            return
        self._conn = open_dbus_connection(bus="SESSION")
        self._sender = self._conn.unique_name.lstrip(":").replace(".", "_")
        try:
            self._open_session()
        except Exception:
            self.close()
            raise

    def _open_session(self) -> None:
        handle = _token()
        session_token = _token()
        created = self._request(
            "CreateSession",
            "a{sv}",
            (
                [
                    ("handle_token", ("s", handle)),
                    ("session_handle_token", ("s", session_token)),
                ],
            ),
            handle,
            timeout=30,
        )
        session = created.get("session_handle")
        if not isinstance(session, str) or not session.startswith("/"):
            raise PortalError(f"portal did not return a session ({created})")
        self._session = session

        select_token = _token()
        options: list[tuple] = [
            ("handle_token", ("s", select_token)),
            ("types", ("u", _KEYBOARD)),
            ("persist_mode", ("u", _PERSIST_UNTIL_REVOKED)),
        ]
        saved = self._load_token()
        if saved:
            options.append(("restore_token", ("s", saved)))
        self._request(
            "SelectDevices",
            "oa{sv}",
            (session, options),
            select_token,
            timeout=30,
        )

        start_token = _token()
        # Start is what shows the permission dialog. Give the user time to
        # read it. SelectDevices above does not.
        started = self._request(
            "Start",
            "osa{sv}",
            (session, "", [("handle_token", ("s", start_token))]),
            start_token,
            timeout=180,
        )
        devices = int(started.get("devices", 0))
        if devices & _KEYBOARD == 0:
            self.close()
            raise PortalError("keyboard input was not granted")
        restore = started.get("restore_token")
        if isinstance(restore, str) and restore:
            self._save_token(restore)

    def insert(self, text: str) -> None:
        if not text:
            return
        if self._session is None or self._conn is None:
            self.open()
        try:
            self._do_insert(text)
        except PortalError:
            # Session may have been revoked upon screen lock or idle timeout.
            # Reconnect using the stored restore token and retry once.
            self.close()
            self.open()
            self._do_insert(text)

    def _do_insert(self, text: str) -> None:
        events = key_events(text)
        try:
            for code, state in events:
                self._notify(code, state)
                if state == 0:
                    time.sleep(self.key_gap_s)
        finally:
            try:
                self._notify(KEY_LEFTSHIFT, 0)
            except PortalError:
                pass

    def close(self) -> None:
        session, self._session = self._session, None
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            if session:
                address = DBusAddress(
                    session,
                    bus_name=_BUS,
                    interface="org.freedesktop.portal.Session",
                )
                _checked(conn, new_method_call(address, "Close"), timeout=5)
        except PortalError:
            pass
        finally:
            conn.close()

    def _notify(self, keycode: int, state: int) -> None:
        address = DBusAddress(_PATH, bus_name=_BUS, interface=_REMOTE)
        _checked(
            self._conn,
            new_method_call(
                address,
                "NotifyKeyboardKeycode",
                "oa{sv}iu",
                (self._session, [], keycode, state),
            ),
            timeout=5,
        )

    def _request(self, method: str, signature: str, body: tuple, token: str, timeout: float) -> dict:
        return portal_call(
            self._conn,
            _REMOTE,
            method,
            signature,
            body,
            token,
            self._sender,
            timeout,
        )

    def _load_token(self) -> str:
        if not self.token_path.is_file():
            return ""
        return self.token_path.read_text(encoding="utf-8").strip()

    def _save_token(self, token: str) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(token + "\n", encoding="utf-8")
        self.token_path.chmod(0o600)


def _token() -> str:
    return secrets.token_hex(8)


def portal_call(conn, interface: str, method: str, signature: str, body: tuple, token: str, sender: str, timeout: float) -> dict:
    """Call a portal method and wait for its Request.Response signal.

    The signal's sender header is the portal's unique bus name, not
    org.freedesktop.portal.Desktop, so matching on that well-known name
    drops the Response and the call waits forever.
    """
    expected = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"
    rule = MatchRule(
        type="signal",
        interface="org.freedesktop.portal.Request",
        member="Response",
    )
    queue: deque = deque()
    address = DBusAddress(_PATH, bus_name=_BUS, interface=interface)
    with conn.filter(rule, queue=queue):
        _checked(conn, message_bus.AddMatch(rule), timeout=5)
        try:
            replied = _checked(
                conn,
                new_method_call(address, method, signature, body),
                timeout=timeout,
            )
            request_path = replied[0] if replied else expected
            message = _take_response(conn, queue, request_path, timeout)
        finally:
            try:
                _checked(conn, message_bus.RemoveMatch(rule), timeout=5)
            except PortalError:
                pass
    code = int(message.body[0])
    results = _vardict(message.body[1]) if len(message.body) > 1 else {}
    if code != 0:
        reason = _RESPONSE.get(code, f"portal request failed ({code})")
        raise PortalError(reason)
    return results


def _take_response(conn, queue: deque, request_path: str, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        while queue:
            message = queue.popleft()
            path = message.header.fields.get(HeaderFields.path)
            if path == request_path:
                return message
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PortalError(
                "timed out waiting for the desktop portal. "
                "If a permission dialog is open, allow it and try again."
            )
        message = conn.recv_until_filtered(queue, timeout=remaining)
        path = message.header.fields.get(HeaderFields.path)
        if path == request_path:
            return message


def _checked(conn, message, timeout: float):
    reply = conn.send_and_get_reply(message, timeout=timeout)
    if reply.header.message_type == MessageType.error:
        name = reply.header.fields.get(HeaderFields.error_name, "org.freedesktop.DBus.Error")
        detail = reply.body[0] if reply.body else ""
        raise PortalError(f"{name}: {detail}")
    return reply.body
