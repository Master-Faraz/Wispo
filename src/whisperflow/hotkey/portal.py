"""Hold-to-talk through GNOME's global shortcut portal.

Reading the keyboard directly still delivers Left Ctrl, but GNOME also sees
those keys. Super would open the overview. A shortcut registered here is
handled by the compositor: Activated is the press, Deactivated is the
release, and the chord is not delivered to the rest of the desktop.

The trigger string uses the XDG shortcut spec. Super is named LOGO there,
so the preferred chord is CTRL+LOGO. The first bind shows a dialog where
the combination can be confirmed or changed.
"""

from __future__ import annotations

import threading
from collections import deque

from jeepney import HeaderFields, MatchRule
from jeepney.bus_messages import message_bus
from jeepney.io.blocking import open_dbus_connection

from whisperflow.injection.portal import (
    PortalError,
    _checked,
    _token,
    _vardict,
    portal_call,
)

_SHORTCUTS = "org.freedesktop.portal.GlobalShortcuts"
_SHORTCUT_ID = "push-to-talk"
# XDG shortcut spec: LOGO is the Super key. This is a suggestion; the dialog
# can replace it.
_PREFERRED = "CTRL+LOGO"


class PortalShortcut:
    def __init__(self, trigger: str = _PREFERRED) -> None:
        self.trigger = trigger
        self.held = threading.Event()
        self.description = "Ctrl+Super"
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="whisperflow-shortcut", daemon=True)
        self._conn = None
        self._session: str | None = None

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=180):
            self.stop()
            raise PortalError("timed out waiting for the shortcut dialog")
        if self._error is not None:
            self.stop()
            raise PortalError(str(self._error)) from self._error

    def stop(self) -> None:
        self._stop.set()
        self.held.clear()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            self._bind()
        except Exception as exc:
            self._error = exc
            self._ready.set()
            self._shutdown()
            return
        self._ready.set()
        try:
            if not self._stop.is_set():
                self._listen()
        finally:
            self._shutdown()

    def _bind(self) -> None:
        self._conn = open_dbus_connection(bus="SESSION")
        sender = self._conn.unique_name.lstrip(":").replace(".", "_")
        handle = _token()
        session_token = _token()
        created = portal_call(
            self._conn,
            _SHORTCUTS,
            "CreateSession",
            "a{sv}",
            (
                [
                    ("handle_token", ("s", handle)),
                    ("session_handle_token", ("s", session_token)),
                ],
            ),
            handle,
            sender,
            30,
        )
        session = created.get("session_handle")
        if not isinstance(session, str) or not session.startswith("/"):
            raise PortalError(f"shortcut portal did not return a session ({created})")
        self._session = session

        list_token = _token()
        listed = portal_call(
            self._conn,
            _SHORTCUTS,
            "ListShortcuts",
            "oa{sv}",
            (session, [("handle_token", ("s", list_token))]),
            list_token,
            sender,
            30,
        )
        existing = describe_binding(listed)
        if existing:
            self.description = existing
            return

        bind_token = _token()
        bound = portal_call(
            self._conn,
            _SHORTCUTS,
            "BindShortcuts",
            "oa(sa{sv})sa{sv}",
            (
                session,
                [
                    (
                        _SHORTCUT_ID,
                        [
                            ("description", ("s", "Hold to transcribe speech")),
                            ("preferred_trigger", ("s", self.trigger)),
                        ],
                    )
                ],
                "",
                [("handle_token", ("s", bind_token))],
            ),
            bind_token,
            sender,
            180,
        )
        self.description = describe_binding(bound) or self.description

    def _listen(self) -> None:
        rule = MatchRule(
            type="signal",
            interface=_SHORTCUTS,
        )
        queue: deque = deque()
        with self._conn.filter(rule, queue=queue):
            _checked(self._conn, message_bus.AddMatch(rule), timeout=5)
            while not self._stop.is_set():
                if not queue:
                    try:
                        self._conn.recv_messages(timeout=0.2)
                    except TimeoutError:
                        continue
                while queue:
                    self._on_signal(queue.popleft())

    def _on_signal(self, message) -> None:
        member = message.header.fields.get(HeaderFields.member)
        body = message.body or ()
        if len(body) < 2 or body[1] != _SHORTCUT_ID:
            return
        if member == "Activated":
            self.held.set()
        elif member == "Deactivated":
            self.held.clear()

    def _shutdown(self) -> None:
        conn, self._conn = self._conn, None
        session, self._session = self._session, None
        if conn is None:
            return
        try:
            if session:
                from jeepney import DBusAddress, new_method_call

                address = DBusAddress(
                    session,
                    bus_name="org.freedesktop.portal.Desktop",
                    interface="org.freedesktop.portal.Session",
                )
                _checked(conn, new_method_call(address, "Close"), timeout=2)
        except PortalError:
            pass
        finally:
            conn.close()


def describe_binding(results: dict) -> str:
    """User-visible chord from a ListShortcuts or BindShortcuts result."""
    shortcuts = results.get("shortcuts") or []
    for item in shortcuts:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        shortcut_id, props = item[0], item[1]
        if shortcut_id != _SHORTCUT_ID:
            continue
        parsed = _vardict(props)
        text = parsed.get("trigger_description")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return "Ctrl+Super"
    return ""
