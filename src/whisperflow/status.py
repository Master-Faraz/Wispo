"""Tray icon for the background process.

Ubuntu's app indicator extension watches org.kde.StatusNotifierItem. The
icon name changes while recording and while the finished sentence is typed.
A left click opens settings. The menu can also quit the process.
"""

from __future__ import annotations

import threading

from jeepney import DBusAddress, HeaderFields, MessageType, new_error, new_method_return, new_signal

from whisperflow.desktop import DESKTOP_ID
from jeepney.bus_messages import message_bus
from jeepney.io.blocking import open_dbus_connection

_BUS_NAME = "org.whisperflow.Service"
_ITEM_PATH = "/StatusNotifierItem"
_MENU_PATH = "/Menu"
_ITEM = "org.kde.StatusNotifierItem"
_MENU = "com.canonical.dbusmenu"
_PROPS = "org.freedesktop.DBus.Properties"
_LAUNCHER = "com.canonical.Unity.LauncherEntry"
_LAUNCHER_PATH = "/com/canonical/Unity/LauncherEntry"

ICON_NAMES = {
    "idle": "audio-input-microphone-symbolic",
    "recording": "media-record-symbolic",
    "processing": "system-run-symbolic",
}
TITLES = {
    "idle": "WhisperFlow",
    "recording": "WhisperFlow — listening",
    "processing": "WhisperFlow — writing",
}


class StatusIcon:
    def __init__(self, on_settings, on_quit) -> None:
        self._on_settings = on_settings
        self._on_quit = on_quit
        self._state = "idle"
        self._dirty = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="whisperflow-status", daemon=True)
        self._conn = None

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=5)
        if self._error is not None:
            raise self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)

    def set_state(self, state: str) -> None:
        with self._lock:
            if state == self._state:
                return
            self._state = state
            self._dirty = True

    def _snapshot(self) -> tuple[str, str]:
        with self._lock:
            return self._state, ICON_NAMES.get(self._state, ICON_NAMES["idle"])

    def _run(self) -> None:
        try:
            self._conn = open_dbus_connection(bus="SESSION")
            claimed = self._conn.send_and_get_reply(
                message_bus.RequestName(_BUS_NAME, flags=4),
                timeout=5,
            )
            if claimed.body[0] != 1:
                self._error = RuntimeError("WhisperFlow is already running")
                return
            self._register()
        except Exception as exc:
            self._error = exc
            return
        finally:
            self._ready.set()
        if self._error is not None:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            return
        try:
            while not self._stop.is_set():
                self._flush_state()
                try:
                    message = self._conn.receive(timeout=0.2)
                except TimeoutError:
                    continue
                if message.header.message_type == MessageType.method_call:
                    self._handle(message)
        finally:
            if self._conn is not None:
                self._conn.close()

    def _register(self) -> None:
        address = DBusAddress(
            "/StatusNotifierWatcher",
            bus_name="org.kde.StatusNotifierWatcher",
            interface="org.kde.StatusNotifierWatcher",
        )
        from jeepney import new_method_call

        self._conn.send_and_get_reply(
            new_method_call(address, "RegisterStatusNotifierItem", "s", (_BUS_NAME,)),
            timeout=5,
        )

    def _flush_state(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
            state = self._state
        icon = ICON_NAMES.get(state, ICON_NAMES["idle"])
        item = DBusAddress(_ITEM_PATH, interface=_ITEM)
        self._conn.send(new_signal(item, "NewIcon"))
        self._conn.send(new_signal(item, "NewTitle"))
        self._conn.send(
            new_signal(
                DBusAddress(_ITEM_PATH, interface=_PROPS),
                "PropertiesChanged",
                "sa{sv}as",
                (
                    _ITEM,
                    {"IconName": ("s", icon), "Title": ("s", TITLES.get(state, TITLES["idle"]))},
                    [],
                ),
            )
        )
        urgent = state != "idle"
        self._conn.send(
            new_signal(
                DBusAddress(_LAUNCHER_PATH, interface=_LAUNCHER),
                "Update",
                "sa{sv}",
                (
                    f"application://{DESKTOP_ID}",
                    {"urgent": ("b", urgent), "progress-visible": ("b", False)},
                ),
            )
        )

    def _handle(self, message) -> None:
        path = message.header.fields.get(HeaderFields.path, "")
        interface = message.header.fields.get(HeaderFields.interface, "")
        member = message.header.fields.get(HeaderFields.member, "")
        if interface == _PROPS and member == "Get":
            self._get_property(message, path)
            return
        if interface == _PROPS and member == "GetAll":
            self._get_all(message, path)
            return
        if member == "Introspect":
            self._reply(message, "s", (_INTROSPECT,))
            return
        if path == "/org/whisperflow/Service" and member == "OpenSettings":
            self._reply(message)
            self._on_settings()
            return
        if path == "/org/whisperflow/Service" and member == "Quit":
            self._reply(message)
            self._on_quit()
            return
        if path == _ITEM_PATH and member == "Activate":
            self._reply(message)
            self._on_settings()
            return
        if path == _ITEM_PATH and member == "SecondaryActivate":
            self._reply(message)
            self._on_quit()
            return
        if path == _MENU_PATH and member == "Event":
            self._reply(message)
            self._menu_event(message.body)
            return
        if path == _MENU_PATH and member == "GetLayout":
            self._reply(message, "u(ia{sv}av)", (1, _menu_layout()))
            return
        if path == _MENU_PATH and member == "AboutToShow":
            self._reply(message, "b", (False,))
            return
        if path == _MENU_PATH and member == "GetGroupProperties":
            self._reply(message, "a(ia{sv})", ([],))
            return
        if member in {"ContextMenu", "Scroll", "ProvideXdgActivationToken", "XAyatanaSecondaryActivate"}:
            self._reply(message)
            return
        self._conn.send(
            new_error(message, "org.freedesktop.DBus.Error.UnknownMethod", "s", (member,))
        )

    def _get_property(self, message, path: str) -> None:
        name = message.body[1] if len(message.body) > 1 else ""
        props = self._properties(path)
        if name not in props:
            self._conn.send(
                new_error(message, "org.freedesktop.DBus.Error.UnknownProperty", "s", (name,))
            )
            return
        self._reply(message, "v", (props[name],))

    def _get_all(self, message, path: str) -> None:
        self._reply(message, "a{sv}", (self._properties(path),))

    def _properties(self, path: str) -> dict:
        _state, icon = self._snapshot()
        if path == _MENU_PATH:
            return {"Status": ("s", "normal"), "TextDirection": ("s", "ltr"), "Version": ("u", 3)}
        return {
            "Category": ("s", "ApplicationStatus"),
            "Id": ("s", "whisperflow"),
            "Title": ("s", TITLES.get(_state, TITLES["idle"])),
            "Status": ("s", "Active"),
            "WindowId": ("i", 0),
            "IconThemePath": ("s", ""),
            "Menu": ("o", _MENU_PATH),
            "ItemIsMenu": ("b", False),
            "IconName": ("s", icon),
            "IconPixmap": ("a(iiay)", []),
            "OverlayIconName": ("s", ""),
            "OverlayIconPixmap": ("a(iiay)", []),
            "AttentionIconName": ("s", ICON_NAMES["recording"]),
            "AttentionIconPixmap": ("a(iiay)", []),
            "AttentionMovieName": ("s", ""),
        }

    def _menu_event(self, body) -> None:
        if not body:
            return
        item_id = body[0]
        event = body[1] if len(body) > 1 else ""
        if event != "clicked":
            return
        if item_id == 1:
            self._on_settings()
        elif item_id == 2:
            self._on_quit()

    def _reply(self, message, signature: str | None = None, body: tuple = ()) -> None:
        self._conn.send(new_method_return(message, signature, body))


def _menu_layout():
    settings = (1, {"label": ("s", "Settings"), "enabled": ("b", True)}, [])
    quit_item = (2, {"label": ("s", "Quit"), "enabled": ("b", True)}, [])
    children = [("v", ("(ia{sv}av)", settings)), ("v", ("(ia{sv}av)", quit_item))]
    return (0, {"children-display": ("s", "submenu")}, children)


_INTROSPECT = """<!DOCTYPE node PUBLIC "-//freedesktop//DTD D-BUS Object Introspection 1.0//EN"
"http://www.freedesktop.org/standards/dbus/1.0/introspect.dtd">
<node>
  <interface name="org.kde.StatusNotifierItem">
    <method name="Activate"><arg type="i" direction="in"/><arg type="i" direction="in"/></method>
    <method name="SecondaryActivate"><arg type="i" direction="in"/><arg type="i" direction="in"/></method>
    <method name="ContextMenu"><arg type="i" direction="in"/><arg type="i" direction="in"/></method>
    <property name="IconName" type="s" access="read"/>
    <property name="Title" type="s" access="read"/>
    <property name="Status" type="s" access="read"/>
    <property name="Menu" type="o" access="read"/>
  </interface>
  <interface name="com.canonical.dbusmenu">
    <method name="GetLayout">
      <arg type="i" direction="in"/>
      <arg type="i" direction="in"/>
      <arg type="as" direction="in"/>
      <arg type="u" direction="out"/>
      <arg type="(ia{sv}av)" direction="out"/>
    </method>
    <method name="Event">
      <arg type="i" direction="in"/>
      <arg type="s" direction="in"/>
      <arg type="v" direction="in"/>
      <arg type="u" direction="in"/>
    </method>
    <method name="AboutToShow"><arg type="i" direction="in"/><arg type="b" direction="out"/></method>
  </interface>
  <interface name="org.whisperflow.Service">
    <method name="OpenSettings"/>
    <method name="Quit"/>
  </interface>
</node>
"""
