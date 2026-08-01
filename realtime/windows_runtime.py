"""Small, injectable Windows primitives used by the public-beta shell.

The production implementations deliberately use operating-system identities:
one named kernel mutex for ownership, named auto-reset events for activation
and graceful shutdown, and the executable path reported by the process
handle.  Window titles and process basenames are not trusted as game identity.
"""
from __future__ import annotations

import ntpath
import os
import sys
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Iterable, Protocol


DEFAULT_MUTEX_NAME = r"Local\STS2Guide.Controller.v1"
DEFAULT_ACTIVATION_EVENT_NAME = r"Local\STS2Guide.Activate.v1"
DEFAULT_SHUTDOWN_EVENT_NAME = r"Local\STS2Guide.Shutdown.v1"
EXPECTED_GAME_EXE_NAME = "SlayTheSpire2.exe"


class WindowsRuntimeUnavailable(RuntimeError):
    """The required Win32 API is absent or could not be initialized."""


class WindowsRuntimeError(RuntimeError):
    """A Win32 operation failed after the API was initialized."""


class KernelApi(Protocol):
    def create_mutex(self, name: str) -> tuple[object, bool]: ...

    def create_event(self, name: str) -> object: ...

    def signal_event(self, handle: object) -> None: ...

    def event_is_signaled(self, handle: object) -> bool: ...

    def close_handle(self, handle: object) -> None: ...


class CtypesKernelApi:
    """ctypes adapter for the five kernel operations the controller needs."""

    ERROR_ALREADY_EXISTS = 183
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 258

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise WindowsRuntimeUnavailable("Win32 kernel API is unavailable")
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.argtypes = (
                wintypes.LPVOID,
                wintypes.BOOL,
                wintypes.LPCWSTR,
            )
            kernel32.CreateMutexW.restype = wintypes.HANDLE
            kernel32.CreateEventW.argtypes = (
                wintypes.LPVOID,
                wintypes.BOOL,
                wintypes.BOOL,
                wintypes.LPCWSTR,
            )
            kernel32.CreateEventW.restype = wintypes.HANDLE
            kernel32.SetEvent.argtypes = (wintypes.HANDLE,)
            kernel32.SetEvent.restype = wintypes.BOOL
            kernel32.WaitForSingleObject.argtypes = (
                wintypes.HANDLE,
                wintypes.DWORD,
            )
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
        except (AttributeError, ImportError, OSError) as exc:
            raise WindowsRuntimeUnavailable(
                "Win32 kernel API could not be initialized"
            ) from exc
        self._ctypes = ctypes
        self._kernel32 = kernel32

    def _last_error(self, operation: str) -> WindowsRuntimeError:
        code = int(self._ctypes.get_last_error())
        return WindowsRuntimeError(f"{operation} failed with Win32 error {code}")

    def create_mutex(self, name: str) -> tuple[object, bool]:
        self._ctypes.set_last_error(0)
        handle = self._kernel32.CreateMutexW(None, False, name)
        error = int(self._ctypes.get_last_error())
        if not handle:
            raise self._last_error("CreateMutexW")
        return handle, error == self.ERROR_ALREADY_EXISTS

    def create_event(self, name: str) -> object:
        # Auto-reset is intentional: one activation wakes one primary poll.
        handle = self._kernel32.CreateEventW(None, False, False, name)
        if not handle:
            raise self._last_error("CreateEventW")
        return handle

    def signal_event(self, handle: object) -> None:
        if not self._kernel32.SetEvent(handle):
            raise self._last_error("SetEvent")

    def event_is_signaled(self, handle: object) -> bool:
        result = int(self._kernel32.WaitForSingleObject(handle, 0))
        if result == self.WAIT_OBJECT_0:
            return True
        if result == self.WAIT_TIMEOUT:
            return False
        raise self._last_error("WaitForSingleObject")

    def close_handle(self, handle: object) -> None:
        if handle and not self._kernel32.CloseHandle(handle):
            raise self._last_error("CloseHandle")


class NamedInstanceCoordinator:
    """Own the controller mutex or activate the already-running instance."""

    def __init__(
        self,
        api: KernelApi,
        *,
        mutex_name: str = DEFAULT_MUTEX_NAME,
        activation_event_name: str = DEFAULT_ACTIVATION_EVENT_NAME,
        shutdown_event_name: str = DEFAULT_SHUTDOWN_EVENT_NAME,
    ) -> None:
        if not mutex_name or not activation_event_name or not shutdown_event_name:
            raise ValueError("named Windows objects require non-empty names")
        self._api = api
        self.mutex_name = mutex_name
        self.activation_event_name = activation_event_name
        self.shutdown_event_name = shutdown_event_name
        self._mutex: object | None = None
        self._activation_event: object | None = None
        self._shutdown_event: object | None = None
        self._primary = False

    @property
    def is_primary(self) -> bool:
        return self._primary

    def acquire_or_activate(self) -> bool:
        if (
            self._mutex is not None
            or self._activation_event is not None
            or self._shutdown_event is not None
        ):
            raise RuntimeError("instance coordinator was already acquired")
        mutex, already_exists = self._api.create_mutex(self.mutex_name)
        try:
            activation = self._api.create_event(self.activation_event_name)
            try:
                shutdown = self._api.create_event(self.shutdown_event_name)
            except Exception:
                self._api.close_handle(activation)
                raise
        except Exception:
            self._api.close_handle(mutex)
            raise
        if already_exists:
            try:
                self._api.signal_event(activation)
            finally:
                self._api.close_handle(shutdown)
                self._api.close_handle(activation)
                self._api.close_handle(mutex)
            return False
        self._mutex = mutex
        self._activation_event = activation
        self._shutdown_event = shutdown
        self._primary = True
        return True

    def consume_activation(self) -> bool:
        if not self._primary or self._activation_event is None:
            return False
        return self._api.event_is_signaled(self._activation_event)

    def consume_shutdown(self) -> bool:
        if not self._primary or self._shutdown_event is None:
            return False
        return self._api.event_is_signaled(self._shutdown_event)

    def request_existing_shutdown(self) -> bool:
        """Signal the primary controller without taking over its mutex."""

        if (
            self._mutex is not None
            or self._activation_event is not None
            or self._shutdown_event is not None
        ):
            raise RuntimeError("instance coordinator is already in use")
        mutex, already_exists = self._api.create_mutex(self.mutex_name)
        if not already_exists:
            self._api.close_handle(mutex)
            return False
        shutdown = None
        try:
            shutdown = self._api.create_event(self.shutdown_event_name)
            self._api.signal_event(shutdown)
            return True
        finally:
            if shutdown is not None:
                self._api.close_handle(shutdown)
            self._api.close_handle(mutex)

    def close(self) -> None:
        shutdown = self._shutdown_event
        activation, mutex = self._activation_event, self._mutex
        self._shutdown_event = None
        self._activation_event = None
        self._mutex = None
        self._primary = False
        errors: list[Exception] = []
        for handle in (shutdown, activation, mutex):
            if handle is None:
                continue
            try:
                self._api.close_handle(handle)
            except Exception as exc:  # release the other handle as well
                errors.append(exc)
        if errors:
            raise errors[0]


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    executable_path: str

    def __post_init__(self) -> None:
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid < 1:
            raise ValueError("process pid must be a positive integer")
        if not isinstance(self.executable_path, str) or not self.executable_path.strip():
            raise ValueError("process executable path must not be empty")


class ProcessApi(Protocol):
    def iter_processes(self) -> Iterable[ProcessInfo]: ...


def canonical_windows_path(value: str | os.PathLike[str]) -> str:
    """Normalize a Windows path without touching the host filesystem."""

    text = str(value).strip()
    if not text:
        raise ValueError("executable path must not be empty")
    path = PureWindowsPath(text)
    if not path.is_absolute():
        raise ValueError("executable path must be absolute")
    return ntpath.normcase(ntpath.normpath(str(path)))


class ExactGameProcessDetector:
    """Detect STS2 only when the kernel-reported executable path matches."""

    def __init__(
        self,
        expected_executable: str | os.PathLike[str],
        api: ProcessApi,
    ) -> None:
        expected = canonical_windows_path(expected_executable)
        if ntpath.basename(expected).casefold() != EXPECTED_GAME_EXE_NAME.casefold():
            raise ValueError(
                f"expected executable must be named {EXPECTED_GAME_EXE_NAME}"
            )
        self.expected_executable = expected
        self._api = api

    def matching_processes(self) -> tuple[ProcessInfo, ...]:
        matches: list[ProcessInfo] = []
        for process in self._api.iter_processes():
            try:
                observed = canonical_windows_path(process.executable_path)
            except ValueError:
                continue
            if observed == self.expected_executable:
                matches.append(process)
        return tuple(matches)

    def is_running(self) -> bool:
        return bool(self.matching_processes())


class CtypesWindowsProcessApi:
    """Enumerate processes and query their full image names via kernel32."""

    TH32CS_SNAPPROCESS = 0x00000002
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    ERROR_NO_MORE_FILES = 18

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise WindowsRuntimeUnavailable("Win32 process API is unavailable")
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

            class PROCESSENTRY32W(ctypes.Structure):
                _fields_ = (
                    ("dwSize", wintypes.DWORD),
                    ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260),
                )

            kernel32.CreateToolhelp32Snapshot.argtypes = (
                wintypes.DWORD,
                wintypes.DWORD,
            )
            kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
            kernel32.Process32FirstW.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(PROCESSENTRY32W),
            )
            kernel32.Process32FirstW.restype = wintypes.BOOL
            kernel32.Process32NextW.argtypes = kernel32.Process32FirstW.argtypes
            kernel32.Process32NextW.restype = wintypes.BOOL
            kernel32.OpenProcess.argtypes = (
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            )
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.QueryFullProcessImageNameW.argtypes = (
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.LPWSTR,
                ctypes.POINTER(wintypes.DWORD),
            )
            kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
        except (AttributeError, ImportError, OSError) as exc:
            raise WindowsRuntimeUnavailable(
                "Win32 process API could not be initialized"
            ) from exc
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = kernel32
        self._entry_type = PROCESSENTRY32W

    def _full_image_name(self, pid: int) -> str | None:
        handle = self._kernel32.OpenProcess(
            self.PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not handle:
            return None
        try:
            capacity = 32768
            buffer = self._ctypes.create_unicode_buffer(capacity)
            size = self._wintypes.DWORD(capacity)
            if not self._kernel32.QueryFullProcessImageNameW(
                handle,
                0,
                buffer,
                self._ctypes.byref(size),
            ):
                return None
            return buffer.value[: int(size.value)]
        finally:
            self._kernel32.CloseHandle(handle)

    def iter_processes(self) -> Iterable[ProcessInfo]:
        invalid_handle_value = self._wintypes.HANDLE(-1).value
        snapshot = self._kernel32.CreateToolhelp32Snapshot(
            self.TH32CS_SNAPPROCESS,
            0,
        )
        if not snapshot or snapshot == invalid_handle_value:
            code = int(self._ctypes.get_last_error())
            raise WindowsRuntimeError(
                f"CreateToolhelp32Snapshot failed with Win32 error {code}"
            )
        try:
            entry = self._entry_type()
            entry.dwSize = self._ctypes.sizeof(self._entry_type)
            ok = bool(self._kernel32.Process32FirstW(snapshot, self._ctypes.byref(entry)))
            if not ok:
                code = int(self._ctypes.get_last_error())
                if code == self.ERROR_NO_MORE_FILES:
                    return
                raise WindowsRuntimeError(
                    f"Process32FirstW failed with Win32 error {code}"
                )
            while ok:
                pid = int(entry.th32ProcessID)
                path = self._full_image_name(pid) if pid > 0 else None
                if path:
                    yield ProcessInfo(pid=pid, executable_path=path)
                ok = bool(
                    self._kernel32.Process32NextW(
                        snapshot,
                        self._ctypes.byref(entry),
                    )
                )
            code = int(self._ctypes.get_last_error())
            if code not in (0, self.ERROR_NO_MORE_FILES):
                raise WindowsRuntimeError(
                    f"Process32NextW failed with Win32 error {code}"
                )
        finally:
            self._kernel32.CloseHandle(snapshot)
