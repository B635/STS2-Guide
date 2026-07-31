"""Strict, fail-closed compatibility checks for a packaged STS2 Guide build.

The manifest parser and assessment functions stay pure.  The Host observes
its own packaged components, while each v7 event supplies the game/Mod side
of the handshake before the processor may publish advice.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence


CURRENT_MANIFEST_VERSION = 2
ENABLED_STATUS = "enabled"
_ALLOWED_STATUSES = frozenset({"disabled", "pending_validation", ENABLED_STATUS})
CAPABILITY_KEYS = (
    "card_reward",
    "route_choice",
    "merchant",
    "rest_site",
    "neow_choice",
    "event_choice",
    "deck_edit",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GAME_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


class CompatibilityManifestError(ValueError):
    """The compatibility manifest is absent, malformed, or unsupported."""


@dataclass(frozen=True)
class GuideCompatibility:
    version: str


@dataclass(frozen=True)
class GameCompatibility:
    version: str
    sts2_dll_sha256: str


@dataclass(frozen=True)
class ModCompatibility:
    id: str
    version: str
    producer: str


@dataclass(frozen=True)
class ProtocolCompatibility:
    state_event_schema_version: int
    minimum_replay_schema_version: int
    producer: str


@dataclass(frozen=True)
class SqliteCompatibility:
    schema_version: int
    snapshot_id: str
    knowledge_sha256: str
    community_scores_sha256: str


@dataclass(frozen=True)
class PolicyCompatibility:
    bundle_version: str
    card_reward_version: str
    route_version: str
    merchant_version: str
    campfire_version: str
    neow_version: str
    event_version: str
    deck_edit_version: str


@dataclass(frozen=True)
class CompatibilityManifest:
    manifest_version: int
    release_fingerprint: str
    status: str
    guide: GuideCompatibility
    game: GameCompatibility
    mod: ModCompatibility
    protocol: ProtocolCompatibility
    sqlite: SqliteCompatibility
    policy: PolicyCompatibility
    capabilities: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            MappingProxyType(dict(self.capabilities)),
        )

    @property
    def is_enabled(self) -> bool:
        return self.status == ENABLED_STATUS

    def capability_is_enabled(self, event_type: str) -> bool:
        status = self.capabilities.get(str(event_type))
        return status == ENABLED_STATUS


@dataclass(frozen=True)
class RuntimeComponents:
    """Observed runtime component identities.

    Every field is optional so callers can report what they managed to
    observe.  Missing values never inherit a manifest default: assessment
    records a blocking issue instead.
    """

    guide_version: Optional[str] = None
    game_version: Optional[str] = None
    sts2_dll_sha256: Optional[str] = None
    mod_id: Optional[str] = None
    mod_version: Optional[str] = None
    protocol_schema_version: Optional[int] = None
    sqlite_schema_version: Optional[int] = None
    sqlite_snapshot_id: Optional[str] = None
    knowledge_sha256: Optional[str] = None
    community_scores_sha256: Optional[str] = None
    policy_bundle_version: Optional[str] = None
    card_reward_policy_version: Optional[str] = None
    route_policy_version: Optional[str] = None
    merchant_policy_version: Optional[str] = None
    campfire_policy_version: Optional[str] = None
    neow_policy_version: Optional[str] = None
    event_policy_version: Optional[str] = None
    deck_edit_policy_version: Optional[str] = None


@dataclass(frozen=True)
class EventHandshake:
    """Version identity carried by one state event producer."""

    game_version: Optional[str]
    schema_version: Optional[int]
    source: Optional[str]
    producer_id: Optional[str]
    producer_version: Optional[str]
    game_assembly_sha256: Optional[str]
    release_fingerprint: Optional[str]


@dataclass(frozen=True)
class CompatibilityIssue:
    code: str
    component: str
    expected: Optional[str]
    actual: Optional[str]


@dataclass(frozen=True)
class CompatibilityAssessment:
    issues: tuple[CompatibilityIssue, ...]

    @property
    def compatible(self) -> bool:
        return not self.issues

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


def default_manifest_path() -> Path:
    """Return the repository or PyInstaller-bundled manifest path."""

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root) / "packaging" / "compatibility.json"
    return Path(__file__).resolve().parents[1] / "packaging" / "compatibility.json"


def load_compatibility_manifest(
    path: str | Path | None = None,
) -> CompatibilityManifest:
    """Load one strict manifest.

    Unknown keys, missing keys, duplicate JSON keys, type coercion and
    unsupported manifest versions are rejected instead of silently ignored.
    """

    manifest_path = Path(path) if path is not None else default_manifest_path()
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CompatibilityManifestError(
            f"compatibility manifest cannot be read: {manifest_path}"
        ) from exc

    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, CompatibilityManifestError) as exc:
        raise CompatibilityManifestError(
            f"compatibility manifest is not valid strict JSON: {manifest_path}"
        ) from exc

    try:
        return _parse_manifest(payload)
    except CompatibilityManifestError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise CompatibilityManifestError(
            f"compatibility manifest is malformed: {manifest_path}"
        ) from exc


def assess_runtime_compatibility(
    manifest: CompatibilityManifest,
    observed: RuntimeComponents,
) -> CompatibilityAssessment:
    """Compare every release component against the single manifest."""

    issues: list[CompatibilityIssue] = []
    if not manifest.is_enabled:
        issues.append(
            CompatibilityIssue(
                code="manifest_not_enabled",
                component="manifest.status",
                expected=ENABLED_STATUS,
                actual=manifest.status,
            )
        )

    checks: Sequence[tuple[str, object, object, bool]] = (
        ("guide.version", manifest.guide.version, observed.guide_version, False),
        ("game.version", manifest.game.version, observed.game_version, False),
        (
            "game.sts2_dll_sha256",
            manifest.game.sts2_dll_sha256,
            observed.sts2_dll_sha256,
            True,
        ),
        ("mod.id", manifest.mod.id, observed.mod_id, False),
        ("mod.version", manifest.mod.version, observed.mod_version, False),
        (
            "protocol.state_event_schema_version",
            manifest.protocol.state_event_schema_version,
            observed.protocol_schema_version,
            False,
        ),
        (
            "sqlite.schema_version",
            manifest.sqlite.schema_version,
            observed.sqlite_schema_version,
            False,
        ),
        (
            "sqlite.snapshot_id",
            manifest.sqlite.snapshot_id,
            observed.sqlite_snapshot_id,
            False,
        ),
        (
            "sqlite.knowledge_sha256",
            manifest.sqlite.knowledge_sha256,
            observed.knowledge_sha256,
            True,
        ),
        (
            "sqlite.community_scores_sha256",
            manifest.sqlite.community_scores_sha256,
            observed.community_scores_sha256,
            True,
        ),
        (
            "policy.bundle_version",
            manifest.policy.bundle_version,
            observed.policy_bundle_version,
            False,
        ),
        (
            "policy.card_reward_version",
            manifest.policy.card_reward_version,
            observed.card_reward_policy_version,
            False,
        ),
        (
            "policy.route_version",
            manifest.policy.route_version,
            observed.route_policy_version,
            False,
        ),
        (
            "policy.merchant_version",
            manifest.policy.merchant_version,
            observed.merchant_policy_version,
            False,
        ),
        (
            "policy.campfire_version",
            manifest.policy.campfire_version,
            observed.campfire_policy_version,
            False,
        ),
        (
            "policy.neow_version",
            manifest.policy.neow_version,
            observed.neow_policy_version,
            False,
        ),
        (
            "policy.event_version",
            manifest.policy.event_version,
            observed.event_policy_version,
            False,
        ),
        (
            "policy.deck_edit_version",
            manifest.policy.deck_edit_version,
            observed.deck_edit_policy_version,
            False,
        ),
    )
    for component, expected, actual, hash_value in checks:
        _append_comparison_issue(
            issues,
            component=component,
            expected=expected,
            actual=actual,
            normalize_hash=hash_value,
        )
    return CompatibilityAssessment(tuple(issues))


def assess_local_runtime_compatibility(
    manifest: CompatibilityManifest,
    observed: RuntimeComponents,
) -> CompatibilityAssessment:
    """Validate components owned by the packaged Host process.

    Game and Mod identities are deliberately excluded here because the Host
    can only learn those facts from a v7 event emitted by the running game.
    """

    issues: list[CompatibilityIssue] = []
    if not manifest.is_enabled:
        issues.append(
            CompatibilityIssue(
                code="manifest_not_enabled",
                component="manifest.status",
                expected=ENABLED_STATUS,
                actual=manifest.status,
            )
        )
    checks: Sequence[tuple[str, object, object, bool]] = (
        ("guide.version", manifest.guide.version, observed.guide_version, False),
        (
            "protocol.state_event_schema_version",
            manifest.protocol.state_event_schema_version,
            observed.protocol_schema_version,
            False,
        ),
        (
            "sqlite.schema_version",
            manifest.sqlite.schema_version,
            observed.sqlite_schema_version,
            False,
        ),
        (
            "sqlite.snapshot_id",
            manifest.sqlite.snapshot_id,
            observed.sqlite_snapshot_id,
            False,
        ),
        (
            "sqlite.knowledge_sha256",
            manifest.sqlite.knowledge_sha256,
            observed.knowledge_sha256,
            True,
        ),
        (
            "sqlite.community_scores_sha256",
            manifest.sqlite.community_scores_sha256,
            observed.community_scores_sha256,
            True,
        ),
        (
            "policy.bundle_version",
            manifest.policy.bundle_version,
            observed.policy_bundle_version,
            False,
        ),
        (
            "policy.card_reward_version",
            manifest.policy.card_reward_version,
            observed.card_reward_policy_version,
            False,
        ),
        (
            "policy.route_version",
            manifest.policy.route_version,
            observed.route_policy_version,
            False,
        ),
        (
            "policy.merchant_version",
            manifest.policy.merchant_version,
            observed.merchant_policy_version,
            False,
        ),
        (
            "policy.campfire_version",
            manifest.policy.campfire_version,
            observed.campfire_policy_version,
            False,
        ),
        (
            "policy.neow_version",
            manifest.policy.neow_version,
            observed.neow_policy_version,
            False,
        ),
        (
            "policy.event_version",
            manifest.policy.event_version,
            observed.event_policy_version,
            False,
        ),
        (
            "policy.deck_edit_version",
            manifest.policy.deck_edit_version,
            observed.deck_edit_policy_version,
            False,
        ),
    )
    for component, expected, actual, hash_value in checks:
        _append_comparison_issue(
            issues,
            component=component,
            expected=expected,
            actual=actual,
            normalize_hash=hash_value,
        )
    return CompatibilityAssessment(tuple(issues))


def assess_event_handshake(
    manifest: CompatibilityManifest,
    handshake: EventHandshake,
) -> CompatibilityAssessment:
    """Validate the exact live producer/game/protocol tuple.

    Replay support for older fixtures is deliberately not accepted here.
    ``minimum_replay_schema_version`` documents offline compatibility only;
    live events must use the manifest's exact production schema.
    """

    issues: list[CompatibilityIssue] = []
    if not manifest.is_enabled:
        issues.append(
            CompatibilityIssue(
                code="manifest_not_enabled",
                component="manifest.status",
                expected=ENABLED_STATUS,
                actual=manifest.status,
            )
        )
    for component, expected, actual in (
        ("event.game_version", manifest.game.version, handshake.game_version),
        (
            "event.schema_version",
            manifest.protocol.state_event_schema_version,
            handshake.schema_version,
        ),
        ("event.source", manifest.protocol.producer, handshake.source),
        ("event.producer_id", manifest.mod.id, handshake.producer_id),
        (
            "event.producer_version",
            manifest.mod.version,
            handshake.producer_version,
        ),
        (
            "event.game_assembly_sha256",
            manifest.game.sts2_dll_sha256,
            handshake.game_assembly_sha256,
        ),
        (
            "event.release_fingerprint",
            manifest.release_fingerprint,
            handshake.release_fingerprint,
        ),
    ):
        _append_comparison_issue(
            issues,
            component=component,
            expected=expected,
            actual=actual,
            normalize_hash=(
                component
                in {
                    "event.game_assembly_sha256",
                    "event.release_fingerprint",
                }
            ),
        )
    return CompatibilityAssessment(tuple(issues))


def _parse_manifest(payload: object) -> CompatibilityManifest:
    root = _require_mapping(payload, "manifest")
    _require_exact_keys(
        root,
        "manifest",
        {
            "manifest_version",
            "release_fingerprint",
            "status",
            "guide",
            "game",
            "mod",
            "protocol",
            "sqlite",
            "policy",
            "capabilities",
        },
    )
    manifest_version = _require_int(root, "manifest_version", minimum=1)
    if manifest_version != CURRENT_MANIFEST_VERSION:
        raise CompatibilityManifestError(
            f"unsupported compatibility manifest version: {manifest_version}"
        )
    status = _require_string(root, "status")
    if status not in _ALLOWED_STATUSES:
        raise CompatibilityManifestError(f"unsupported manifest status: {status}")

    guide = _section(root, "guide", {"version"})
    game = _section(root, "game", {"version", "sts2_dll_sha256"})
    mod = _section(root, "mod", {"id", "version", "producer"})
    protocol = _section(
        root,
        "protocol",
        {
            "state_event_schema_version",
            "minimum_replay_schema_version",
            "producer",
        },
    )
    sqlite = _section(
        root,
        "sqlite",
        {
            "schema_version",
            "snapshot_id",
            "knowledge_sha256",
            "community_scores_sha256",
        },
    )
    policy = _section(
        root,
        "policy",
        {
            "bundle_version",
            "card_reward_version",
            "route_version",
            "merchant_version",
            "campfire_version",
            "neow_version",
            "event_version",
            "deck_edit_version",
        },
    )
    capabilities = _section(
        root,
        "capabilities",
        set(CAPABILITY_KEYS),
    )

    game_version = _require_string(game, "version")
    if not _GAME_VERSION_RE.fullmatch(game_version):
        raise CompatibilityManifestError("game.version must be an exact x.y.z version")

    protocol_version = _require_int(
        protocol,
        "state_event_schema_version",
        minimum=1,
    )
    replay_version = _require_int(
        protocol,
        "minimum_replay_schema_version",
        minimum=1,
    )
    if replay_version > protocol_version:
        raise CompatibilityManifestError(
            "protocol.minimum_replay_schema_version exceeds the live schema"
        )
    mod_producer = _require_string(mod, "producer")
    protocol_producer = _require_string(protocol, "producer")
    if mod_producer != protocol_producer:
        raise CompatibilityManifestError(
            "mod.producer and protocol.producer must match"
        )

    parsed = CompatibilityManifest(
        manifest_version=manifest_version,
        release_fingerprint=_require_sha256(
            root,
            "release_fingerprint",
        ),
        status=status,
        guide=GuideCompatibility(version=_require_string(guide, "version")),
        game=GameCompatibility(
            version=game_version,
            sts2_dll_sha256=_require_sha256(game, "sts2_dll_sha256"),
        ),
        mod=ModCompatibility(
            id=_require_string(mod, "id"),
            version=_require_string(mod, "version"),
            producer=mod_producer,
        ),
        protocol=ProtocolCompatibility(
            state_event_schema_version=protocol_version,
            minimum_replay_schema_version=replay_version,
            producer=protocol_producer,
        ),
        sqlite=SqliteCompatibility(
            schema_version=_require_int(sqlite, "schema_version", minimum=1),
            snapshot_id=_require_string(sqlite, "snapshot_id"),
            knowledge_sha256=_require_sha256(sqlite, "knowledge_sha256"),
            community_scores_sha256=_require_sha256(
                sqlite,
                "community_scores_sha256",
            ),
        ),
        policy=PolicyCompatibility(
            bundle_version=_require_string(policy, "bundle_version"),
            card_reward_version=_require_string(policy, "card_reward_version"),
            route_version=_require_string(policy, "route_version"),
            merchant_version=_require_string(policy, "merchant_version"),
            campfire_version=_require_string(policy, "campfire_version"),
            neow_version=_require_string(policy, "neow_version"),
            event_version=_require_string(policy, "event_version"),
            deck_edit_version=_require_string(policy, "deck_edit_version"),
        ),
        capabilities={
            name: _require_capability_status(capabilities, name)
            for name in CAPABILITY_KEYS
        },
    )
    computed_fingerprint = compute_release_fingerprint(parsed)
    if parsed.release_fingerprint != computed_fingerprint:
        raise CompatibilityManifestError(
            "release_fingerprint does not match the canonical release "
            "component payload"
        )
    return parsed


def compute_release_fingerprint(
    manifest: CompatibilityManifest,
) -> str:
    """Hash one canonical identity for every packaged release component.

    ``status`` is deliberately excluded: promoting a byte-identical package
    from pending validation to enabled must not change its release identity.
    The manifest version is included so a future canonicalization contract
    cannot collide with version 1.
    """

    payload = {
        "manifest_version": manifest.manifest_version,
        "guide": {"version": manifest.guide.version},
        "game": {
            "version": manifest.game.version,
            "sts2_dll_sha256": manifest.game.sts2_dll_sha256,
        },
        "mod": {
            "id": manifest.mod.id,
            "version": manifest.mod.version,
            "producer": manifest.mod.producer,
        },
        "protocol": {
            "state_event_schema_version": (
                manifest.protocol.state_event_schema_version
            ),
            "minimum_replay_schema_version": (
                manifest.protocol.minimum_replay_schema_version
            ),
            "producer": manifest.protocol.producer,
        },
        "sqlite": {
            "schema_version": manifest.sqlite.schema_version,
            "snapshot_id": manifest.sqlite.snapshot_id,
            "knowledge_sha256": manifest.sqlite.knowledge_sha256,
            "community_scores_sha256": (
                manifest.sqlite.community_scores_sha256
            ),
        },
        "policy": {
            "bundle_version": manifest.policy.bundle_version,
            "card_reward_version": manifest.policy.card_reward_version,
            "route_version": manifest.policy.route_version,
            "merchant_version": manifest.policy.merchant_version,
            "campfire_version": manifest.policy.campfire_version,
            "neow_version": manifest.policy.neow_version,
            "event_version": manifest.policy.event_version,
            "deck_edit_version": manifest.policy.deck_edit_version,
        },
        "capabilities": {
            name: manifest.capabilities[name]
            for name in CAPABILITY_KEYS
        },
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompatibilityManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CompatibilityManifestError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    label: str,
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise CompatibilityManifestError(
            f"{label} keys mismatch; missing={missing}, unknown={unknown}"
        )


def _section(
    root: Mapping[str, Any],
    name: str,
    keys: set[str],
) -> Mapping[str, Any]:
    value = _require_mapping(root[name], name)
    _require_exact_keys(value, name, keys)
    return value


def _require_string(value: Mapping[str, Any], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item.strip() or item != item.strip():
        raise CompatibilityManifestError(f"{key} must be a non-empty trimmed string")
    return item


def _require_capability_status(
    value: Mapping[str, Any],
    key: str,
) -> str:
    status = _require_string(value, key)
    if status not in _ALLOWED_STATUSES:
        raise CompatibilityManifestError(
            f"unsupported capability status: {key}={status}"
        )
    return status


def _require_int(
    value: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
) -> int:
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
        raise CompatibilityManifestError(
            f"{key} must be an integer greater than or equal to {minimum}"
        )
    return item


def _require_sha256(value: Mapping[str, Any], key: str) -> str:
    item = _require_string(value, key)
    if not _SHA256_RE.fullmatch(item):
        raise CompatibilityManifestError(
            f"{key} must be a lowercase 64-character SHA-256 digest"
        )
    return item


def _display(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _append_comparison_issue(
    issues: list[CompatibilityIssue],
    *,
    component: str,
    expected: object,
    actual: object,
    normalize_hash: bool = False,
) -> None:
    if actual is None or (isinstance(actual, str) and not actual.strip()):
        issues.append(
            CompatibilityIssue(
                code="component_missing",
                component=component,
                expected=_display(expected),
                actual=None,
            )
        )
        return

    expected_value = expected
    actual_value = actual
    if normalize_hash and isinstance(actual, str):
        actual_value = actual.lower()
    if actual_value != expected_value:
        issues.append(
            CompatibilityIssue(
                code="component_mismatch",
                component=component,
                expected=_display(expected),
                actual=_display(actual),
            )
        )
