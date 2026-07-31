from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from realtime.compatibility import (
    CompatibilityManifestError,
    EventHandshake,
    RuntimeComponents,
    assess_event_handshake,
    assess_local_runtime_compatibility,
    assess_runtime_compatibility,
    compute_release_fingerprint,
    default_manifest_path,
    load_compatibility_manifest,
)
from realtime.checkpoint import ActiveRunCheckpointStore
from realtime.file_bridge import GameStateFileBridge
from realtime.host import _observe_local_components, build_bridge
from realtime.processor import RealtimeEventProcessor
from realtime.protocol import CURRENT_SCHEMA_VERSION, GameStateEvent
from realtime.version import GUIDE_VERSION
from storage.relational import RelationalRepository


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "packaging" / "compatibility.json"


def _enabled_manifest(manifest):
    return replace(
        manifest,
        status="enabled",
        capabilities={
            key: "enabled" for key in manifest.capabilities
        },
    )


class CompatibilityManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_compatibility_manifest(MANIFEST_PATH)

    def test_checked_in_manifest_enables_validated_01101_baseline(self):
        self.assertEqual(self.manifest.game.version, "0.110.1")
        self.assertEqual(
            self.manifest.release_fingerprint,
            compute_release_fingerprint(self.manifest),
        )
        self.assertEqual(
            self.manifest.game.sts2_dll_sha256,
            "7c446efabf80614c429b5088e87101423aa5bb4c04fc3e73393261f6e6d404fd",
        )
        self.assertEqual(self.manifest.status, "enabled")
        self.assertTrue(self.manifest.is_enabled)

    def test_default_manifest_path_points_to_packaged_source(self):
        self.assertEqual(default_manifest_path(), MANIFEST_PATH)
        self.assertEqual(
            load_compatibility_manifest(),
            self.manifest,
        )

    def test_manifest_matches_packaged_component_sources(self):
        mod_manifest = json.loads(
            (
                ROOT
                / "mod"
                / "STS2Guide.ReadOnlyExporter"
                / "STS2GuideReadOnlyExporter.json"
            ).read_text(encoding="utf-8")
        )
        mod_protocol_source = (
            ROOT
            / "mod"
            / "STS2Guide.ReadOnlyExporter"
            / "ProtocolModels.cs"
        ).read_text(encoding="utf-8")
        protocol_schema = json.loads(
            (ROOT / "protocol" / "state-event.schema.json").read_text(
                encoding="utf-8"
            )
        )
        from storage.relational import SCHEMA_VERSION
        from advisor.versions import (
            CAMPFIRE_POLICY_VERSION,
            CARD_REWARD_POLICY_VERSION,
            DECK_EDIT_POLICY_VERSION,
            EVENT_POLICY_VERSION,
            MERCHANT_POLICY_VERSION,
            NEOW_POLICY_VERSION,
            POLICY_BUNDLE_VERSION,
            ROUTE_POLICY_VERSION,
        )

        self.assertEqual(self.manifest.guide.version, GUIDE_VERSION)
        self.assertEqual(self.manifest.mod.id, mod_manifest["id"])
        self.assertEqual(self.manifest.mod.version, mod_manifest["version"])
        self.assertIn(
            f'ProducerId = "{self.manifest.mod.id}"',
            mod_protocol_source,
        )
        self.assertIn(
            f'ProducerVersion = "{self.manifest.mod.version}"',
            mod_protocol_source,
        )
        self.assertIn(
            self.manifest.release_fingerprint,
            mod_protocol_source,
        )
        self.assertIn(
            f'ProducerSource = "{self.manifest.protocol.producer}"',
            mod_protocol_source,
        )
        self.assertEqual(
            self.manifest.protocol.state_event_schema_version,
            max(protocol_schema["properties"]["schema_version"]["enum"]),
        )
        self.assertEqual(self.manifest.sqlite.schema_version, SCHEMA_VERSION)
        self.assertEqual(
            self.manifest.protocol.state_event_schema_version,
            CURRENT_SCHEMA_VERSION,
        )
        self.assertEqual(
            self.manifest.policy.bundle_version,
            POLICY_BUNDLE_VERSION,
        )
        self.assertEqual(
            self.manifest.policy.card_reward_version,
            CARD_REWARD_POLICY_VERSION,
        )
        self.assertEqual(
            self.manifest.policy.route_version,
            ROUTE_POLICY_VERSION,
        )
        self.assertEqual(
            self.manifest.policy.merchant_version,
            MERCHANT_POLICY_VERSION,
        )
        self.assertEqual(
            self.manifest.policy.campfire_version,
            CAMPFIRE_POLICY_VERSION,
        )
        self.assertEqual(
            self.manifest.policy.neow_version,
            NEOW_POLICY_VERSION,
        )
        self.assertEqual(
            self.manifest.policy.event_version,
            EVENT_POLICY_VERSION,
        )
        self.assertEqual(
            self.manifest.policy.deck_edit_version,
            DECK_EDIT_POLICY_VERSION,
        )
        self.assertEqual(
            set(self.manifest.capabilities),
            {
                "card_reward",
                "route_choice",
                "merchant",
                "rest_site",
                "neow_choice",
                "event_choice",
                "deck_edit",
            },
        )
        self.assertEqual(self.manifest.capabilities["card_reward"], "enabled")
        self.assertEqual(self.manifest.capabilities["route_choice"], "enabled")
        self.assertEqual(
            {
                value
                for name, value in self.manifest.capabilities.items()
                if name not in {"card_reward", "route_choice"}
            },
            {"pending_validation"},
        )
        for name, expected in (
            ("knowledge.json", self.manifest.sqlite.knowledge_sha256),
            (
                "community_scores.json",
                self.manifest.sqlite.community_scores_sha256,
            ),
        ):
            digest = hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, name)

        packaging_spec = (ROOT / "packaging" / "sts2-guide.spec").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '(str(root / "packaging" / "compatibility.json"), "packaging")',
            packaging_spec,
        )

    def test_manifest_rejects_malformed_and_ambiguous_shapes(self):
        source = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cases = {}

        missing = json.loads(json.dumps(source))
        del missing["game"]["version"]
        cases["missing key"] = json.dumps(missing)

        unknown = json.loads(json.dumps(source))
        unknown["game"]["build"] = "unknown"
        cases["unknown key"] = json.dumps(unknown)

        coerced = json.loads(json.dumps(source))
        coerced["protocol"]["state_event_schema_version"] = "6"
        cases["coerced integer"] = json.dumps(coerced)

        boolean = json.loads(json.dumps(source))
        boolean["sqlite"]["schema_version"] = True
        cases["boolean integer"] = json.dumps(boolean)

        bad_hash = json.loads(json.dumps(source))
        bad_hash["game"]["sts2_dll_sha256"] = "not-a-hash"
        cases["bad hash"] = json.dumps(bad_hash)

        uppercase_hash = json.loads(json.dumps(source))
        uppercase_hash["game"]["sts2_dll_sha256"] = source["game"][
            "sts2_dll_sha256"
        ].upper()
        cases["noncanonical hash"] = json.dumps(uppercase_hash)

        producer_mismatch = json.loads(json.dumps(source))
        producer_mismatch["protocol"]["producer"] = "different-producer"
        cases["producer mismatch"] = json.dumps(producer_mismatch)

        stale_fingerprint = json.loads(json.dumps(source))
        stale_fingerprint["guide"]["version"] = "0.1.0-other"
        cases["stale release fingerprint"] = json.dumps(stale_fingerprint)

        unsupported_status = json.loads(json.dumps(source))
        unsupported_status["status"] = "validated-ish"
        cases["unknown status"] = json.dumps(unsupported_status)

        duplicate = MANIFEST_PATH.read_text(encoding="utf-8").replace(
            '"manifest_version": 2,',
            '"manifest_version": 2, "manifest_version": 2,',
            1,
        )
        cases["duplicate key"] = duplicate

        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "compatibility.json"
            for label, content in cases.items():
                with self.subTest(label=label):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(CompatibilityManifestError):
                        load_compatibility_manifest(path)

    def test_missing_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "missing.json"
            with self.assertRaises(CompatibilityManifestError):
                load_compatibility_manifest(path)


class RuntimeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        pending = load_compatibility_manifest(MANIFEST_PATH)
        self.manifest = _enabled_manifest(pending)
        self.matching = RuntimeComponents(
            guide_version=self.manifest.guide.version,
            game_version=self.manifest.game.version,
            sts2_dll_sha256=self.manifest.game.sts2_dll_sha256.upper(),
            mod_id=self.manifest.mod.id,
            mod_version=self.manifest.mod.version,
            protocol_schema_version=(
                self.manifest.protocol.state_event_schema_version
            ),
            sqlite_schema_version=self.manifest.sqlite.schema_version,
            sqlite_snapshot_id=self.manifest.sqlite.snapshot_id,
            knowledge_sha256=self.manifest.sqlite.knowledge_sha256,
            community_scores_sha256=(
                self.manifest.sqlite.community_scores_sha256
            ),
            policy_bundle_version=self.manifest.policy.bundle_version,
            card_reward_policy_version=(
                self.manifest.policy.card_reward_version
            ),
            route_policy_version=self.manifest.policy.route_version,
            merchant_policy_version=self.manifest.policy.merchant_version,
            campfire_policy_version=self.manifest.policy.campfire_version,
            neow_policy_version=self.manifest.policy.neow_version,
            event_policy_version=self.manifest.policy.event_version,
            deck_edit_policy_version=self.manifest.policy.deck_edit_version,
        )

    def test_all_exact_components_pass_when_release_is_enabled(self):
        assessment = assess_runtime_compatibility(
            self.manifest,
            self.matching,
        )
        self.assertTrue(assessment.compatible)
        self.assertEqual(assessment.issues, ())

    def test_pending_manifest_blocks_even_exact_components(self):
        pending = replace(self.manifest, status="pending_validation")
        assessment = assess_runtime_compatibility(pending, self.matching)
        self.assertFalse(assessment.compatible)
        self.assertEqual(assessment.reason_codes, ("manifest_not_enabled",))

    def test_missing_runtime_components_all_block(self):
        assessment = assess_runtime_compatibility(
            self.manifest,
            RuntimeComponents(),
        )
        self.assertFalse(assessment.compatible)
        self.assertEqual(
            len(
                [
                    issue
                    for issue in assessment.issues
                    if issue.code == "component_missing"
                ]
            ),
            18,
        )

    def test_hash_and_each_versioned_component_mismatch_block(self):
        mismatches = {
            "game.sts2_dll_sha256": replace(
                self.matching,
                sts2_dll_sha256="0" * 64,
            ),
            "guide.version": replace(
                self.matching,
                guide_version="0.1.0-other",
            ),
            "mod.version": replace(self.matching, mod_version="0.0.9"),
            "protocol.state_event_schema_version": replace(
                self.matching,
                protocol_schema_version=5,
            ),
            "sqlite.schema_version": replace(
                self.matching,
                sqlite_schema_version=7,
            ),
            "sqlite.snapshot_id": replace(
                self.matching,
                sqlite_snapshot_id="old-snapshot",
            ),
            "sqlite.knowledge_sha256": replace(
                self.matching,
                knowledge_sha256="0" * 64,
            ),
            "sqlite.community_scores_sha256": replace(
                self.matching,
                community_scores_sha256="0" * 64,
            ),
            "policy.bundle_version": replace(
                self.matching,
                policy_bundle_version="old-policy",
            ),
            "policy.card_reward_version": replace(
                self.matching,
                card_reward_policy_version="card_reward:old",
            ),
            "policy.route_version": replace(
                self.matching,
                route_policy_version="route:old",
            ),
            "policy.merchant_version": replace(
                self.matching,
                merchant_policy_version="merchant:old",
            ),
            "policy.campfire_version": replace(
                self.matching,
                campfire_policy_version="campfire:old",
            ),
            "policy.neow_version": replace(
                self.matching,
                neow_policy_version="neow:old",
            ),
            "policy.event_version": replace(
                self.matching,
                event_policy_version="event:old",
            ),
            "policy.deck_edit_version": replace(
                self.matching,
                deck_edit_policy_version="deck-edit:old",
            ),
        }
        for expected_component, observed in mismatches.items():
            with self.subTest(component=expected_component):
                assessment = assess_runtime_compatibility(
                    self.manifest,
                    observed,
                )
                self.assertFalse(assessment.compatible)
                mismatch_components = {
                    issue.component
                    for issue in assessment.issues
                    if issue.code == "component_mismatch"
                }
                self.assertEqual(mismatch_components, {expected_component})

    def test_host_local_assessment_excludes_game_and_mod_event_facts(self):
        local_only = replace(
            self.matching,
            game_version=None,
            sts2_dll_sha256=None,
            mod_id=None,
            mod_version=None,
        )
        assessment = assess_local_runtime_compatibility(
            self.manifest,
            local_only,
        )
        self.assertTrue(assessment.compatible)

    def test_host_observes_database_and_source_identities_independently(self):
        with tempfile.TemporaryDirectory() as tempdir:
            repository = RelationalRepository(
                str(Path(tempdir) / "runtime.db")
            )
            repository.ensure_schema()
            catalog = ROOT / "data" / "knowledge.json"
            community = ROOT / "data" / "community_scores.json"
            repository.sync_catalog(str(catalog))
            repository.sync_entity_statistics(str(community))

            observed = _observe_local_components(
                repository,
                catalog_path=catalog,
                community_path=community,
            )

        self.assertEqual(observed.guide_version, self.manifest.guide.version)
        self.assertEqual(
            observed.protocol_schema_version,
            self.manifest.protocol.state_event_schema_version,
        )
        self.assertEqual(
            observed.sqlite_snapshot_id,
            self.manifest.sqlite.snapshot_id,
        )
        self.assertTrue(
            assess_local_runtime_compatibility(
                self.manifest,
                observed,
            ).compatible
        )


class EventHandshakeTests(unittest.TestCase):
    def setUp(self):
        pending = load_compatibility_manifest(MANIFEST_PATH)
        self.manifest = _enabled_manifest(pending)

    def assess_game_version(self, version):
        return assess_event_handshake(
            self.manifest,
            EventHandshake(
                game_version=version,
                schema_version=CURRENT_SCHEMA_VERSION,
                source="sts2-guide-readonly-mod",
                producer_id="STS2GuideReadOnlyExporter",
                producer_version=self.manifest.mod.version,
                game_assembly_sha256=(
                    self.manifest.game.sts2_dll_sha256
                ),
                release_fingerprint=self.manifest.release_fingerprint,
            ),
        )

    def test_null_game_version_fails_closed(self):
        assessment = self.assess_game_version(None)
        self.assertFalse(assessment.compatible)
        self.assertEqual(assessment.reason_codes, ("component_missing",))
        self.assertEqual(assessment.issues[0].component, "event.game_version")

    def test_01080_game_version_fails_closed(self):
        assessment = self.assess_game_version("0.108.0")
        self.assertFalse(assessment.compatible)
        self.assertEqual(assessment.reason_codes, ("component_mismatch",))

    def test_01101_game_version_matches_enabled_manifest(self):
        assessment = self.assess_game_version("0.110.1")
        self.assertTrue(assessment.compatible)

    def test_unknown_game_version_fails_closed(self):
        assessment = self.assess_game_version("0.999.0")
        self.assertFalse(assessment.compatible)
        self.assertEqual(assessment.reason_codes, ("component_mismatch",))

    def test_schema_and_producer_mismatches_both_block(self):
        assessment = assess_event_handshake(
            self.manifest,
            EventHandshake(
                game_version="0.110.1",
                schema_version=6,
                source="unknown-mod",
                producer_id="other-mod",
                producer_version="0.0.9",
                game_assembly_sha256="0" * 64,
                release_fingerprint="0" * 64,
            ),
        )
        self.assertFalse(assessment.compatible)
        self.assertEqual(
            {issue.component for issue in assessment.issues},
            {
                "event.schema_version",
                "event.source",
                "event.producer_id",
                "event.producer_version",
                "event.game_assembly_sha256",
                "event.release_fingerprint",
            },
        )

    def test_checked_in_pending_status_blocks_matching_event(self):
        pending = replace(self.manifest, status="pending_validation")
        assessment = assess_event_handshake(
            pending,
            EventHandshake(
                game_version="0.110.1",
                schema_version=CURRENT_SCHEMA_VERSION,
                source="sts2-guide-readonly-mod",
                producer_id="STS2GuideReadOnlyExporter",
                producer_version=pending.mod.version,
                game_assembly_sha256=(
                    pending.game.sts2_dll_sha256
                ),
                release_fingerprint=pending.release_fingerprint,
            ),
        )
        self.assertFalse(assessment.compatible)
        self.assertEqual(assessment.reason_codes, ("manifest_not_enabled",))


class LiveCompatibilityGateTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.repository = RelationalRepository(str(self.root / "runtime.db"))
        self.repository.ensure_schema()

    def tearDown(self):
        self.tempdir.cleanup()

    def _event(self, *, game_version: str | None = "0.110.1") -> dict:
        event = json.loads(
            (ROOT / "protocol" / "state-event.example.json").read_text(
                encoding="utf-8"
            )
        )
        event["game_version"] = game_version
        return event

    def _bridge(
        self,
        manifest,
        event: dict,
        runtime_components: RuntimeComponents | None = None,
    ) -> tuple[GameStateFileBridge, Path]:
        input_path = self.root / "state-event.json"
        output_path = self.root / "advice-event.json"
        input_path.write_text(
            json.dumps(event, ensure_ascii=False),
            encoding="utf-8",
        )
        output_path.write_text(
            json.dumps(
                {
                    "run_id": "stale-run",
                    "decision_id": "stale-decision",
                }
            ),
            encoding="utf-8",
        )
        processor = RealtimeEventProcessor(
            self.repository,
            compatibility_manifest=manifest,
            runtime_components=runtime_components,
        )
        return (
            GameStateFileBridge(
                processor,
                input_path=input_path,
                output_path=output_path,
            ),
            output_path,
        )

    @staticmethod
    def _matching_runtime(manifest) -> RuntimeComponents:
        return RuntimeComponents(
            guide_version=manifest.guide.version,
            protocol_schema_version=(
                manifest.protocol.state_event_schema_version
            ),
            sqlite_schema_version=manifest.sqlite.schema_version,
            sqlite_snapshot_id=manifest.sqlite.snapshot_id,
            knowledge_sha256=manifest.sqlite.knowledge_sha256,
            community_scores_sha256=(
                manifest.sqlite.community_scores_sha256
            ),
            policy_bundle_version=manifest.policy.bundle_version,
            card_reward_policy_version=(
                manifest.policy.card_reward_version
            ),
            route_policy_version=manifest.policy.route_version,
            merchant_policy_version=manifest.policy.merchant_version,
            campfire_policy_version=manifest.policy.campfire_version,
            neow_policy_version=manifest.policy.neow_version,
            event_policy_version=manifest.policy.event_version,
            deck_edit_policy_version=manifest.policy.deck_edit_version,
        )

    def test_explicit_pending_manifest_clears_visible_advice(self):
        manifest = replace(
            load_compatibility_manifest(MANIFEST_PATH),
            status="pending_validation",
        )
        bridge, output_path = self._bridge(manifest, self._event())

        result = bridge.run_once()

        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(
            result["compatibility_issues"],
            ["manifest_not_enabled"],
        )
        self.assertFalse(output_path.exists())

    def test_enabled_manifest_rejects_old_or_missing_game_version(self):
        manifest = _enabled_manifest(
            load_compatibility_manifest(MANIFEST_PATH)
        )
        for version in ("0.108.0", None):
            with self.subTest(version=version):
                bridge, output_path = self._bridge(
                    manifest,
                    self._event(game_version=version),
                )
                result = bridge.run_once()
                self.assertEqual(result["status"], "unsupported")
                self.assertIn(
                    result["compatibility_issues"][0],
                    {
                        "component_mismatch",
                        "component_missing",
                        "invalid_production_event",
                    },
                )
                self.assertFalse(output_path.exists())

    def test_enabled_exact_v8_handshake_marks_published_advice_compatible(self):
        manifest = _enabled_manifest(
            load_compatibility_manifest(MANIFEST_PATH)
        )
        self.repository.sync_catalog(str(ROOT / "data" / "knowledge.json"))
        self.repository.sync_entity_statistics(
            str(ROOT / "data" / "community_scores.json")
        )
        runtime = self._matching_runtime(manifest)
        bridge, output_path = self._bridge(
            manifest,
            self._event(),
            runtime_components=runtime,
        )

        result = bridge.run_once()

        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["compatibility"]["status"], "compatible")
        self.assertEqual(
            result["compatibility"]["state_event_schema_version"],
            CURRENT_SCHEMA_VERSION,
        )
        published = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(published["compatibility"], result["compatibility"])

    def test_each_v8_producer_identity_gap_clears_stale_advice(self):
        manifest = _enabled_manifest(
            load_compatibility_manifest(MANIFEST_PATH)
        )
        runtime = self._matching_runtime(manifest)
        cases = {
            "old_schema": ("schema_version", 6),
            "source_mismatch": ("source", "other-source"),
            "producer_id_mismatch": ("producer_id", "OtherMod"),
            "producer_version_mismatch": ("producer_version", "0.0.9"),
            "assembly_mismatch": ("game_assembly_sha256", "0" * 64),
            "release_mismatch": ("release_fingerprint", "0" * 64),
            "producer_id_missing": ("producer_id", None),
            "assembly_missing": ("game_assembly_sha256", None),
            "release_missing": ("release_fingerprint", None),
        }
        for label, (field, value) in cases.items():
            with self.subTest(label=label):
                event = self._event()
                if value is None:
                    event.pop(field)
                else:
                    event[field] = value
                bridge, output_path = self._bridge(
                    manifest,
                    event,
                    runtime_components=runtime,
                )

                result = bridge.run_once()

                self.assertEqual(result["status"], "unsupported")
                self.assertFalse(output_path.exists())

    def test_local_source_hash_mismatch_clears_stale_advice(self):
        manifest = _enabled_manifest(
            load_compatibility_manifest(MANIFEST_PATH)
        )
        runtime = replace(
            self._matching_runtime(manifest),
            knowledge_sha256="0" * 64,
        )
        bridge, output_path = self._bridge(
            manifest,
            self._event(),
            runtime_components=runtime,
        )

        result = bridge.run_once()

        self.assertEqual(result["status"], "unsupported")
        self.assertIn("component_mismatch", result["compatibility_issues"])
        self.assertFalse(output_path.exists())

    def test_old_checkpoint_or_result_is_recomputed_not_relabelled(self):
        manifest = _enabled_manifest(
            load_compatibility_manifest(MANIFEST_PATH)
        )
        self.repository.sync_catalog(str(ROOT / "data" / "knowledge.json"))
        self.repository.sync_entity_statistics(
            str(ROOT / "data" / "community_scores.json")
        )
        runtime = self._matching_runtime(manifest)
        checkpoint_path = self.root / "active-run.json"
        checkpoint = ActiveRunCheckpointStore(checkpoint_path)
        event = GameStateEvent.model_validate(self._event())
        first = RealtimeEventProcessor(
            self.repository,
            checkpoint=checkpoint,
            compatibility_manifest=manifest,
            runtime_components=runtime,
        ).process(event)
        self.assertEqual(first["status"], "processed")

        pristine = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        for label, stale_whole_checkpoint in (
            ("stale_result_only", False),
            ("stale_checkpoint_and_result", True),
        ):
            with self.subTest(label=label):
                cached = json.loads(json.dumps(pristine))
                if stale_whole_checkpoint:
                    cached["release_fingerprint"] = "0" * 64
                for slot in ("latest_event", "current_decision"):
                    record = cached.get(slot)
                    if record is None:
                        continue
                    if stale_whole_checkpoint:
                        record["payload"]["release_fingerprint"] = "0" * 64
                        record["content_hash"] = hashlib.sha256(
                            json.dumps(
                                record["payload"],
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            ).encode("utf-8")
                        ).hexdigest()
                    record["result"]["recommendation"]["policy_version"] = (
                        "card_reward:obsolete"
                    )
                    record["result"]["compatibility"][
                        "release_fingerprint"
                    ] = "0" * 64
                checkpoint_path.write_text(
                    json.dumps(cached, ensure_ascii=False),
                    encoding="utf-8",
                )

                replay = RealtimeEventProcessor(
                    self.repository,
                    checkpoint=checkpoint,
                    compatibility_manifest=manifest,
                    runtime_components=runtime,
                ).process(event)

                self.assertFalse(replay["duplicate"])
                self.assertEqual(
                    replay["recommendation"]["policy_version"],
                    manifest.policy.card_reward_version,
                )
                self.assertEqual(
                    replay["compatibility"]["release_fingerprint"],
                    manifest.release_fingerprint,
                )
                stored = checkpoint.load()
                self.assertEqual(
                    stored["release_fingerprint"],
                    manifest.release_fingerprint,
                )
                self.assertEqual(
                    stored["current_decision"]["result"]["compatibility"][
                        "release_fingerprint"
                    ],
                    manifest.release_fingerprint,
                )


class HostStartupCompatibilityTests(unittest.TestCase):
    def test_explicit_pending_local_gate_clears_advice_before_polling(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            output = root / "advice-event.json"
            output.write_text('{"status":"stale"}', encoding="utf-8")
            pending_manifest = json.loads(
                MANIFEST_PATH.read_text(encoding="utf-8")
            )
            pending_manifest["status"] = "pending_validation"
            pending_path = root / "compatibility.json"
            pending_path.write_text(
                json.dumps(pending_manifest),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                compatibility_manifest=pending_path,
                database=root / "runtime.db",
                catalog=ROOT / "data" / "knowledge.json",
                community_scores=ROOT / "data" / "community_scores.json",
                local_card_tiers=root / "missing-local-tiers.json",
                checkpoint=root / "active-run.json",
                input=root / "state-event.json",
                output=output,
                events_dir=root / "events",
            )

            bridge = build_bridge(args)

            self.assertFalse(
                bridge.processor.local_compatibility.compatible
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
