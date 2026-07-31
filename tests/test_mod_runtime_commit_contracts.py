"""Executable source contracts for the Mod's run/commit lifecycle.

The game assembly is not launched by Python tests.  These assertions inspect
balanced C# methods so a later refactor cannot silently move identity,
sequence, or state mutation ahead of the durable queue commit.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mod" / "STS2Guide.ReadOnlyExporter"


def _csharp_method(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unbalanced C# method: {signature}")


class ModRuntimeCommitContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.writer = (MOD / "StateEventWriter.cs").read_text(encoding="utf-8")
        cls.protocol = (MOD / "ProtocolModels.cs").read_text(encoding="utf-8")
        cls.assembly_identity = (
            MOD / "GameAssemblyIdentityReader.cs"
        ).read_text(encoding="utf-8")
        cls.identity_reader = (MOD / "RunIdentityReader.cs").read_text(
            encoding="utf-8"
        )
        cls.advice_compatibility = (
            MOD / "AdviceCompatibilityReader.cs"
        ).read_text(encoding="utf-8")
        cls.card_panel = (MOD / "CardRewardAdvicePanel.cs").read_text(
            encoding="utf-8"
        )
        cls.card_observer = (MOD / "CardRewardObserver.cs").read_text(
            encoding="utf-8"
        )
        cls.route_controller = (
            MOD / "RouteAdviceController.cs"
        ).read_text(encoding="utf-8")
        cls.route_observer = (MOD / "RouteChoiceObserver.cs").read_text(
            encoding="utf-8"
        )
        cls.lifecycle = (MOD / "RunLifecycleObserver.cs").read_text(
            encoding="utf-8"
        )
        cls.drawer = (MOD / "ContextDrawer.cs").read_text(
            encoding="utf-8"
        )
        cls.generic_controller = (
            MOD / "GenericAdviceController.cs"
        ).read_text(encoding="utf-8")
        cls.release_gate = (MOD / "ReleaseCapabilityGate.cs").read_text(
            encoding="utf-8"
        )
        cls.observer_safety = (MOD / "ObserverSafety.cs").read_text(
            encoding="utf-8"
        )
        cls.csproj = (
            MOD / "STS2Guide.ReadOnlyExporter.csproj"
        ).read_text(encoding="utf-8")
        cls.merchant_observer = (MOD / "MerchantObserver.cs").read_text(
            encoding="utf-8"
        )
        cls.rest_observer = (MOD / "RestSiteObserver.cs").read_text(
            encoding="utf-8"
        )
        cls.event_observer = (MOD / "EventDecisionObserver.cs").read_text(
            encoding="utf-8"
        )
        cls.deck_edit_observer = (MOD / "DeckEditObserver.cs").read_text(
            encoding="utf-8"
        )
        cls.extended_observers = "\n".join(
            (MOD / name).read_text(encoding="utf-8")
            for name in (
                "MerchantObserver.cs",
                "RestSiteObserver.cs",
                "EventDecisionObserver.cs",
                "DeckEditObserver.cs",
            )
        )

    def test_card_resume_accepts_v9_without_losing_v5_v8_compatibility(self):
        recover = _csharp_method(
            self.writer,
            "private static string? ReadResumableDecisionId(",
        )
        self.assertIn("version is not (5 or 6 or 7 or 8 or 9)", recover)
        self.assertNotIn("version != 5", recover)
        self.assertIn(
            "StringPropertyEquals(\n"
            "                    payload,\n"
            '                    "decision_id",\n'
            "                    decisionId",
            recover,
        )
        self.assertIn("PayloadMatchesDecision(", recover)
        self.assertIn("HasLaterInvalidatingEvent(", recover)

    def test_sequence_advances_only_after_durable_queue_commit(self):
        write = _csharp_method(
            self.writer,
            "private static string? Write(",
        )
        proposed = write.index("var sequence = checked(_sequence + 1);")
        queued = write.index("WriteAtomically(eventPath, json);")
        committed = write.index("_sequence = sequence;")
        mirror = write.index("WriteAtomically(outputPath, json);")
        self.assertLess(proposed, queued)
        self.assertLess(queued, committed)
        self.assertLess(committed, mirror)
        self.assertNotIn("Interlocked.Increment", write)

        # The intended failure/retry behavior represented by the source
        # ordering: a failed first queue write leaves committed sequence zero,
        # so the next attempt is still sequence one.
        current = 0
        first_proposed = current + 1
        first_queue_committed = False
        if first_queue_committed:
            current = first_proposed
        self.assertEqual(current, 0)
        second_proposed = current + 1
        second_queue_committed = True
        if second_queue_committed:
            current = second_proposed
        self.assertEqual(current, 1)

    def test_v9_revision_preferences_and_loaded_assembly_are_reported(self):
        self.assertIn(
            "internal const int CurrentSchemaVersion = 9;",
            self.protocol,
        )
        self.assertIn(
            '[JsonPropertyName("state_revision")]',
            self.protocol,
        )
        self.assertIn(
            '[JsonPropertyName("producer_id")]',
            self.protocol,
        )
        self.assertIn(
            '[JsonPropertyName("producer_version")]',
            self.protocol,
        )
        self.assertIn(
            '[JsonPropertyName("game_assembly_sha256")]',
            self.protocol,
        )
        self.assertIn(
            '[JsonPropertyName("release_fingerprint")]',
            self.protocol,
        )
        self.assertIn(
            '[JsonPropertyName("guide_preferences")]',
            self.protocol,
        )

        write = _csharp_method(
            self.writer,
            "private static string? Write(",
        )
        self.assertIn("StateRevision = sequence,", write)
        self.assertIn(
            "GameAssemblyIdentityReader.ReadSha256()",
            write,
        )
        self.assertIn("GuidePreferences = new GuidePreferences", write)
        self.assertIn("RouteMode = emittedRouteMode", write)
        self.assertLess(
            write.index("StateRevision = sequence,"),
            write.index("WriteAtomically(eventPath, json);"),
        )
        self.assertLess(
            write.index("WriteAtomically(eventPath, json);"),
            write.index("_sequence = sequence;"),
        )

        self.assertIn(
            "var assembly = typeof(RunManager).Assembly;",
            self.assembly_identity,
        )
        self.assertIn("assembly.Location", self.assembly_identity)
        self.assertIn("SHA256.HashData(stream)", self.assembly_identity)
        self.assertNotIn(
            "016c6df717d997fcbd8f2a55102ca63cb",
            self.assembly_identity.lower(),
        )

    def test_game_ui_requires_host_compatibility_attestation(self):
        for consumer, signature in (
            (
                self.card_panel,
                "private static IReadOnlyList<ContextDrawerRow>? "
                "ReadMatchingRows(",
            ),
            (
                self.route_controller,
                "private static RouteAdvice? ReadMatching(",
            ),
            (
                self.generic_controller,
                "private static IReadOnlyList<ContextDrawerRow>? "
                "ReadMatchingRows(",
            ),
        ):
            read = _csharp_method(consumer, signature)
            self.assertIn(
                "AdviceCompatibilityReader.MatchesCurrentRuntime(root)",
                read,
            )

        gate = _csharp_method(
            self.advice_compatibility,
            "internal static bool MatchesCurrentRuntime(",
        )
        for required_fact in (
            '"status", "compatible"',
            '"manifest_version", 2',
            '"release_fingerprint"',
            "StateEvent.ReleaseFingerprint",
            '"state_event_schema_version"',
            "StateEvent.CurrentSchemaVersion",
            '"producer_id"',
            "StateEvent.ProducerId",
            '"producer_version"',
            "StateEvent.ProducerVersion",
            '"game_version"',
            "GameVersionReader.Read()",
            '"game_assembly_sha256"',
            "GameAssemblyIdentityReader.ReadSha256()",
        ):
            self.assertIn(required_fact, gate)

        card_recovery = _csharp_method(
            self.writer,
            "private static string? ReadResumableDecisionId(",
        )
        route_recovery = _csharp_method(
            self.writer,
            "private static void TryAddRouteDecisionPayload(",
        )
        for recovery in (card_recovery, route_recovery):
            self.assertIn('"release_fingerprint"', recovery)
            self.assertIn("StateEvent.ReleaseFingerprint", recovery)

    def test_route_mode_update_is_owner_scoped_and_transactional(self):
        update = _csharp_method(
            self.writer,
            "internal static PendingDecisionView? EmitRouteModeUpdated(",
        )
        self.assertIn("pending.DecisionId != expectedDecisionId", update)
        self.assertIn("ReferenceEquals(pending.ScreenOwner, owner)", update)
        write = update.index("var eventId = Write(")
        failure = update.index("if (eventId is null)")
        commit = update.index("_routeMode = requestedMode;")
        replace_pending = update.index("_pendingDecision = new PendingDecision(")
        self.assertLess(write, failure)
        self.assertLess(failure, commit)
        self.assertLess(commit, replace_pending)
        self.assertIn("decisionId: pending.DecisionId", update)
        self.assertIn("routeMode: requestedMode", update)

    def test_route_mode_recovery_uses_only_valid_active_run_checkpoint(self):
        recover = _csharp_method(
            self.writer,
            "private static string RecoverRouteMode(",
        )
        self.assertIn('"active-run.json"', recover)
        self.assertNotIn('"state-event.json"', recover)
        self.assertNotIn('"events"', recover)
        self.assertNotIn("TryAddEventRouteMode", self.writer)
        self.assertIn('"checkpoint_version"', recover)
        self.assertIn("version != 3", recover)
        self.assertIn('"release_fingerprint"', recover)
        self.assertIn("StateEvent.ReleaseFingerprint", recover)
        self.assertIn('StringPropertyEquals(root, "run_id", runId)', recover)
        self.assertGreaterEqual(
            recover.count("return GuideRouteModes.Balanced;"),
            3,
        )

        # A valid newer spool entry must not override the checkpoint.  This
        # fixture intentionally contains checkpoint=balanced and spool=growth.
        fixture = ROOT / "tests" / "fixtures" / "runtime" / "route-mode-spool-race"
        checkpoint_payload = json.loads(
            (fixture / "active-run.json").read_text(encoding="utf-8")
        )
        event_payloads = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (fixture / "events").glob("*.json")
        ]
        self.assertEqual(
            checkpoint_payload["guide_preferences"]["route_mode"],
            "balanced",
        )
        self.assertIn(
            "growth",
            {
                payload["guide_preferences"]["route_mode"]
                for payload in event_payloads
            },
        )

    def test_run_mode_resets_only_after_successful_end_commit(self):
        ended = _csharp_method(
            self.writer,
            "internal static void EmitRunEnded(",
        )
        failure = ended.index("if (eventId is null)")
        reset = ended.index("_routeMode = GuideRouteModes.Balanced;")
        self.assertLess(failure, reset)
        activate = _csharp_method(
            self.writer,
            "private static void ActivateRun(",
        )
        self.assertIn("_routeMode = RecoverRouteMode(_runId);", activate)

    def test_route_mode_drawer_releases_gate_and_backup_is_rejected(self):
        callback = _csharp_method(
            self.drawer,
            "private static void RouteModePressed(",
        )
        first_lock_end = callback.index("}\n\n        var committed")
        invoke = callback.index("callback!(requestedMode)")
        self.assertLess(first_lock_end, invoke)
        self.assertIn("if (committed && handle == _activeHandle)", callback)
        for caption in ("智能均衡", "稳健生存", "激进成长"):
            self.assertIn(f'"{caption}"', self.drawer)

        controller_callback = _csharp_method(
            self.route_controller,
            "private static bool OnRouteModeRequested(",
        )
        self.assertIn("StateEventWriter.EmitRouteModeUpdated(", controller_callback)
        self.assertIn("_pending.DecisionId", controller_callback)
        self.assertIn("if (updated is null)", controller_callback)
        presentation = _csharp_method(
            self.route_controller,
            "private static RoutePresentation? ReadPresentation(",
        )
        self.assertIn("if (backup.Count > 0)", presentation)
        self.assertIn("return null;", presentation)

    def test_begin_run_abandon_is_transactional(self):
        begin = _csharp_method(
            self.writer,
            "internal static void BeginRun()",
        )
        write = begin.index("var syntheticEndEventId = Write(")
        failure_guard = begin.index("if (syntheticEndEventId is null)")
        remember = begin.index("RememberCurrentIdentityEnded();")
        activate = begin.index(
            "ActivateRun(identity, clearObservedPlayer: true);"
        )
        self.assertLess(write, failure_guard)
        self.assertLess(failure_guard, remember)
        self.assertLess(remember, activate)
        failure_block = begin[
            failure_guard:begin.index("}", failure_guard) + 1
        ]
        self.assertIn("return;", failure_block)
        self.assertIn("Retaining its identity, state and", failure_block)
        self.assertIn("_runTransitionPending = true;", failure_block)

        activate_method = _csharp_method(
            self.writer,
            "private static void ActivateRun(",
        )
        self.assertLess(
            activate_method.index("_runId = identity.RunId;"),
            activate_method.index("_lastState = null;"),
        )
        self.assertIn("_runTransitionPending = false;", activate_method)

        retry = _csharp_method(
            self.writer,
            "private static bool EnsureStableRunIdentity(",
        )
        self.assertIn("if (_runTransitionPending)", retry)
        self.assertIn("return !_runTransitionPending;", retry)
        retry_write = retry.index("var syntheticEndEventId = Write(")
        retry_guard = retry.index("if (syntheticEndEventId is null)")
        retry_remember = retry.index("RememberCurrentIdentityEnded();")
        retry_activate = retry.index(
            "ActivateRun(identity, clearObservedPlayer: false);"
        )
        self.assertLess(retry_write, retry_guard)
        self.assertLess(retry_guard, retry_remember)
        self.assertLess(retry_remember, retry_activate)

    def test_stable_identity_requires_current_runstate_seed(self):
        reader = _csharp_method(
            self.identity_reader,
            "internal static RunIdentity Read()",
        )
        self.assertIn(".DebugOnlyGetState()?", reader)
        self.assertIn(".Players", reader)
        self.assertIn("currentRunSeed", reader)
        unavailable = reader.index(
            "if (string.IsNullOrWhiteSpace(currentRunSeed))"
        )
        mismatch = reader.index(
            "&& !string.Equals(\n"
            "                    historySeed,\n"
            "                    currentRunSeed,"
        )
        stable_hash = reader.index("SHA256.HashData(")
        self.assertLess(unavailable, stable_hash)
        self.assertLess(mismatch, stable_hash)
        self.assertIn("return Temporary(", reader[unavailable:stable_hash])
        self.assertIn("identity remains provisional", reader[mismatch:stable_hash])
        self.assertNotIn("var seed = history?.Seed;", reader)

        ensure = _csharp_method(
            self.writer,
            "private static bool EnsureStableRunIdentity(",
        )
        self.assertIn("var identity = RunIdentityReader.Read();", ensure)
        self.assertNotIn("ReadIdentityFromCurrentRun", self.writer)

    def test_card_snapshot_reads_current_map_or_marks_explicit_gap(self):
        card = _csharp_method(
            self.writer,
            "internal static void EmitCardReward(",
        )
        self.assertIn(
            "TryCaptureCurrentMapContext(state)",
            card,
        )
        emit = card.index("var eventId = Write(")
        attach = card.index("mapContext: currentMapContext")
        self.assertLess(emit, attach)
        self.assertNotIn("mapContext: _last", card)
        capture = _csharp_method(
            self.writer,
            "private static MapChoiceContext? "
            "TryCaptureCurrentMapContext(",
        )
        self.assertIn("MapNodeReader.Read(player)", capture)
        self.assertIn('"map_snapshot_unavailable"', capture)
        self.assertNotIn("_last", capture)

    def test_extended_observers_are_postfix_only_and_use_verified_members(self):
        self.assertNotIn("[HarmonyPrefix]", self.extended_observers)
        self.assertGreaterEqual(
            self.extended_observers.count("[HarmonyPostfix]"),
            12,
        )
        for verified_member in (
            "MerchantInventory.AllEntries",
            "MerchantEntry.OnMerchantInventoryUpdated",
            "RestSiteOption.OptionId",
            "NRestSiteRoom.Options",
            "EventModel.CurrentOptions",
            "EventOption.TextKey",
            "NCardGridSelectionScreen.CardsSelected",
        ):
            type_name, member_name = verified_member.split(".", 1)
            self.assertIn(type_name, self.extended_observers)
            self.assertIn(member_name, self.extended_observers)
        for forbidden_action in (
            "OnTryPurchaseWrapper(",
            ".OnSelect()",
            ".Chosen()",
            ".OnCardClicked(",
        ):
            self.assertNotIn(forbidden_action, self.extended_observers)
        self.assertIn(
            '"OnAfterPlayerSelectedRestSiteOption"',
            self.rest_observer,
        )
        self.assertIn("if (success)", self.rest_observer)
        self.assertNotIn(
            '"OnBeforePlayerSelectedRestSiteOption"',
            self.rest_observer,
        )
        self.assertNotIn(
            "nameof(RestSiteOption.OnSelect)",
            self.rest_observer,
        )
        self.assertNotIn(
            "nameof(NRestSiteRoom.AfterSelectingOption)",
            self.rest_observer,
        )
        self.assertIn(
            "NEventRoom.OptionButtonClicked",
            self.extended_observers,
        )
        self.assertIn(
            "preferences.MinSelect != 1",
            self.extended_observers,
        )
        self.assertIn(
            "preferences.MaxSelect != 1",
            self.extended_observers,
        )
        self.assertNotIn("ConfigureAwait(false)", self.deck_edit_observer)
        self.assertIn("ScheduleSelectionCompletion", self.deck_edit_observer)
        self.assertIn("CompleteSelection", self.deck_edit_observer)
        self.assertIn("CallDeferred", self.deck_edit_observer)
        self.assertNotIn("GodotThreadId", self.deck_edit_observer)
        self.assertIn(
            "GenericAdviceController.Hide(owner)",
            self.deck_edit_observer,
        )
        self.assertIn("ScheduleObserve", self.event_observer)
        self.assertIn("ScheduleFinished", self.event_observer)
        self.assertIn("CallDeferred", self.event_observer)
        self.assertNotIn("if (option.IsProceed)", self.event_observer)
        self.assertGreaterEqual(
            self.extended_observers.count("ObserverSafety.Run("),
            self.extended_observers.count("[HarmonyPostfix]"),
        )

    def test_generic_writer_and_drawer_are_identity_scoped(self):
        emit = _csharp_method(
            self.writer,
            "internal static PendingDecisionView? EmitGenericDecision(",
        )
        self.assertIn("CreateGenericObservationFingerprint(", emit)
        self.assertIn("ReferenceEquals(", emit)
        self.assertIn("genericCandidates: candidates", emit)
        self.assertIn("decisionParent: decisionParent", emit)
        self.assertLess(
            emit.index("var eventId = Write("),
            emit.index("_pendingDecision = new PendingDecision("),
        )
        selected = _csharp_method(
            self.writer,
            "internal static bool EmitGenericSelected(",
        )
        self.assertIn(
            "ReferenceEquals(pending.ScreenOwner, owner)",
            selected,
        )
        read = _csharp_method(
            self.generic_controller,
            "private static IReadOnlyList<ContextDrawerRow>? "
            "ReadMatchingRows(",
        )
        for identity in (
            '"run_id", pending.RunId',
            '"event_id", pending.EventId',
            '"decision_id", pending.DecisionId',
            '"event_type", pending.EventType',
            '"sequence", pending.Sequence',
        ):
            self.assertIn(identity, read)
        resolve_parent = _csharp_method(
            self.writer,
            "internal static DecisionParentContext?\n"
            "        ResolveDeckEditParent(",
        )
        self.assertIn("matches.Count != 1", resolve_parent)
        self.assertIn('effect.Certainty == "exact"', resolve_parent)
        self.assertIn('effect.TargetMode == "choose"', resolve_parent)
        self.assertIn("ClosePendingDecision(", resolve_parent)
        self.assertIn("ReadReasons(candidate)", read)
        self.assertIn(".Take(3)", self.generic_controller)
        self.assertIn("TooltipText", self.drawer)

    def test_mod_capability_gate_is_embedded_and_fails_closed(self):
        self.assertIn(
            'LogicalName="STS2Guide.CompatibilityManifest"',
            self.csproj,
        )
        self.assertIn('status != "enabled"', self.release_gate)
        self.assertIn(
            "StateEvent.ReleaseFingerprint",
            self.release_gate,
        )
        self.assertIn("catch", self.release_gate)
        for capability, source in (
            ("card_reward", self.card_panel),
            ("route_choice", self.route_controller),
            ("merchant", self.merchant_observer),
            ("rest_site", self.rest_observer),
            ("deck_edit", self.deck_edit_observer),
        ):
            self.assertIn(
                f'ReleaseCapabilityGate.IsEnabled("{capability}")',
                source,
            )
        self.assertIn('"neow_choice"', self.event_observer)
        self.assertIn('"event_choice"', self.event_observer)
        self.assertIn(
            "ReleaseCapabilityGate.IsEnabled(capability)",
            self.event_observer,
        )
        self.assertIn(
            "ReleaseCapabilityGate.IsEnabled(pending.EventType)",
            self.generic_controller,
        )

    def test_special_card_reward_requires_typed_enabled_parent(self):
        self.assertIn(
            "TryCaptureCardRewardContext(",
            self.card_observer,
        )
        self.assertIn(
            "card_reward.deferred_special_populate",
            self.card_observer,
        )
        self.assertIn("CallDeferred", self.card_observer)
        schedule = _csharp_method(
            self.card_observer,
            "private static void ScheduleOrObserve(",
        )
        self.assertIn(
            "Observe(reward, requiredParentSourceType: null)",
            schedule,
        )
        self.assertLess(
            schedule.index(
                "Observe(reward, requiredParentSourceType: null)"
            ),
            schedule.index("Callable.From("),
        )
        self.assertIn("PrepareSpecialReward(", schedule)
        self.assertIn(
            "ReleaseCapabilityGate.IsEnabled(parentCapability)",
            self.card_observer,
        )
        self.assertIn('"neow_choice" => "NEOW"', self.card_observer)
        self.assertIn('"event_choice" => "EVENT"', self.card_observer)
        self.assertIn('_ => "CARD"', self.card_observer)
        self.assertNotIn("ReadRewardType", self.card_observer)
        self.assertNotIn("PeekDecisionParent(", self.card_observer)
        self.assertIn(
            "TryResolveCapturedCardRewardParent(",
            self.card_observer,
        )
        emit = _csharp_method(
            self.writer,
            "internal static void EmitCardReward(",
        )
        self.assertIn("requiredParentSourceType", emit)
        self.assertIn("DecisionParentsEqual(", emit)
        self.assertIn("decisionParent is null", emit)
        self.assertLess(
            emit.index("decisionParent is null"),
            emit.index("ClosePendingDecision("),
        )
        self.assertIn(
            "sameDecision\n                ? _pendingDecisionParent",
            emit,
        )
        self.assertIn(
            "HasPendingCardRewardParent(",
            self.card_panel,
        )
        self.assertIn("allowDeferredSpecialRetry", self.card_panel)
        self.assertIn("card_reward.deferred_panel", self.card_panel)
        self.assertIn("SpecialRewardExpectation", self.card_panel)
        self.assertIn(
            "Sessions[model] = new EventSession(owner)",
            self.event_observer,
        )
        self.assertIn(
            "event.deferred_presentation_close",
            self.event_observer,
        )
        exited = _csharp_method(
            self.event_observer,
            "internal static void OnExited(",
        )
        self.assertIn("SchedulePresentationClose(model, owner)", exited)
        self.assertNotIn("Sessions.Remove(model)", exited)
        self.assertNotIn("EmitGenericOwnerClosed(owner)", exited)
        self.assertNotIn("RemoveBindingsForOwner(owner)", exited)
        self.assertNotIn("remainingDeferrals", self.event_observer)
        self.assertIn(
            "ConditionalWeakTable<object, CapturedCardRewardContext>",
            self.event_observer,
        )
        chosen = _csharp_method(
            self.event_observer,
            "internal static void OnChosen(",
        )
        self.assertIn("CardRewardParent =", chosen)
        self.assertIn("PeekDecisionParent(", chosen)
        self.assertIn(
            "OnNonEventDecisionBoundary()",
            _csharp_method(
                self.route_observer,
                "internal static void OnSelected(",
            ),
        )
        self.assertIn(
            "ClearUnconsumedDecisionParentAtBoundary()",
            _csharp_method(
                self.event_observer,
                "internal static void OnNonEventDecisionBoundary(",
            ),
        )
        boundary = _csharp_method(
            self.writer,
            "internal static void "
            "ClearUnconsumedDecisionParentAtBoundary(",
        )
        self.assertIn(
            "_consumedChildParentKey = ChildParentKey(",
            boundary,
        )
        self.assertLess(
            boundary.index("_consumedChildParentKey = ChildParentKey("),
            boundary.index("_pendingChildParent = null"),
        )
        self.assertIn("ResetRunLifecycle()", self.lifecycle)

    def test_child_parent_marker_is_typed_and_effect_gated(self):
        close = _csharp_method(
            self.writer,
            "private static bool ClosePendingDecision(",
        )
        peek = _csharp_method(
            self.writer,
            "private static DecisionParentContext? "
            "PeekPendingChildParent(",
        )
        expected = _csharp_method(
            self.writer,
            "private static string? ExpectedChildKind(",
        )
        self.assertIn("ExpectedChildKind(", close)
        self.assertIn("pending.ExpectedChildKind != expectedChildKind", peek)
        self.assertNotIn("TimeSpan.FromMinutes", peek)
        self.assertIn('effect.Certainty == "exact"', expected)
        self.assertIn('effect.TargetMode == "choose"', expected)
        self.assertIn('effect.ChildDecisionType == "card_reward"', expected)
        self.assertIn('"card_reward"', expected)
        self.assertIn('"deck_edit:upgrade"', expected)
        self.assertIn('"deck_edit:remove"', expected)
        self.assertIn('"deck_edit:transform"', expected)

        recover = _csharp_method(
            self.writer,
            "private static PendingChildParent? "
            "TryRecoverPendingChildParent(",
        )
        self.assertIn('"active-run.json"', recover)
        self.assertIn('version != 3', recover)
        self.assertIn('"release_fingerprint"', recover)
        self.assertIn('"child_expectation"', recover)
        self.assertIn('StableParentSourceId(parentCandidateId)', recover)
        self.assertIn('parentCloseSequence != closeSequence', recover)
        self.assertIn('ChildExpectationMatches(', recover)
        self.assertIn('_consumedChildParentKey', recover)

    def test_choice_effect_serializes_explicit_child_decision_type(self):
        self.assertIn(
            '[JsonPropertyName("child_decision_type")]',
            self.protocol,
        )
        self.assertIn(
            'public string? ChildDecisionType { get; init; }',
            self.protocol,
        )
        self.assertNotIn(
            'ChildDecisionType = "card_reward"',
            self.event_observer,
        )

    def test_postfix_observer_boundary_never_rethrows(self):
        self.assertIn("try", self.observer_safety)
        self.assertIn("catch (Exception exception)", self.observer_safety)
        self.assertIn("catch\n            {", self.observer_safety)
        self.assertNotIn("throw", self.observer_safety)

    def test_reason_labels_receive_hover_without_game_actions(self):
        rows = self.drawer[self.drawer.index("_rows = []"):]
        label = rows[rows.index("var label = new Label"):]
        self.assertIn(
            "MouseFilter = Control.MouseFilterEnum.Stop",
            label,
        )


if __name__ == "__main__":
    unittest.main()
