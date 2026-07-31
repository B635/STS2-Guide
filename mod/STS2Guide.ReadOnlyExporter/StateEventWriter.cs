using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace STS2Guide.ReadOnlyExporter;

internal static class StateEventWriter
{
    private static readonly object WriteGate = new();
    private static string _runId = $"temporary-{Guid.NewGuid():N}";
    private static string _runStartedAt =
        DateTimeOffset.UtcNow.ToString("O");
    private static bool _runIdentityStable;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
    };
    private static long _sequence;
    private static RunStateSnapshot? _lastState;
    private static PendingDecision? _pendingDecision;
    private static DecisionParentContext? _pendingDecisionParent;
    private static bool _ended;
    private static string? _lastEndedStableRunId;
    private static bool _resumeDecisionRecoveryAvailable;
    private static bool _runIdentityLockedByEmission;
    private static bool _runTransitionPending;
    private static string _routeMode = GuideRouteModes.Balanced;
    private static PendingChildParent? _pendingChildParent;
    private static string? _consumedChildParentKey;

    internal static PendingDecisionView? GetPendingDecision()
    {
        lock (WriteGate)
        {
            return _pendingDecision is null
                ? null
                : ToPendingDecisionView(_pendingDecision);
        }
    }

    private static PendingDecisionView ToPendingDecisionView(
        PendingDecision pending
    )
        => new(
            _runId,
            pending.ParentEventId,
            pending.DecisionId,
            pending.Sequence,
            pending.RouteMode,
            pending.CanSkip,
            pending.EventType,
            pending.Options.Select(option => new PendingDecisionOptionView(
                option.CandidateId ?? "",
                option.Card,
                option.Upgrades,
                option.Enchantment,
                option.EnchantmentAmount,
                option.Affliction,
                option.AfflictionAmount
            )).ToList(),
            pending.Candidates.Select(candidate => new PendingDecisionCandidateView(
                candidate.CandidateId,
                candidate.Label,
                candidate.DisplayIndex,
                candidate.Eligible
            )).ToList(),
            pending.RouteContext
        );

    internal static bool BindCardRewardScreen(
        NCardRewardSelectionScreen owner,
        string expectedDecisionId)
        => BindDecisionScreen(owner, expectedDecisionId);

    internal static bool BindDecisionScreen(
        object owner,
        string expectedDecisionId)
    {
        lock (WriteGate)
        {
            if (_pendingDecision is null
                || _pendingDecision.DecisionId != expectedDecisionId)
            {
                return false;
            }
            _pendingDecision.ScreenOwner = owner;
            return true;
        }
    }

    internal static void UnbindCardRewardScreen(
        NCardRewardSelectionScreen owner)
        => UnbindDecisionScreen(owner);

    internal static void UnbindDecisionScreen(object owner)
    {
        lock (WriteGate)
        {
            if (_pendingDecision is not null
                && ReferenceEquals(_pendingDecision.ScreenOwner, owner))
            {
                _pendingDecision.ScreenOwner = null;
            }
        }
    }

    internal static void BeginRun()
    {
        lock (WriteGate)
        {
            var identity = RunIdentityReader.Read();
            var previousState = _lastState;
            var hasUnendedState = previousState is not null && !_ended;

            // If the previous run was still in progress (never ended),
            // only a confirmed different stable identity proves a new run.
            // An unknown/temporary launch is normal during save+continue.
            if (RunIdentityGuard.ShouldCloseActiveRun(
                hasUnendedState,
                _runId,
                _runIdentityStable,
                identity.RunId,
                identity.IsStable
            ))
            {
                var syntheticEndEventId = Write(
                    "run_ended",
                    previousState!,
                    new List<DecisionOption>(),
                    runResult: CreateRunResult("abandon")
                );
                if (syntheticEndEventId is null)
                {
                    _runTransitionPending = true;
                    Log.Error(
                        "[STS2-Guide] New run activation deferred: the "
                        + "previous run's synthetic abandon event was not "
                        + "committed. Retaining its identity, state and "
                        + "pending decision for retry."
                    );
                    return;
                }
                RememberCurrentIdentityEnded();
            }

            if (RunIdentityGuard.ShouldRetainCurrentIdentity(
                hasUnendedState,
                _runIdentityStable,
                identity.IsStable
            ))
            {
                _runTransitionPending = true;
                Log.Info(
                    "[STS2-Guide] Run launch identity remains provisional "
                    + "until the current RunState seed is available; "
                    + "retaining the previous identity and state."
                );
                return;
            }

            // Guard: if the new identity matches a previous stable Run ID
            // and the previous run ended, the game's History.Seed is stale.
            // Force a new temporary ID to prevent cross-run contamination.
            if (identity.IsStable
                && !RunIdentityGuard.TryAcceptStableIdentity(
                    ref _lastEndedStableRunId,
                    identity.RunId,
                    identity.IsStable
                ))
            {
                Log.Error(
                    $"[STS2-Guide] Stale History.Seed produced same Run ID "
                    + $"{identity.RunId} as previous ended run. "
                    + "Generating new temporary identity."
                );
                identity = new RunIdentity(
                    $"temporary-{Guid.NewGuid():N}",
                    DateTimeOffset.UtcNow.ToString("O"),
                    false
                );
            }

            ActivateRun(identity, clearObservedPlayer: true);
        }
    }

    internal static void EmitRunEnded(string outcome)
    {
        lock (WriteGate)
        {
            if (_ended)
            {
                return;
            }
            RunStateSnapshot? state = null;
            if (!RunStateReader.TryCapture(out state) || state is null)
            {
                state = _lastState;
            }
            if (state is null)
            {
                Log.Error(
                    "[STS2-Guide] Run end not committed: no state is "
                    + "available; a later lifecycle callback may retry."
                );
                return;
            }
            if (!EnsureStableRunIdentity(state))
            {
                return;
            }
            var eventId = Write(
                "run_ended",
                state,
                new List<DecisionOption>(),
                runResult: CreateRunResult(outcome)
            );
            if (eventId is null)
            {
                Log.Error(
                    "[STS2-Guide] Run end not committed because the event "
                    + "write failed; retaining state for retry."
                );
                return;
            }
            _ended = true;
            // Preserve the most recently ended stable identity until a
            // genuinely different stable identity is accepted.
            RunIdentityGuard.RememberEndedStableIdentity(
                ref _lastEndedStableRunId,
                _runId,
                _runIdentityStable
            );
            _lastState = null;
            _pendingDecision = null;
            _pendingDecisionParent = null;
            _pendingChildParent = null;
            _resumeDecisionRecoveryAvailable = false;
            _routeMode = GuideRouteModes.Balanced;
            RunStateReader.Clear();
        }
    }

    internal static void EmitCardReward(
        List<DecisionOption> options,
        DecisionContext decision,
        object decisionSource,
        string? requiredParentSourceType = null,
        DecisionParentContext? requiredParent = null
    )
    {
        if (!ReleaseCapabilityGate.IsEnabled("card_reward"))
        {
            return;
        }
        if (!RunStateReader.TryCapture(out var state) || state is null)
        {
            return;
        }
        lock (WriteGate)
        {
            if (!EnsureStableRunIdentity(state))
            {
                return;
            }
            var currentMapContext = TryCaptureCurrentMapContext(state);
            var sameDecision = _pendingDecision is not null
                && ReferenceEquals(
                    _pendingDecision.DecisionSource,
                    decisionSource
                );
            // Validate lineage before closing or otherwise mutating the
            // currently pending decision. The first observation consumes a
            // typed parent marker; a repeated Populate of the same reward
            // object reuses the already-bound parent instead.
            var decisionParent = sameDecision
                ? _pendingDecisionParent
                : requiredParentSourceType is null
                    ? null
                    : PeekPendingChildParent(
                        "card_reward",
                        "neow_choice",
                        "event_choice"
                    );
            if (requiredParentSourceType is not null
                && (decisionParent is null
                    || decisionParent.SourceType
                        != requiredParentSourceType
                    || requiredParent is null
                    || !DecisionParentsEqual(
                        decisionParent,
                        requiredParent
                    )))
            {
                return;
            }
            if (_pendingDecision is not null && !sameDecision)
            {
                if (!ClosePendingDecision(new DecisionOutcome
                {
                    Kind = "closed_unknown",
                }))
                {
                    return;
                }
            }
            _lastState = state;
            for (var index = 0; index < options.Count; index++)
            {
                options[index].CandidateId = CandidateId(index, options[index]);
            }
            var recoveredDecisionId = sameDecision
                ? null
                : TryRecoverDecisionId(options, decision, state);
            var decisionId = sameDecision
                ? _pendingDecision!.DecisionId
                : recoveredDecisionId
                    ?? $"{_runId}:card-reward:{Guid.NewGuid():N}";
            var eventId = Write(
                "card_reward",
                state,
                options,
                decision: decision,
                mapContext: currentMapContext,
                decisionId: decisionId,
                decisionParent: decisionParent
            );
            if (eventId is not null)
            {
                if (!sameDecision && decisionParent is not null)
                {
                    ConsumePendingChildParent(decisionParent);
                }
                _pendingDecisionParent = decisionParent;
                _resumeDecisionRecoveryAvailable = false;
                var screenOwner = sameDecision
                    ? _pendingDecision?.ScreenOwner
                    : null;
                _pendingDecision = new PendingDecision(
                    eventId,
                    decisionId,
                    _sequence,
                    _routeMode,
                    decision.CanSkip,
                    options,
                    [],
                    "card_reward",
                    decisionSource,
                    null,
                    null,
                    null
                )
                {
                    ScreenOwner = screenOwner,
                };
            }
        }
    }

    internal static void EmitCardSelected(
        CardModel card,
        NCardRewardSelectionScreen owner)
    {
        var selected = RunStateReader.ReadDecisionOption(card);
        lock (WriteGate)
        {
            var pending = _pendingDecision;
            var state = _lastState;
            if (pending is null || pending.EventType != "card_reward" || state is null)
            {
                return;
            }
            if (!ReferenceEquals(pending.ScreenOwner, owner))
            {
                Log.Error(
                    "[STS2-Guide] Ignoring selection from a stale card "
                    + "screen owner."
                );
                return;
            }
            var exactMatches = pending.Options
                .Select((option, index) => (option, index))
                .Where(item => SameOption(item.option, selected))
                .Select(item => item.index)
                .ToList();
            var optionIndex = exactMatches.Count == 1
                ? exactMatches[0]
                : -1;
            if (optionIndex < 0)
            {
                var cardMatches = pending.Options
                    .Select((option, index) => (option, index))
                    .Where(item => item.option.Card == selected.Card)
                    .Select(item => item.index)
                    .ToList();
                optionIndex = cardMatches.Count == 1
                    ? cardMatches[0]
                    : -1;
            }
            if (optionIndex < 0)
            {
                Log.Error(
                    "[STS2-Guide] Selected card does not uniquely match "
                    + "the active decision; closing without inventing a "
                    + "candidate identity."
                );
                ClosePendingDecision(
                    new DecisionOutcome { Kind = "closed_unknown" },
                    expectedDecisionId: pending.DecisionId
                );
                return;
            }
            var eventId = Write(
                "decision_closed",
                state,
                new List<DecisionOption>(),
                parentEventId: pending.ParentEventId,
                outcome: new DecisionOutcome
                {
                    Kind = "selected",
                    SelectedCard = selected.Card,
                    SelectedCandidateId = pending.Options[optionIndex]
                        .CandidateId,
                    SelectedOptionIndex = optionIndex,
                },
                decisionId: pending.DecisionId
            );
            if (eventId is not null)
            {
                _pendingDecision = null;
                _pendingDecisionParent = null;
            }
        }
    }

    internal static bool EmitCardSkipped(object expectedDecisionSource)
    {
        lock (WriteGate)
        {
            if (_pendingDecision is null
                || _pendingDecision.EventType != "card_reward")
            {
                return false;
            }
            return ClosePendingDecision(new DecisionOutcome
            {
                Kind = "skipped",
            }, expectedDecisionSource: expectedDecisionSource);
        }
    }

    internal static bool EmitCardSkippedFromScreen(
        NCardRewardSelectionScreen owner)
    {
        lock (WriteGate)
        {
            var pending = _pendingDecision;
            if (pending is null || pending.EventType != "card_reward")
            {
                return false;
            }
            if (!ReferenceEquals(pending.ScreenOwner, owner))
            {
                Log.Error(
                    "[STS2-Guide] Ignoring skip from a stale card "
                    + "screen owner."
                );
                return false;
            }
            return ClosePendingDecision(
                new DecisionOutcome { Kind = "skipped" },
                expectedDecisionId: pending.DecisionId
            );
        }
    }

    internal static void EmitMapChoice(
        RunStateSnapshot state,
        MapSnapshot snapshot
    )
    {
        lock (WriteGate)
        {
            if (!EnsureStableRunIdentity(state))
            {
                return;
            }
            _lastState = state;
            Write(
                "map_choice",
                state,
                new List<DecisionOption>(),
                mapContext: new MapChoiceContext
                {
                    Nodes = snapshot.Nodes,
                    PlayerRow = snapshot.PlayerRow,
                    NodeCount = snapshot.Nodes.Count,
                    MapName = $"Act_{state.Act}",
                    CurrentNodeId = snapshot.CurrentNodeId,
                    OriginNodeId = snapshot.OriginNodeId,
                    AvailableNextNodeIds =
                        snapshot.AvailableNextNodeIds,
                    BossNodeIds = snapshot.BossNodeIds,
                    BossEncounterIds = snapshot.BossEncounterIds,
                }
            );
        }
    }

    internal static PendingDecisionView? EmitRouteChoice(
        RunStateSnapshot state,
        MapSnapshot snapshot,
        object decisionSource)
    {
        if (!ReleaseCapabilityGate.IsEnabled("route_choice"))
        {
            return null;
        }
        lock (WriteGate)
        {
            if (!EnsureStableRunIdentity(state))
            {
                return null;
            }
            if (string.IsNullOrWhiteSpace(snapshot.OriginNodeId)
                || string.IsNullOrWhiteSpace(snapshot.MapFingerprint)
                || snapshot.AvailableNextNodeIds.Count == 0)
            {
                return null;
            }
            var routeContext = CreateRouteContext(state, snapshot);
            var candidates = snapshot.AvailableNextNodeIds
                .Select((nodeId, index) => new PendingDecisionCandidate(
                    nodeId,
                    routeContext.Nodes.FirstOrDefault(node =>
                        node.NodeId == nodeId)?.Label ?? nodeId,
                    index,
                    true
                ))
                .ToList();
            var opportunity = new RouteOpportunityIdentity(
                state.Act,
                snapshot.OriginNodeId,
                snapshot.MapFingerprint
            );
            var observationFingerprint = CreateRouteObservationFingerprint(
                state,
                routeContext,
                snapshot.MapFingerprint,
                _routeMode
            );
            var pending = _pendingDecision;
            var sameOpportunity = pending is not null
                && pending.EventType == "route_choice"
                && pending.RouteOpportunity == opportunity;
            if (sameOpportunity
                && pending!.ObservationFingerprint == observationFingerprint)
            {
                return ToPendingDecisionView(pending);
            }

            // Close the previous decision using its own last observed state.
            // Moving this below _lastState=state would attach a new origin to
            // an old decision closure during travel.
            if (_pendingDecision is not null && !sameOpportunity)
            {
                if (!ClosePendingDecision(new DecisionOutcome { Kind = "closed_unknown" }))
                {
                    return null;
                }
            }
            _lastState = state;
            var decisionId = sameOpportunity
                ? pending!.DecisionId
                : TryRecoverRouteDecisionId(routeContext, state)
                    ?? Guid.NewGuid().ToString("N");
            var eventId = Write(
                "route_choice",
                state,
                new List<DecisionOption>(),
                decision: new DecisionContext
                {
                    CanSkip = false,
                    CanReroll = false,
                    RewardSource = "MAP",
                },
                mapContext: routeContext,
                decisionId: decisionId
            );
            if (eventId is not null)
            {
                _pendingDecisionParent = null;
                _pendingChildParent = null;
                _resumeDecisionRecoveryAvailable = false;
                var screenOwner = sameOpportunity
                    ? pending!.ScreenOwner
                    : null;
                _pendingDecision = new PendingDecision(
                    eventId,
                    decisionId,
                    _sequence,
                    _routeMode,
                    false,
                    [],
                    candidates,
                    "route_choice",
                    decisionSource,
                    opportunity,
                    routeContext,
                    observationFingerprint
                )
                {
                    ScreenOwner = screenOwner,
                };
                return ToPendingDecisionView(_pendingDecision);
            }
            // Do not mutate the pending identity/observation on a failed
            // commit.  RouteChoiceObserver will retry the same observation.
            return null;
        }
    }

    internal static bool EmitRouteSelected(
        NMapScreen owner,
        string selectedNodeId)
    {
        lock (WriteGate)
        {
            var pending = _pendingDecision;
            if (pending is null
                || pending.EventType != "route_choice"
                || !ReferenceEquals(pending.ScreenOwner, owner))
            {
                return false;
            }
            var candidate = pending.Candidates.FirstOrDefault(candidate =>
                candidate.CandidateId == selectedNodeId);
            if (candidate is null)
            {
                return ClosePendingDecision(
                    new DecisionOutcome { Kind = "closed_unknown" },
                    expectedDecisionId: pending.DecisionId
                );
            }
            return ClosePendingDecision(
                new DecisionOutcome
                {
                    Kind = "selected",
                    SelectedCandidateId = candidate.CandidateId,
                },
                expectedDecisionId: pending.DecisionId
            );
        }
    }

    internal static PendingDecisionView? EmitGenericDecision(
        string eventType,
        List<DecisionCandidateEnvelope> candidates,
        DecisionContext decision,
        object decisionSource,
        DecisionParentContext? decisionParent = null)
    {
        if (!ReleaseCapabilityGate.IsEnabled(eventType)
            || eventType is not (
            "merchant"
            or "rest_site"
            or "neow_choice"
            or "event_choice"
            or "deck_edit"
        ) || candidates.Count == 0)
        {
            return null;
        }
        if (!RunStateReader.TryCapture(out var state) || state is null)
        {
            return null;
        }
        lock (WriteGate)
        {
            if (!EnsureStableRunIdentity(state))
            {
                return null;
            }
            var mapContext = TryCaptureCurrentMapContext(state);
            var fingerprint = CreateGenericObservationFingerprint(
                state,
                eventType,
                candidates,
                decision,
                decisionParent,
                mapContext
            );
            var pending = _pendingDecision;
            var sameDecision = pending is not null
                && pending.EventType == eventType
                && ReferenceEquals(
                    pending.DecisionSource,
                    decisionSource
                );
            if (sameDecision
                && pending!.ObservationFingerprint == fingerprint)
            {
                return ToPendingDecisionView(pending);
            }
            if (pending is not null && !sameDecision)
            {
                if (!ClosePendingDecision(new DecisionOutcome
                {
                    Kind = "closed_unknown",
                }))
                {
                    return null;
                }
            }
            _lastState = state;
            var decisionId = sameDecision
                ? pending!.DecisionId
                : $"{_runId}:{eventType}:{Guid.NewGuid():N}";
            var eventId = Write(
                eventType,
                state,
                [],
                decision: decision,
                mapContext: mapContext,
                decisionId: decisionId,
                genericCandidates: candidates,
                decisionParent: decisionParent
            );
            if (eventId is null)
            {
                return null;
            }
            _resumeDecisionRecoveryAvailable = false;
            _pendingDecisionParent = decisionParent;
            if (decisionParent is null)
            {
                _pendingChildParent = null;
            }
            else if (
                _pendingChildParent is not null
                && ReferenceEquals(
                    _pendingChildParent.Context,
                    decisionParent
                )
            )
            {
                ConsumePendingChildParent(decisionParent);
            }
            var screenOwner = sameDecision
                ? pending!.ScreenOwner
                : null;
            _pendingDecision = new PendingDecision(
                eventId,
                decisionId,
                _sequence,
                _routeMode,
                decision.CanSkip,
                [],
                candidates.Select((candidate, index) =>
                    new PendingDecisionCandidate(
                        candidate.CandidateId,
                        CandidateCaption(candidate),
                        index,
                        candidate.Eligible,
                        candidate.Kind,
                        candidate.Payload
                    )
                ).ToList(),
                eventType,
                decisionSource,
                null,
                mapContext,
                fingerprint
            )
            {
                ScreenOwner = screenOwner,
            };
            return ToPendingDecisionView(_pendingDecision);
        }
    }

    internal static bool EmitGenericSelected(
        object owner,
        string selectedCandidateId)
    {
        lock (WriteGate)
        {
            var pending = _pendingDecision;
            if (pending is null
                || pending.EventType is "card_reward" or "route_choice"
                || !ReferenceEquals(pending.ScreenOwner, owner))
            {
                return false;
            }
            var candidate = pending.Candidates.FirstOrDefault(item =>
                item.CandidateId == selectedCandidateId
                && item.Eligible);
            if (candidate is null)
            {
                return ClosePendingDecision(
                    new DecisionOutcome { Kind = "closed_unknown" },
                    expectedDecisionId: pending.DecisionId
                );
            }
            return ClosePendingDecision(
                new DecisionOutcome
                {
                    Kind = "selected",
                    SelectedCandidateId = selectedCandidateId,
                },
                expectedDecisionId: pending.DecisionId
            );
        }
    }

    internal static bool EmitGenericOwnerClosed(object owner)
    {
        lock (WriteGate)
        {
            var pending = _pendingDecision;
            if (pending is null
                || pending.EventType is "card_reward" or "route_choice"
                || !ReferenceEquals(pending.ScreenOwner, owner))
            {
                return false;
            }
            return ClosePendingDecision(
                new DecisionOutcome { Kind = "closed_unknown" },
                expectedDecisionId: pending.DecisionId
            );
        }
    }

    internal static DecisionParentContext? PeekDecisionParent(
        string expectedChildKind,
        params string[] allowedSourceTypes)
    {
        lock (WriteGate)
        {
            return PeekPendingChildParent(
                expectedChildKind,
                allowedSourceTypes
            );
        }
    }

    internal static void ClearUnconsumedDecisionParentAtBoundary()
    {
        lock (WriteGate)
        {
            if (_pendingChildParent is not null)
            {
                // Boundary invalidation is semantically the same as
                // consuming this one-shot authorization.  Preserve its key
                // before clearing so lazy active-run recovery cannot revive
                // it for a delayed callback in this process.
                _consumedChildParentKey = ChildParentKey(
                    _pendingChildParent
                );
            }
            _pendingChildParent = null;
        }
    }

    internal static bool HasPendingCardRewardParent(string sourceType)
    {
        lock (WriteGate)
        {
            return _pendingDecision?.EventType == "card_reward"
                && _pendingDecisionParent?.SourceType == sourceType;
        }
    }

    internal static DecisionParentContext?
        ResolveDeckEditParent(string operation)
    {
        lock (WriteGate)
        {
            var allowedSourceTypes = new[]
            {
                "rest_site",
                "merchant",
                "event_choice",
                "neow_choice",
            };
            var expectedChildKind = $"deck_edit:{operation}";
            var existing = PeekPendingChildParent(
                expectedChildKind,
                allowedSourceTypes
            );
            if (existing is not null)
            {
                return existing;
            }
            var effectKind = operation switch
            {
                "upgrade" => "upgrade_card",
                "remove" => "remove_card",
                "transform" => "transform_card",
                _ => null,
            };
            var pending = _pendingDecision;
            if (effectKind is null
                || pending is null
                || !allowedSourceTypes.Contains(
                    pending.EventType,
                    StringComparer.Ordinal
                ))
            {
                return null;
            }
            // The child screen itself proves that a deck-edit operation was
            // opened.  We only promote the parent when exactly one currently
            // eligible candidate declares that exact structured effect.
            // Ambiguous event/Neow options remain unlinked and fail closed.
            var matches = pending.Candidates
                .Where(candidate =>
                    candidate.Eligible
                    && candidate.Payload?.Effects.Any(effect =>
                        effect.Kind == effectKind
                        && effect.Certainty == "exact"
                        && effect.TargetMode == "choose"
                    ) is true
                )
                .ToList();
            if (matches.Count != 1
                || !ClosePendingDecision(
                    new DecisionOutcome
                    {
                        Kind = "selected",
                        SelectedCandidateId =
                            matches[0].CandidateId,
                    },
                    expectedDecisionId: pending.DecisionId
                ))
            {
                return null;
            }
            return PeekPendingChildParent(
                expectedChildKind,
                allowedSourceTypes
            );
        }
    }

    internal static PendingDecisionView? EmitRouteModeUpdated(
        NMapScreen owner,
        string expectedDecisionId,
        string requestedMode)
    {
        lock (WriteGate)
        {
            var pending = _pendingDecision;
            var state = _lastState;
            if (!GuideRouteModes.IsValid(requestedMode)
                || pending is null
                || state is null
                || pending.EventType != "route_choice"
                || pending.DecisionId != expectedDecisionId
                || !ReferenceEquals(pending.ScreenOwner, owner)
                || pending.RouteContext is null
                || pending.RouteOpportunity is null)
            {
                return null;
            }
            if (pending.RouteMode == requestedMode)
            {
                return ToPendingDecisionView(pending);
            }

            var observationFingerprint =
                CreateRouteObservationFingerprint(
                    state,
                    pending.RouteContext,
                    pending.RouteOpportunity.MapFingerprint,
                    requestedMode
                );
            var eventId = Write(
                "route_choice",
                state,
                new List<DecisionOption>(),
                decision: new DecisionContext
                {
                    CanSkip = false,
                    CanReroll = false,
                    RewardSource = "MAP",
                },
                mapContext: pending.RouteContext,
                decisionId: pending.DecisionId,
                routeMode: requestedMode
            );
            if (eventId is null)
            {
                // Preference state is committed with the event.  A failed
                // queue write must leave both mode and visible owner on the
                // previous committed observation.
                return null;
            }

            _routeMode = requestedMode;
            _resumeDecisionRecoveryAvailable = false;
            _pendingDecision = new PendingDecision(
                eventId,
                pending.DecisionId,
                _sequence,
                requestedMode,
                false,
                [],
                pending.Candidates,
                "route_choice",
                pending.DecisionSource,
                pending.RouteOpportunity,
                pending.RouteContext,
                observationFingerprint
            )
            {
                ScreenOwner = owner,
            };
            return ToPendingDecisionView(_pendingDecision);
        }
    }

    private static MapChoiceContext CreateRouteContext(
        RunStateSnapshot state,
        MapSnapshot snapshot
    )
        => new()
        {
            Nodes = snapshot.Nodes,
            PlayerRow = snapshot.PlayerRow,
            NodeCount = snapshot.Nodes.Count,
            MapName = $"Act_{state.Act}",
            CurrentNodeId = snapshot.CurrentNodeId,
            OriginNodeId = snapshot.OriginNodeId,
            AvailableNextNodeIds = snapshot.AvailableNextNodeIds,
            BossNodeIds = snapshot.BossNodeIds,
            BossEncounterIds = snapshot.BossEncounterIds,
        };

    private static string CreateRouteObservationFingerprint(
        RunStateSnapshot state,
        MapChoiceContext context,
        string mapFingerprint,
        string routeMode
    )
    {
        // Serialize the complete exported state and complete logical map
        // context.  This avoids a fragile hand-maintained list: any policy
        // input such as Ascension, CaptureWarnings, Character or potion-slot
        // capacity now advances the same route decision as UPDATED.
        var observation = JsonSerializer.Serialize(new
        {
            State = state,
            MapContext = context,
            MapFingerprint = mapFingerprint,
            RouteMode = routeMode,
        }, JsonOptions);
        return Convert.ToHexString(
            SHA256.HashData(Encoding.UTF8.GetBytes(observation))
        );
    }

    private static string CreateGenericObservationFingerprint(
        RunStateSnapshot state,
        string eventType,
        List<DecisionCandidateEnvelope> candidates,
        DecisionContext decision,
        DecisionParentContext? decisionParent,
        MapChoiceContext? mapContext)
    {
        var observation = JsonSerializer.Serialize(new
        {
            State = state,
            EventType = eventType,
            Candidates = candidates,
            Decision = decision,
            DecisionParent = decisionParent,
            MapContext = mapContext,
            RouteMode = _routeMode,
        }, JsonOptions);
        return Convert.ToHexString(
            SHA256.HashData(Encoding.UTF8.GetBytes(observation))
        );
    }

    private static MapChoiceContext? TryCaptureCurrentMapContext(
        RunStateSnapshot state)
    {
        var player = RunStateReader.GetObservedPlayer();
        if (player is not null)
        {
            var snapshot = MapNodeReader.Read(player);
            if (snapshot.Nodes.Count > 0
                && !string.IsNullOrWhiteSpace(snapshot.MapFingerprint))
            {
                return CreateRouteContext(state, snapshot);
            }
        }
        if (!state.CaptureWarnings.Contains(
            "map_snapshot_unavailable"
        ))
        {
            state.CaptureWarnings.Add("map_snapshot_unavailable");
        }
        return null;
    }

    private static string CandidateCaption(
        DecisionCandidateEnvelope candidate)
    {
        var costs = candidate.Costs
            .Where(cost => cost.Amount > 0)
            .Select(cost => cost.Kind switch
            {
                "gold" => $"{cost.Amount} 金币",
                "hp" => $"{cost.Amount} 生命",
                "max_hp" => $"{cost.Amount} 最大生命",
                "energy" => $"{cost.Amount} 能量",
                _ => $"{cost.Amount} {cost.ResourceId ?? "资源"}",
            })
            .ToList();
        var caption = costs.Count == 0
            ? candidate.Label
            : $"{candidate.Label} · {string.Join(" + ", costs)}";
        return candidate.Eligible
            ? caption
            : $"{caption}（不可用）";
    }

    private static bool ClosePendingDecision(
        DecisionOutcome outcome,
        string? expectedDecisionId = null,
        object? expectedDecisionSource = null)
    {
        var pending = _pendingDecision;
        var state = _lastState;
        if (pending is null || state is null)
        {
            return false;
        }
        if (expectedDecisionId is not null
            && pending.DecisionId != expectedDecisionId)
        {
            return false;
        }
        if (expectedDecisionSource is not null
            && !ReferenceEquals(
                pending.DecisionSource,
                expectedDecisionSource
            ))
        {
            return false;
        }
        var eventId = Write(
            "decision_closed",
            state,
            new List<DecisionOption>(),
                parentEventId: pending.ParentEventId,
                outcome: outcome,
                decisionId: pending.DecisionId
        );
        if (eventId is null)
        {
            return false;
        }
        var expectedChildKind = outcome.Kind == "selected"
            && !string.IsNullOrWhiteSpace(
                outcome.SelectedCandidateId
            )
            ? ExpectedChildKind(
                pending,
                outcome.SelectedCandidateId!
            )
            : null;
        if (expectedChildKind is not null
            && !string.IsNullOrWhiteSpace(
                outcome.SelectedCandidateId
            )
            && pending.EventType is (
                "merchant"
                or "rest_site"
                or "neow_choice"
                or "event_choice"
            ))
        {
            _pendingChildParent = new PendingChildParent(
                _runId,
                expectedChildKind,
                _sequence,
                new DecisionParentContext
                {
                    DecisionId = pending.DecisionId,
                    CandidateId = outcome.SelectedCandidateId,
                    SourceType = pending.EventType,
                    // Candidate IDs may use the protocol's 240-character
                    // allowance while source_id is deliberately capped at
                    // 200.  A compact digest preserves stable identity
                    // without truncation collisions.
                    SourceId = StableParentSourceId(
                        outcome.SelectedCandidateId!
                    ),
                }
            );
        }
        else if (pending.EventType is (
            "merchant"
            or "rest_site"
            or "neow_choice"
            or "event_choice"
        ))
        {
            _pendingChildParent = null;
        }
        _pendingDecision = null;
        _pendingDecisionParent = null;
        return true;
    }

    private static DecisionParentContext? PeekPendingChildParent(
        string expectedChildKind,
        params string[] allowedSourceTypes)
    {
        var pending = _pendingChildParent
            ?? TryRecoverPendingChildParent(
                expectedChildKind,
                allowedSourceTypes
            );
        _pendingChildParent = pending;
        if (pending is null
            || pending.RunId != _runId
            || pending.ExpectedChildKind != expectedChildKind
            || !allowedSourceTypes.Contains(
                pending.Context.SourceType,
                StringComparer.Ordinal
            ))
        {
            _pendingChildParent = null;
            return null;
        }
        return pending.Context;
    }

    private static void ConsumePendingChildParent(
        DecisionParentContext context)
    {
        var pending = _pendingChildParent;
        if (pending is not null
            && ReferenceEquals(pending.Context, context))
        {
            _consumedChildParentKey = ChildParentKey(pending);
            _pendingChildParent = null;
        }
    }

    private static string ChildParentKey(PendingChildParent pending)
    {
        return string.Join(
            ":",
            pending.RunId,
            pending.Context.DecisionId,
            pending.Context.CandidateId,
            pending.ExpectedChildKind,
            pending.ParentCloseSequence
        );
    }

    private static bool DecisionParentsEqual(
        DecisionParentContext left,
        DecisionParentContext right)
        => left.DecisionId == right.DecisionId
            && left.CandidateId == right.CandidateId
            && left.SourceType == right.SourceType
            && left.SourceId == right.SourceId;

    private static PendingChildParent? TryRecoverPendingChildParent(
        string expectedChildKind,
        IReadOnlyCollection<string> allowedSourceTypes)
    {
        var directory = GetExchangeDirectory();
        if (directory is null)
        {
            return null;
        }
        var checkpointPath = Path.Combine(directory, "active-run.json");
        if (!File.Exists(checkpointPath))
        {
            return null;
        }
        try
        {
            using var document = JsonDocument.Parse(
                File.ReadAllText(checkpointPath)
            );
            var root = document.RootElement;
            if (!root.TryGetProperty(
                    "checkpoint_version",
                    out var checkpointVersion
                )
                || !checkpointVersion.TryGetInt32(out var version)
                || version != 3
                || !StringPropertyEquals(root, "run_id", _runId)
                || !StringPropertyEquals(
                    root,
                    "release_fingerprint",
                    StateEvent.ReleaseFingerprint
                )
                || !root.TryGetProperty(
                    "current_decision",
                    out var currentDecision
                )
                || currentDecision.ValueKind != JsonValueKind.Null
                || !root.TryGetProperty(
                    "closed_decision",
                    out var closedDecision
                )
                || closedDecision.ValueKind != JsonValueKind.Object
                || !StringPropertyEquals(
                    closedDecision,
                    "run_id",
                    _runId
                )
                || !StringPropertyEquals(
                    closedDecision,
                    "release_fingerprint",
                    StateEvent.ReleaseFingerprint
                )
                || !StringPropertyEquals(
                    closedDecision,
                    "event_type",
                    "decision_closed"
                )
                || !TryReadRequiredString(
                    closedDecision,
                    "decision_id",
                    out var parentDecisionId
                )
                || !closedDecision.TryGetProperty(
                    "sequence",
                    out var sequenceElement
                )
                || !sequenceElement.TryGetInt64(out var closeSequence)
                || closeSequence < 1
                || !closedDecision.TryGetProperty(
                    "result",
                    out var result
                )
                || result.ValueKind != JsonValueKind.Object
                || !StringPropertyEquals(
                    result,
                    "decision_id",
                    parentDecisionId
                )
                || !result.TryGetProperty(
                    "child_expectation",
                    out var expectation
                )
                || expectation.ValueKind != JsonValueKind.Object
                || expectation.EnumerateObject().Count() != 7
                || !StringPropertyEquals(
                    expectation,
                    "parent_decision_id",
                    parentDecisionId
                )
                || !TryReadRequiredString(
                    expectation,
                    "parent_candidate_id",
                    out var parentCandidateId
                )
                || !TryReadRequiredString(
                    expectation,
                    "source_type",
                    out var sourceType
                )
                || !allowedSourceTypes.Contains(
                    sourceType,
                    StringComparer.Ordinal
                )
                || !StringPropertyEquals(
                    expectation,
                    "source_id",
                    StableParentSourceId(parentCandidateId)
                )
                || !expectation.TryGetProperty(
                    "parent_close_sequence",
                    out var expectedSequence
                )
                || !expectedSequence.TryGetInt64(
                    out var parentCloseSequence
                )
                || parentCloseSequence != closeSequence
                || !ChildExpectationMatches(
                    expectation,
                    expectedChildKind
                )
                || !StringPropertyEquals(
                    result,
                    "chosen_option",
                    parentCandidateId
                )
                || !closedDecision.TryGetProperty(
                    "outcome",
                    out var outcome
                )
                || outcome.ValueKind != JsonValueKind.Object
                || !StringPropertyEquals(outcome, "kind", "selected")
                || !StringPropertyEquals(
                    outcome,
                    "selected_candidate_id",
                    parentCandidateId
                ))
            {
                return null;
            }
            var recovered = new PendingChildParent(
                _runId,
                expectedChildKind,
                closeSequence,
                new DecisionParentContext
                {
                    DecisionId = parentDecisionId,
                    CandidateId = parentCandidateId,
                    SourceType = sourceType,
                    SourceId = StableParentSourceId(parentCandidateId),
                }
            );
            if (ChildParentKey(recovered) == _consumedChildParentKey)
            {
                return null;
            }
            Log.Info(
                "[STS2-Guide] Recovered verified child decision parent "
                + parentDecisionId + "."
            );
            return recovered;
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Child parent recovery failed: "
                + exception.Message
            );
            return null;
        }
    }

    private static bool ChildExpectationMatches(
        JsonElement expectation,
        string expectedChildKind)
    {
        if (expectedChildKind == "card_reward")
        {
            return StringPropertyEquals(
                    expectation,
                    "child_decision_type",
                    "card_reward"
                )
                && expectation.TryGetProperty("operation", out var operation)
                && operation.ValueKind == JsonValueKind.Null;
        }
        const string prefix = "deck_edit:";
        if (!expectedChildKind.StartsWith(
                prefix,
                StringComparison.Ordinal
            ))
        {
            return false;
        }
        var expectedOperation = expectedChildKind[prefix.Length..];
        return expectedOperation is "upgrade" or "remove" or "transform"
            && StringPropertyEquals(
                expectation,
                "child_decision_type",
                "deck_edit"
            )
            && StringPropertyEquals(
                expectation,
                "operation",
                expectedOperation
            );
    }

    private static string? ExpectedChildKind(
        PendingDecision pending,
        string selectedCandidateId)
    {
        var candidate = pending.Candidates.SingleOrDefault(value =>
            value.Eligible
            && value.CandidateId == selectedCandidateId
        );
        if (candidate?.Payload is null)
        {
            return null;
        }
        var expectations = candidate.Payload.Effects
            .Where(effect =>
                effect.Certainty == "exact"
            )
            .Select(effect =>
                effect.Kind == "followup_choice"
                    && effect.ChildDecisionType == "card_reward"
                    ? "card_reward"
                    : effect.TargetMode == "choose"
                        ? effect.Kind switch
                        {
                            "upgrade_card" => "deck_edit:upgrade",
                            "remove_card" => "deck_edit:remove",
                            "transform_card" => "deck_edit:transform",
                            _ => null,
                        }
                        : null
            )
            .Where(value => value is not null)
            .Distinct(StringComparer.Ordinal)
            .ToList();
        return expectations.Count == 1 ? expectations[0] : null;
    }

    private static string StableParentSourceId(string candidateId)
        => "SOURCE_"
            + Convert.ToHexString(
                SHA256.HashData(
                    Encoding.UTF8.GetBytes(candidateId)
                )
            )[..24];

    private static string? Write(
        string eventType,
        RunStateSnapshot state,
        List<DecisionOption> options,
        DecisionContext? decision = null,
        string? parentEventId = null,
        DecisionOutcome? outcome = null,
        MapChoiceContext? mapContext = null,
        RunResult? runResult = null,
        string? decisionId = null,
        string? routeMode = null,
        List<DecisionCandidateEnvelope>? genericCandidates = null,
        DecisionParentContext? decisionParent = null
    )
    {
        // A sequence is a committed queue fact, not an attempted-write
        // counter.  Do not advance _sequence until the durable event queue
        // file has been atomically replaced.
        var sequence = checked(_sequence + 1);
        var eventId = $"{_runId}:{sequence}";
        var emittedRouteMode = routeMode ?? _routeMode;
        if (!GuideRouteModes.IsValid(emittedRouteMode))
        {
            Log.Error(
                "[STS2-Guide] Refusing to emit an invalid route mode."
            );
            return null;
        }
        var emittedCandidates = genericCandidates
            ?? CreateGenericCandidates(
                eventType,
                options,
                decision,
                mapContext
            );
        var stateEvent = new StateEvent
        {
            EventId = eventId,
            EventType = eventType,
            EmittedAt = DateTimeOffset.UtcNow.ToString("O"),
            GameVersion = GameVersionReader.Read(),
            StateRevision = sequence,
            GameAssemblySha256 =
                GameAssemblyIdentityReader.ReadSha256(),
            GuidePreferences = new GuidePreferences
            {
                RouteMode = emittedRouteMode,
            },
            RunId = _runId,
            Sequence = sequence,
            DecisionId = decisionId,
            State = state,
            // v1-v8 card-only options remain replayable in Python, but v9
            // production writes one authoritative generic candidate list.
            Options = [],
            Candidates = emittedCandidates,
            Decision = decision,
            DecisionParent = decisionParent,
            ParentEventId = parentEventId,
            Outcome = outcome,
            MapContext = mapContext,
            RunResult = runResult,
        };
        var json = JsonSerializer.Serialize(stateEvent, JsonOptions);
        var outputPath = ProjectSettings.GlobalizePath(
            "user://STS2Guide/state-event.json"
        );
        var directory = Path.GetDirectoryName(outputPath);
        if (string.IsNullOrWhiteSpace(directory))
        {
            Log.Error("[STS2-Guide] Invalid state-event output path.");
            return null;
        }

        try
        {
            Directory.CreateDirectory(directory);
            var eventDirectory = Path.Combine(directory, "events");
            Directory.CreateDirectory(eventDirectory);
            var eventPath = Path.Combine(
                eventDirectory,
                $"{_runId}-{sequence:D12}-{eventType}.json"
            );
            WriteAtomically(eventPath, json);
            _sequence = sequence;
            _runIdentityLockedByEmission = true;

            try
            {
                WriteAtomically(outputPath, json);
            }
            catch (Exception latestException)
            {
                Log.Error(
                    "[STS2-Guide] Latest-event mirror write failed; "
                    + $"queued event is intact: {latestException.Message}"
                );
            }
            Log.Info(
                $"[STS2-Guide] Emitted {eventType} event "
                + $"{stateEvent.EventId}."
            );
            return eventId;
        }
        catch (Exception exception)
        {
            Log.Error(
                $"[STS2-Guide] State-event write failed: "
                + exception.Message
            );
            return null;
        }
    }

    private static List<DecisionCandidateEnvelope> CreateGenericCandidates(
        string eventType,
        List<DecisionOption> options,
        DecisionContext? decision,
        MapChoiceContext? mapContext
    )
    {
        if (eventType == "card_reward")
        {
            var cardCandidates = options
                .Select((option, index) => new DecisionCandidateEnvelope
                {
                    CandidateId = option.CandidateId
                        ?? CandidateId(index, option),
                    Kind = "card",
                    EntityId = option.Card,
                    Label = option.Card,
                    Eligible = true,
                    UnavailableReason = null,
                    Costs = [],
                    Payload = new CandidatePayload
                    {
                        Card = option.Card,
                        Upgrades = option.Upgrades,
                        Enchantment = option.Enchantment,
                        EnchantmentAmount = option.EnchantmentAmount,
                        Affliction = option.Affliction,
                        AfflictionAmount = option.AfflictionAmount,
                    },
                })
                .ToList();
            if (decision?.CanSkip == true)
            {
                cardCandidates.Add(new DecisionCandidateEnvelope
                {
                    CandidateId = "skip",
                    Kind = "skip",
                    EntityId = null,
                    Label = "跳过",
                    Eligible = true,
                    UnavailableReason = null,
                    Costs = [],
                    Payload = new CandidatePayload(),
                });
            }
            return cardCandidates;
        }
        if (eventType == "route_choice" && mapContext is not null)
        {
            var nodes = mapContext.Nodes.ToDictionary(
                node => node.NodeId,
                node => node
            );
            return mapContext.AvailableNextNodeIds
                .Select(nodeId => new DecisionCandidateEnvelope
                {
                    CandidateId = nodeId,
                    Kind = "route_node",
                    EntityId = nodeId,
                    Label = (
                        nodes.TryGetValue(nodeId, out var node)
                            ? node.Label
                            : null
                    ) ?? nodeId,
                    Eligible = true,
                    UnavailableReason = null,
                    Costs = [],
                    Payload = new CandidatePayload
                    {
                        NodeId = nodeId,
                    },
                })
                .ToList();
        }
        return [];
    }

    private static void WriteAtomically(string path, string json)
    {
        var temporaryPath = $"{path}.tmp";
        File.WriteAllText(temporaryPath, json);
        File.Move(temporaryPath, path, true);
    }

    private static bool EnsureStableRunIdentity(RunStateSnapshot state)
    {
        if (_runIdentityStable && !_runTransitionPending)
        {
            return true;
        }
        if (_runIdentityLockedByEmission && !_runTransitionPending)
        {
            Log.Error(
                "[STS2-Guide] Stable run identity became available after "
                + "events were emitted; keeping temporary identity."
            );
            return true;
        }
        // Read identity preferring the current player's RunState seed.
        // The player's seed is fresh for this run; RunManager.History.Seed
        // may be stale from a previous ended run.
        var identity = RunIdentityReader.Read();
        if (!identity.IsStable)
        {
            if (!state.CaptureWarnings.Contains("run_identity_fallback"))
            {
                state.CaptureWarnings.Add("run_identity_fallback");
            }
            return !_runTransitionPending;
        }

        if (_runTransitionPending)
        {
            var previousState = _lastState;
            var hasUnendedState = previousState is not null && !_ended;
            if (RunIdentityGuard.ShouldCloseActiveRun(
                hasUnendedState,
                _runId,
                _runIdentityStable,
                identity.RunId,
                identity.IsStable
            ))
            {
                var syntheticEndEventId = Write(
                    "run_ended",
                    previousState!,
                    new List<DecisionOption>(),
                    runResult: CreateRunResult("abandon")
                );
                if (syntheticEndEventId is null)
                {
                    Log.Error(
                        "[STS2-Guide] Deferred run transition still cannot "
                        + "commit the previous run's synthetic abandon; "
                        + "retaining all previous run state for retry."
                    );
                    return false;
                }
                RememberCurrentIdentityEnded();
            }
        }
        // The ended-run guard survives BeginRun's temporary fallback. A
        // stale current-player seed cannot reclaim the ended identity.
        if (!RunIdentityGuard.TryAcceptStableIdentity(
            ref _lastEndedStableRunId,
            identity.RunId,
            identity.IsStable
        ))
        {
            Log.Error(
                $"[STS2-Guide] Stable identity {identity.RunId} matches "
                + "previously ended run; keeping temporary ID to prevent "
                + "cross-run contamination."
            );
            if (!state.CaptureWarnings.Contains("run_identity_fallback"))
            {
                state.CaptureWarnings.Add("run_identity_fallback");
            }
            return !_runTransitionPending;
        }
        if (_runTransitionPending)
        {
            // The caller already captured the new run's player/state. Keep
            // that observation available for the event being emitted.
            ActivateRun(identity, clearObservedPlayer: false);
            return true;
        }
        _runId = identity.RunId;
        _runStartedAt = identity.StartedAt;
        _runIdentityStable = true;
        _sequence = RecoverSequence(_runId);
        _routeMode = RecoverRouteMode(_runId);
        return true;
    }

    private static void ActivateRun(
        RunIdentity identity,
        bool clearObservedPlayer)
    {
        _runId = identity.RunId;
        _runStartedAt = identity.StartedAt;
        _runIdentityStable = identity.IsStable;
        _sequence = RecoverSequence(_runId);
        _routeMode = RecoverRouteMode(_runId);
        _lastState = null;
        _pendingDecision = null;
        _pendingDecisionParent = null;
        _pendingChildParent = null;
        _resumeDecisionRecoveryAvailable = true;
        _runIdentityLockedByEmission = false;
        _runTransitionPending = false;
        _ended = false;
        if (clearObservedPlayer)
        {
            RunStateReader.Clear();
        }
        Log.Info(
            $"[STS2-Guide] Run identity {_runId}; "
            + $"resuming sequence {_sequence}."
        );
    }

    private static void RememberCurrentIdentityEnded()
    {
        RunIdentityGuard.RememberEndedStableIdentity(
            ref _lastEndedStableRunId,
            _runId,
            _runIdentityStable
        );
    }

    private static RunResult CreateRunResult(string outcome)
    {
        int? finalScore = null;
        try
        {
            finalScore = Math.Max(
                0,
                SaveManager.Instance.GetCurrentScore()
            );
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Final score unavailable: "
                + exception.Message
            );
        }
        return new RunResult
        {
            Outcome = outcome,
            FinalScore = finalScore,
            StartedAt = _runStartedAt,
            EndedAt = DateTimeOffset.UtcNow.ToString("O"),
        };
    }

    private static long RecoverSequence(string runId)
    {
        var directory = GetExchangeDirectory();
        if (directory is null)
        {
            return 0;
        }
        var maximum = 0L;
        maximum = Math.Max(
            maximum,
            ReadSequence(
                Path.Combine(directory, "active-run.json"),
                runId,
                "last_sequence"
            )
        );
        maximum = Math.Max(
            maximum,
            ReadSequence(
                Path.Combine(directory, "state-event.json"),
                runId,
                "sequence"
            )
        );
        var eventDirectory = Path.Combine(directory, "events");
        if (Directory.Exists(eventDirectory))
        {
            foreach (var path in Directory.EnumerateFiles(
                eventDirectory,
                "*.json"
            ))
            {
                maximum = Math.Max(
                    maximum,
                    ReadSequence(path, runId, "sequence")
                );
            }
        }
        return maximum;
    }

    private static string RecoverRouteMode(string runId)
    {
        var directory = GetExchangeDirectory();
        if (directory is null)
        {
            return GuideRouteModes.Balanced;
        }

        var checkpointPath = Path.Combine(directory, "active-run.json");
        if (!File.Exists(checkpointPath))
        {
            return GuideRouteModes.Balanced;
        }
        try
        {
            using var document = JsonDocument.Parse(
                File.ReadAllText(checkpointPath)
            );
            var root = document.RootElement;
            if (!root.TryGetProperty(
                    "checkpoint_version",
                    out var checkpointVersion
                )
                || !checkpointVersion.TryGetInt32(out var version)
                || version != 3
                || !StringPropertyEquals(root, "run_id", runId)
                || !StringPropertyEquals(
                    root,
                    "release_fingerprint",
                    StateEvent.ReleaseFingerprint
                )
                || !TryReadGuideRouteMode(root, out var routeMode))
            {
                return GuideRouteModes.Balanced;
            }
            return routeMode;
        }
        catch
        {
            // active-run.json is the only authoritative preference source.
            // Missing, damaged, stale, or incompatible state always resets
            // the per-run preference instead of reviving queued observations.
            return GuideRouteModes.Balanced;
        }
    }

    private static bool TryReadGuideRouteMode(
        JsonElement root,
        out string routeMode)
    {
        routeMode = "";
        if (!root.TryGetProperty(
                "guide_preferences",
                out var preferences
            )
            || preferences.ValueKind != JsonValueKind.Object
            || preferences.EnumerateObject().Count() != 1
            || !TryReadRequiredString(
                preferences,
                "route_mode",
                out routeMode
            )
            || !GuideRouteModes.IsValid(routeMode))
        {
            routeMode = "";
            return false;
        }
        return true;
    }

    private static string? TryRecoverDecisionId(
        IReadOnlyList<DecisionOption> options,
        DecisionContext decision,
        RunStateSnapshot state)
    {
        if (!_resumeDecisionRecoveryAvailable)
        {
            return null;
        }
        var recovered = ReadResumableDecisionId(options, decision, state);
        if (recovered is null)
        {
            // Recovery is intentionally a one-shot resume path. A later card
            // reward in the same process must never reuse a stale checkpoint
            // merely because it happens to have similar candidates.
            _resumeDecisionRecoveryAvailable = false;
        }
        return recovered;
    }

    private static string? TryRecoverRouteDecisionId(
        MapChoiceContext context,
        RunStateSnapshot state)
    {
        if (!_resumeDecisionRecoveryAvailable)
        {
            return null;
        }
        var directory = GetExchangeDirectory();
        if (directory is null)
        {
            return null;
        }
        var candidates = new List<RouteDecisionRecoveryCandidate>();
        TryAddCheckpointRouteDecision(
            Path.Combine(directory, "active-run.json"),
            context,
            state,
            candidates
        );
        TryAddEventRouteDecision(
            Path.Combine(directory, "state-event.json"),
            context,
            state,
            candidates
        );
        var eventDirectory = Path.Combine(directory, "events");
        if (Directory.Exists(eventDirectory))
        {
            foreach (var path in Directory.EnumerateFiles(
                eventDirectory,
                "*.json"
            ))
            {
                TryAddEventRouteDecision(
                    path,
                    context,
                    state,
                    candidates
                );
            }
        }
        foreach (var candidate in candidates
            .Distinct()
            .OrderByDescending(candidate => candidate.Sequence))
        {
            if (HasLaterInvalidatingEvent(
                directory,
                candidate.DecisionId,
                candidate.Sequence
            ))
            {
                continue;
            }
            Log.Info(
                "[STS2-Guide] Resuming stable route decision identity "
                + candidate.DecisionId + "."
            );
            return candidate.DecisionId;
        }
        return null;
    }

    private static void TryAddCheckpointRouteDecision(
        string path,
        MapChoiceContext context,
        RunStateSnapshot state,
        List<RouteDecisionRecoveryCandidate> candidates)
    {
        if (!File.Exists(path))
        {
            return;
        }
        try
        {
            using var document = JsonDocument.Parse(
                File.ReadAllText(path)
            );
            var root = document.RootElement;
            if (!StringPropertyEquals(root, "run_id", _runId)
                || !StringPropertyEquals(
                    root,
                    "release_fingerprint",
                    StateEvent.ReleaseFingerprint
                )
                || !root.TryGetProperty(
                    "current_decision",
                    out var current
                )
                || current.ValueKind != JsonValueKind.Object
                || !StringPropertyEquals(
                    current,
                    "run_id",
                    _runId
                )
                || !StringPropertyEquals(
                    current,
                    "event_type",
                    "route_choice"
                )
                || !TryReadRequiredString(
                    current,
                    "decision_id",
                    out var currentDecisionId
                )
                || !TryReadRequiredString(
                    current,
                    "event_id",
                    out var currentEventId
                )
                || !current.TryGetProperty(
                    "sequence",
                    out var currentSequence
                )
                || !currentSequence.TryGetInt64(
                    out var currentSequenceValue
                )
                || !current.TryGetProperty("payload", out var payload)
                || payload.ValueKind != JsonValueKind.Object
                || !StringPropertyEquals(
                    payload,
                    "decision_id",
                    currentDecisionId
                )
                || !StringPropertyEquals(
                    payload,
                    "event_id",
                    currentEventId
                )
                || !payload.TryGetProperty(
                    "sequence",
                    out var payloadSequence
                )
                || !payloadSequence.TryGetInt64(
                    out var payloadSequenceValue
                )
                || payloadSequenceValue != currentSequenceValue)
            {
                return;
            }
            TryAddRouteDecisionPayload(
                payload,
                context,
                state,
                candidates
            );
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Active route checkpoint recovery failed: "
                + exception.Message
            );
        }
    }

    private static void TryAddEventRouteDecision(
        string path,
        MapChoiceContext context,
        RunStateSnapshot state,
        List<RouteDecisionRecoveryCandidate> candidates)
    {
        if (!File.Exists(path))
        {
            return;
        }
        try
        {
            using var document = JsonDocument.Parse(
                File.ReadAllText(path)
            );
            TryAddRouteDecisionPayload(
                document.RootElement,
                context,
                state,
                candidates
            );
        }
        catch
        {
            // Recovery is best-effort across independent atomic spool files.
        }
    }

    private static void TryAddRouteDecisionPayload(
        JsonElement payload,
        MapChoiceContext context,
        RunStateSnapshot state,
        List<RouteDecisionRecoveryCandidate> candidates)
    {
        if (!StringPropertyEquals(payload, "run_id", _runId)
            || !StringPropertyEquals(
                payload,
                "release_fingerprint",
                StateEvent.ReleaseFingerprint
            )
            || !StringPropertyEquals(
                payload,
                "event_type",
                "route_choice"
            )
            || !TryReadRequiredString(
                payload,
                "decision_id",
                out var decisionId
            )
            || !payload.TryGetProperty("sequence", out var sequence)
            || !sequence.TryGetInt64(out var value)
            || !RoutePayloadMatches(payload, context, state))
        {
            return;
        }
        candidates.Add(new RouteDecisionRecoveryCandidate(
            value,
            decisionId
        ));
    }

    private static bool RoutePayloadMatches(
        JsonElement payload,
        MapChoiceContext context,
        RunStateSnapshot state)
    {
        if (!payload.TryGetProperty("map_context", out var stored)
            || stored.ValueKind != JsonValueKind.Object
            || !StringPropertyEquals(payload, "run_id", _runId)
            || !StringPropertyEquals(payload, "event_type", "route_choice")
            || !payload.TryGetProperty("schema_version", out var schemaVersion)
            || !schemaVersion.TryGetInt32(out var version)
            || version is not (6 or 7 or 8 or 9)
            || (version == StateEvent.CurrentSchemaVersion
                && (!TryReadGuideRouteMode(
                        payload,
                        out var storedRouteMode
                    )
                    || storedRouteMode != _routeMode))
            || !payload.TryGetProperty("state", out var storedState)
            || storedState.ValueKind != JsonValueKind.Object
            || !IntPropertyEquals(storedState, "act", state.Act)
            || !NullableStringPropertyEquals(stored, "origin_node_id", context.OriginNodeId)
            || !stored.TryGetProperty("node_count", out var nodeCount)
            || !nodeCount.TryGetInt32(out var storedNodeCount)
            || storedNodeCount != context.NodeCount
            || !stored.TryGetProperty("available_next_node_ids", out var storedCandidates)
            || storedCandidates.ValueKind != JsonValueKind.Array
            || storedCandidates.GetArrayLength() != context.AvailableNextNodeIds.Count
            || !JsonStringSetMatches(stored, "boss_node_ids", context.BossNodeIds))
        {
            return false;
        }
        var storedCandidateIds = new List<string>();
        foreach (var candidate in storedCandidates.EnumerateArray())
        {
            if (candidate.ValueKind != JsonValueKind.String
                || string.IsNullOrWhiteSpace(candidate.GetString()))
            {
                return false;
            }
            storedCandidateIds.Add(candidate.GetString()!);
        }
        if (storedCandidateIds.Distinct(StringComparer.Ordinal).Count()
                != storedCandidateIds.Count
            || !storedCandidateIds.OrderBy(id => id, StringComparer.Ordinal)
                .SequenceEqual(context.AvailableNextNodeIds
                    .OrderBy(id => id, StringComparer.Ordinal)))
        {
            return false;
        }
        var storedFingerprint = MapNodeReader.CreateMapFingerprint(stored);
        var currentFingerprint = MapNodeReader.CreateMapFingerprint(context.Nodes);
        return storedFingerprint is not null
            && currentFingerprint is not null
            && storedFingerprint == currentFingerprint;
    }

    private static bool JsonStringSetMatches(
        JsonElement parent,
        string property,
        IReadOnlyList<string> expected
    )
    {
        if (!parent.TryGetProperty(property, out var values)
            || values.ValueKind != JsonValueKind.Array
            || values.GetArrayLength() != expected.Count)
        {
            return false;
        }
        var actual = new List<string>();
        foreach (var item in values.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String
                || string.IsNullOrWhiteSpace(item.GetString()))
            {
                return false;
            }
            actual.Add(item.GetString()!);
        }
        return actual.Distinct(StringComparer.Ordinal).Count() == actual.Count
            && actual.OrderBy(value => value, StringComparer.Ordinal)
                .SequenceEqual(expected.OrderBy(value => value, StringComparer.Ordinal));
    }

    private static string? ReadResumableDecisionId(
        IReadOnlyList<DecisionOption> options,
        DecisionContext decision,
        RunStateSnapshot state)
    {
        var directory = GetExchangeDirectory();
        if (directory is null)
        {
            return null;
        }
        var checkpointPath = Path.Combine(directory, "active-run.json");
        if (!File.Exists(checkpointPath))
        {
            return null;
        }
        try
        {
            using var document = JsonDocument.Parse(
                File.ReadAllText(checkpointPath)
            );
            var root = document.RootElement;
            if (!StringPropertyEquals(root, "run_id", _runId)
                || !StringPropertyEquals(
                    root,
                    "release_fingerprint",
                    StateEvent.ReleaseFingerprint
                )
                || !root.TryGetProperty(
                    "current_decision",
                    out var current
                )
                || current.ValueKind != JsonValueKind.Object
                || !StringPropertyEquals(current, "run_id", _runId)
                || !StringPropertyEquals(
                    current,
                    "event_type",
                    "card_reward"
                )
                || !TryReadRequiredString(
                    current,
                    "decision_id",
                    out var decisionId
                )
                || !current.TryGetProperty(
                    "sequence",
                    out var sequenceElement
                )
                || !sequenceElement.TryGetInt64(out var openSequence)
                || !current.TryGetProperty("payload", out var payload)
                || payload.ValueKind != JsonValueKind.Object
                || !StringPropertyEquals(
                    payload,
                    "release_fingerprint",
                    StateEvent.ReleaseFingerprint
                )
                || !StringPropertyEquals(payload, "run_id", _runId)
                || !StringPropertyEquals(
                    payload,
                    "event_type",
                    "card_reward"
                )
                || !StringPropertyEquals(
                    payload,
                    "decision_id",
                    decisionId
                )
                || !payload.TryGetProperty(
                    "schema_version",
                    out var schemaVersion
                )
                || !schemaVersion.TryGetInt32(out var version)
                || version is not (5 or 6 or 7 or 8 or 9)
                || (version == StateEvent.CurrentSchemaVersion
                    && (!TryReadGuideRouteMode(
                            payload,
                            out var storedRouteMode
                        )
                        || storedRouteMode != _routeMode))
                || !PayloadMatchesDecision(
                    payload,
                    options,
                    decision,
                    state
                )
                || HasLaterInvalidatingEvent(
                    directory,
                    decisionId,
                    openSequence
                ))
            {
                return null;
            }
            Log.Info(
                "[STS2-Guide] Resuming stable decision identity "
                + decisionId + "."
            );
            return decisionId;
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Active decision recovery failed: "
                + exception.Message
            );
            return null;
        }
    }

    private static bool PayloadMatchesDecision(
        JsonElement payload,
        IReadOnlyList<DecisionOption> options,
        DecisionContext decision,
        RunStateSnapshot state)
    {
        if (!PayloadStateMatches(payload, state)
            || !payload.TryGetProperty("decision", out var storedDecision)
            || storedDecision.ValueKind != JsonValueKind.Object
            || !BoolPropertyEquals(
                storedDecision,
                "can_skip",
                decision.CanSkip
            )
            || !BoolPropertyEquals(
                storedDecision,
                "can_reroll",
                decision.CanReroll
            )
            || !NullableStringPropertyEquals(
                storedDecision,
                "reward_source",
                decision.RewardSource
            ))
        {
            return false;
        }
        if (payload.TryGetProperty(
                "candidates",
                out var storedCandidates
            )
            && storedCandidates.ValueKind == JsonValueKind.Array
            && storedCandidates.GetArrayLength() > 0)
        {
            return GenericCardCandidatesMatch(
                storedCandidates,
                options,
                decision.CanSkip
            );
        }
        if (!payload.TryGetProperty("options", out var storedOptions)
            || storedOptions.ValueKind != JsonValueKind.Array
            || storedOptions.GetArrayLength() != options.Count)
        {
            return false;
        }
        var index = 0;
        foreach (var stored in storedOptions.EnumerateArray())
        {
            var expected = options[index];
            if (stored.ValueKind != JsonValueKind.Object
                || !StringPropertyEquals(
                    stored,
                    "candidate_id",
                    expected.CandidateId
                )
                || !StringPropertyEquals(stored, "card", expected.Card)
                || !IntPropertyEquals(
                    stored,
                    "upgrades",
                    expected.Upgrades
                )
                || !NullableStringPropertyEquals(
                    stored,
                    "enchantment",
                    expected.Enchantment
                )
                || !NullableIntPropertyEquals(
                    stored,
                    "enchantment_amount",
                    expected.EnchantmentAmount
                )
                || !NullableStringPropertyEquals(
                    stored,
                    "affliction",
                    expected.Affliction
                )
                || !NullableIntPropertyEquals(
                    stored,
                    "affliction_amount",
                    expected.AfflictionAmount
                ))
            {
                return false;
            }
            index++;
        }
        return true;
    }

    private static bool GenericCardCandidatesMatch(
        JsonElement storedCandidates,
        IReadOnlyList<DecisionOption> options,
        bool canSkip
    )
    {
        var expectedCount = options.Count + (canSkip ? 1 : 0);
        if (storedCandidates.GetArrayLength() != expectedCount)
        {
            return false;
        }
        var cardIndex = 0;
        var skipCount = 0;
        foreach (var stored in storedCandidates.EnumerateArray())
        {
            if (stored.ValueKind != JsonValueKind.Object
                || !TryReadRequiredString(
                    stored,
                    "kind",
                    out var kind
                ))
            {
                return false;
            }
            if (kind == "skip")
            {
                if (!StringPropertyEquals(
                        stored,
                        "candidate_id",
                        "skip"
                    ))
                {
                    return false;
                }
                skipCount++;
                continue;
            }
            if (kind != "card"
                || cardIndex >= options.Count
                || !stored.TryGetProperty(
                    "payload",
                    out var storedPayload
                )
                || storedPayload.ValueKind != JsonValueKind.Object)
            {
                return false;
            }
            var expected = options[cardIndex];
            if (!StringPropertyEquals(
                    stored,
                    "candidate_id",
                    expected.CandidateId
                )
                || !StringPropertyEquals(
                    storedPayload,
                    "card",
                    expected.Card
                )
                || !IntPropertyEquals(
                    storedPayload,
                    "upgrades",
                    expected.Upgrades
                )
                || !NullableStringPropertyEquals(
                    storedPayload,
                    "enchantment",
                    expected.Enchantment
                )
                || !NullableIntPropertyEquals(
                    storedPayload,
                    "enchantment_amount",
                    expected.EnchantmentAmount
                )
                || !NullableStringPropertyEquals(
                    storedPayload,
                    "affliction",
                    expected.Affliction
                )
                || !NullableIntPropertyEquals(
                    storedPayload,
                    "affliction_amount",
                    expected.AfflictionAmount
                ))
            {
                return false;
            }
            cardIndex++;
        }
        return cardIndex == options.Count
            && skipCount == (canSkip ? 1 : 0);
    }

    private static bool PayloadStateMatches(
        JsonElement payload,
        RunStateSnapshot current)
    {
        if (!payload.TryGetProperty("state", out var stored)
            || stored.ValueKind != JsonValueKind.Object
            || !StringPropertyEquals(
                stored,
                "character",
                current.Character
            )
            || !IntPropertyEquals(
                stored,
                "ascension",
                current.Ascension
            )
            || !IntPropertyEquals(stored, "act", current.Act)
            || !IntPropertyEquals(stored, "floor", current.Floor)
            || !NullableIntPropertyEquals(stored, "hp", current.Hp)
            || !NullableIntPropertyEquals(
                stored,
                "max_hp",
                current.MaxHp
            )
            || !NullableIntPropertyEquals(stored, "gold", current.Gold)
            || !IntPropertyEquals(stored, "energy", current.Energy)
            || !NullableIntPropertyEquals(
                stored,
                "max_potion_slots",
                current.MaxPotionSlots
            )
            || !DeckMatches(stored, current.Deck)
            || !StringArrayMatches(stored, "relics", current.Relics)
            || !RelicStatesMatch(stored, current.RelicStates)
            || !PotionStatesMatch(stored, current.Potions)
            || !StringArrayMatches(
                stored,
                "modifiers",
                current.Modifiers
            ))
        {
            return false;
        }
        return true;
    }

    private static bool DeckMatches(
        JsonElement state,
        IReadOnlyList<DeckCardState> expected)
    {
        if (!state.TryGetProperty("deck", out var stored)
            || stored.ValueKind != JsonValueKind.Array
            || stored.GetArrayLength() != expected.Count)
        {
            return false;
        }
        var index = 0;
        foreach (var card in stored.EnumerateArray())
        {
            var value = expected[index];
            if (card.ValueKind != JsonValueKind.Object
                || !StringPropertyEquals(card, "card", value.Card)
                || !IntPropertyEquals(card, "count", value.Count)
                || !IntPropertyEquals(card, "upgrades", value.Upgrades)
                || !NullableStringPropertyEquals(
                    card,
                    "enchantment",
                    value.Enchantment
                )
                || !NullableIntPropertyEquals(
                    card,
                    "enchantment_amount",
                    value.EnchantmentAmount
                )
                || !NullableStringPropertyEquals(
                    card,
                    "affliction",
                    value.Affliction
                )
                || !NullableIntPropertyEquals(
                    card,
                    "affliction_amount",
                    value.AfflictionAmount
                ))
            {
                return false;
            }
            index++;
        }
        return true;
    }

    private static bool RelicStatesMatch(
        JsonElement state,
        IReadOnlyList<RelicState> expected)
    {
        if (!state.TryGetProperty("relic_states", out var stored)
            || stored.ValueKind != JsonValueKind.Array
            || stored.GetArrayLength() != expected.Count)
        {
            return false;
        }
        var index = 0;
        foreach (var relic in stored.EnumerateArray())
        {
            var value = expected[index];
            if (relic.ValueKind != JsonValueKind.Object
                || !StringPropertyEquals(relic, "relic", value.Relic)
                || !NullableIntPropertyEquals(
                    relic,
                    "display_amount",
                    value.DisplayAmount
                )
                || !IntPropertyEquals(
                    relic,
                    "stack_count",
                    value.StackCount
                )
                || !NullableStringPropertyEquals(
                    relic,
                    "status",
                    value.Status
                ))
            {
                return false;
            }
            index++;
        }
        return true;
    }

    private static bool PotionStatesMatch(
        JsonElement state,
        IReadOnlyList<PotionState> expected)
    {
        if (!state.TryGetProperty("potions", out var stored)
            || stored.ValueKind != JsonValueKind.Array
            || stored.GetArrayLength() != expected.Count)
        {
            return false;
        }
        var index = 0;
        foreach (var potion in stored.EnumerateArray())
        {
            var value = expected[index];
            if (potion.ValueKind != JsonValueKind.Object
                || !StringPropertyEquals(
                    potion,
                    "potion",
                    value.Potion
                )
                || !IntPropertyEquals(potion, "slot", value.Slot))
            {
                return false;
            }
            index++;
        }
        return true;
    }

    private static bool StringArrayMatches(
        JsonElement parent,
        string property,
        IReadOnlyList<string> expected)
    {
        if (!parent.TryGetProperty(property, out var stored)
            || stored.ValueKind != JsonValueKind.Array
            || stored.GetArrayLength() != expected.Count)
        {
            return false;
        }
        var index = 0;
        foreach (var item in stored.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String
                || item.GetString() != expected[index])
            {
                return false;
            }
            index++;
        }
        return true;
    }

    private static bool HasLaterInvalidatingEvent(
        string directory,
        string decisionId,
        long openSequence)
    {
        var paths = new List<string>
        {
            Path.Combine(directory, "state-event.json"),
        };
        var eventDirectory = Path.Combine(directory, "events");
        if (Directory.Exists(eventDirectory))
        {
            paths.AddRange(Directory.EnumerateFiles(
                eventDirectory,
                "*.json"
            ));
        }
        foreach (var path in paths.Distinct())
        {
            if (!File.Exists(path))
            {
                continue;
            }
            try
            {
                using var document = JsonDocument.Parse(
                    File.ReadAllText(path)
                );
                var root = document.RootElement;
                if (!StringPropertyEquals(root, "run_id", _runId)
                    || !root.TryGetProperty("sequence", out var sequence)
                    || !sequence.TryGetInt64(out var value)
                    || value <= openSequence
                    || !TryReadRequiredString(
                        root,
                        "event_type",
                        out var eventType
                    ))
                {
                    continue;
                }
                if (eventType is "run_ended"
                        or "decision_closed"
                        or "merchant"
                        or "rest_site"
                    || ((eventType == "card_reward" || eventType == "route_choice")
                        && !StringPropertyEquals(
                            root,
                            "decision_id",
                            decisionId
                        )))
                {
                    return true;
                }
            }
            catch
            {
                // Queue files are atomically written. A malformed JSON file
                // means the checkpoint is not trustworthy enough to reuse.
                return true;
            }
        }
        return false;
    }

    private static bool TryReadRequiredString(
        JsonElement element,
        string property,
        out string value)
    {
        value = "";
        return element.TryGetProperty(property, out var child)
            && child.ValueKind == JsonValueKind.String
            && !string.IsNullOrWhiteSpace(value = child.GetString() ?? "");
    }

    private static bool StringPropertyEquals(
        JsonElement element,
        string property,
        string? expected)
    {
        return expected is not null
            && element.TryGetProperty(property, out var child)
            && child.ValueKind == JsonValueKind.String
            && child.GetString() == expected;
    }

    private static bool NullableStringPropertyEquals(
        JsonElement element,
        string property,
        string? expected)
    {
        if (!element.TryGetProperty(property, out var child))
        {
            return false;
        }
        return expected is null
            ? child.ValueKind == JsonValueKind.Null
            : child.ValueKind == JsonValueKind.String
                && child.GetString() == expected;
    }

    private static bool BoolPropertyEquals(
        JsonElement element,
        string property,
        bool expected)
    {
        return element.TryGetProperty(property, out var child)
            && (child.ValueKind == JsonValueKind.True
                || child.ValueKind == JsonValueKind.False)
            && child.GetBoolean() == expected;
    }

    private static bool IntPropertyEquals(
        JsonElement element,
        string property,
        int expected)
    {
        return element.TryGetProperty(property, out var child)
            && child.TryGetInt32(out var value)
            && value == expected;
    }

    private static bool NullableIntPropertyEquals(
        JsonElement element,
        string property,
        int? expected)
    {
        if (!element.TryGetProperty(property, out var child))
        {
            return false;
        }
        return expected is null
            ? child.ValueKind == JsonValueKind.Null
            : child.TryGetInt32(out var value) && value == expected.Value;
    }

    private static long ReadSequence(
        string path,
        string runId,
        string sequenceProperty
    )
    {
        if (!File.Exists(path))
        {
            return 0;
        }
        try
        {
            using var document = JsonDocument.Parse(
                File.ReadAllText(path)
            );
            var root = document.RootElement;
            if (!root.TryGetProperty("run_id", out var storedRunId)
                || storedRunId.GetString() != runId
                || !root.TryGetProperty(
                    sequenceProperty,
                    out var storedSequence
                ))
            {
                return 0;
            }
            return storedSequence.TryGetInt64(out var sequence)
                ? Math.Max(0, sequence)
                : 0;
        }
        catch
        {
            return 0;
        }
    }

    private static string? GetExchangeDirectory()
    {
        var outputPath = ProjectSettings.GlobalizePath(
            "user://STS2Guide/state-event.json"
        );
        return Path.GetDirectoryName(outputPath);
    }

    private static bool SameOption(
        DecisionOption first,
        DecisionOption second
    )
    {
        return first.Card == second.Card
            && first.Upgrades == second.Upgrades
            && first.Enchantment == second.Enchantment
            && first.EnchantmentAmount == second.EnchantmentAmount
            && first.Affliction == second.Affliction
            && first.AfflictionAmount == second.AfflictionAmount;
    }

    private static string CandidateId(int index, DecisionOption option)
    {
        return $"{index}:{option.Card}";
    }

    private sealed record PendingDecision(
        string ParentEventId,
        string DecisionId,
        long Sequence,
        string RouteMode,
        bool CanSkip,
        List<DecisionOption> Options,
        List<PendingDecisionCandidate> Candidates,
        string EventType,
        object? DecisionSource,
        RouteOpportunityIdentity? RouteOpportunity,
        MapChoiceContext? RouteContext,
        string? ObservationFingerprint
    )
    {
        internal object? ScreenOwner { get; set; }
    }

    private sealed record PendingDecisionCandidate(
        string CandidateId,
        string Label,
        int DisplayIndex,
        bool Eligible,
        string? Kind = null,
        CandidatePayload? Payload = null
    );

    private sealed record PendingChildParent(
        string RunId,
        string ExpectedChildKind,
        long ParentCloseSequence,
        DecisionParentContext Context
    );

    private sealed record RouteOpportunityIdentity(
        int Act,
        string OriginNodeId,
        string MapFingerprint
    );

    private sealed record RouteDecisionRecoveryCandidate(
        long Sequence,
        string DecisionId
    );
}

internal sealed record PendingDecisionView(
    string RunId,
    string EventId,
    string DecisionId,
    long Sequence,
    string RouteMode,
    bool CanSkip,
    string EventType,
    IReadOnlyList<PendingDecisionOptionView> Options,
    IReadOnlyList<PendingDecisionCandidateView> Candidates,
    MapChoiceContext? RouteContext
);

internal sealed record PendingDecisionOptionView(
    string CandidateId,
    string CardId,
    int Upgrades,
    string? Enchantment,
    int? EnchantmentAmount,
    string? Affliction,
    int? AfflictionAmount
);

internal sealed record PendingDecisionCandidateView(
    string CandidateId,
    string Label,
    int DisplayIndex,
    bool Eligible
);
