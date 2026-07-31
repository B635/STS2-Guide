using HarmonyLib;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Runs;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// Observes only a verified, currently actionable map decision.  The
/// selection callback is Postfix-only and is never invoked by this class.
/// </summary>
internal static class RouteChoiceObserver
{
    private static readonly object Gate = new();
    private static NMapScreen? _screen;
    private static bool _openedFromTopBar = true;
    private static long _lastAttemptTicks;
    private static string? _stagedActionableFingerprint;
    private static long _stagedAtTicks;
    private static string? _lastCommittedObservationFingerprint;

    internal static void OnOpened(NMapScreen screen, bool isOpenedFromTopBar)
        => Safely(screen, () =>
        {
            lock (Gate)
            {
                if (!ReferenceEquals(_screen, screen))
                {
                    // A replacement owner can arrive before every callback
                    // from the previous map screen has drained.  Remove its
                    // route UI now; subsequent stale callbacks must be a
                    // complete no-op and may not disturb the new owner's
                    // two-sample debounce state.
                    if (_screen is not null)
                    {
                        RouteAdviceController.Hide(_screen);
                    }
                    _stagedActionableFingerprint = null;
                    _lastCommittedObservationFingerprint = null;
                }
                _screen = screen;
                _openedFromTopBar = isOpenedFromTopBar;
                if (isOpenedFromTopBar)
                {
                    _stagedActionableFingerprint = null;
                }
            }
            TryObserve(screen);
        });

    internal static void TryObserve(NMapScreen screen)
        => Safely(screen, () => TryObserveSafe(screen));

    private static void TryObserveSafe(NMapScreen screen)
    {
        lock (Gate)
        {
            if (!ReleaseCapabilityGate.IsEnabled("route_choice"))
            {
                RouteAdviceController.Hide(screen);
                RouteMapOverlay.Hide(screen);
                return;
            }
            // Delayed _Process / SetTravelEnabled callbacks from a replaced
            // NMapScreen are observationally irrelevant.  In particular,
            // they must not clear the current owner's staged debounce.
            if (!ReferenceEquals(_screen, screen))
            {
                return;
            }
            // Open(bool) is the only verified source for the preview flag.
            // Do not infer it from IsTravelEnabled, which is correlated but
            // not an identity for the interaction.
            if (!ReferenceEquals(_screen, screen)
                || _openedFromTopBar
                || !GodotObject.IsInstanceValid(screen)
                || !screen.IsInsideTree()
                || !screen.IsOpen
                || !screen.IsTravelEnabled
                || screen.IsTraveling
                || screen.IsDebugTravelEnabled)
            {
                FailClosed(screen);
                return;
            }
            var now = DateTime.UtcNow.Ticks;
            if (now - _lastAttemptTicks < TimeSpan.TicksPerMillisecond * 250)
            {
                return;
            }
            _lastAttemptTicks = now;
            if (RunManager.Instance is null
                || !RunManager.Instance.IsSingleplayerOrFakeMultiplayer)
            {
                FailClosed(screen);
                return;
            }
            if (RunStateReader.GetObservedPlayer() is null
                && !RunStateReader.TryObserveFromRunManager())
            {
                FailClosed(screen);
                return;
            }
            if (!RunStateReader.TryCapture(out var state) || state is null)
            {
                FailClosed(screen);
                return;
            }
            var player = RunStateReader.GetObservedPlayer();
            if (player is null)
            {
                FailClosed(screen);
                return;
            }
            var snapshot = MapNodeReader.Read(player);
            if (snapshot.Nodes.Count == 0
                || string.IsNullOrWhiteSpace(snapshot.OriginNodeId)
                || string.IsNullOrWhiteSpace(snapshot.MapFingerprint)
                || snapshot.AvailableNextNodeIds.Count == 0)
            {
                FailClosed(screen);
                return;
            }
            if (!TryReadVisualTravelableIds(screen, out var visualCandidates))
            {
                FailClosed(screen);
                return;
            }
            var modelCandidates = snapshot.AvailableNextNodeIds
                .OrderBy(id => id, StringComparer.Ordinal)
                .ToList();
            if (modelCandidates.Distinct(StringComparer.Ordinal).Count()
                    != modelCandidates.Count
                || !modelCandidates.SequenceEqual(visualCandidates))
            {
                FailClosed(screen);
                return;
            }

            var actionableFingerprint = string.Join(
                "|",
                snapshot.OriginNodeId,
                snapshot.MapFingerprint,
                string.Join(",", modelCandidates)
            );
            // A complete predicate must hold in two independent samples.  It
            // filters scene construction and click/travel transitions where
            // model and visual state briefly disagree.
            if (_stagedActionableFingerprint != actionableFingerprint)
            {
                // A newly observed candidate/topology identity makes any
                // route advice for the previous one unsafe during the
                // two-sample debounce window.  It is less confusing to show
                // no route than to keep drawing a line for a stale choice.
                RouteAdviceController.Hide(screen);
                _stagedActionableFingerprint = actionableFingerprint;
                _stagedAtTicks = now;
                return;
            }
            if (now - _stagedAtTicks < TimeSpan.TicksPerMillisecond * 250)
            {
                return;
            }

            var pending = StateEventWriter.EmitRouteChoice(state, snapshot, screen);
            if (pending is null
                || pending.EventType != "route_choice"
                || !StateEventWriter.BindDecisionScreen(screen, pending.DecisionId))
            {
                // The writer intentionally did not commit (or its pending
                // handle could not be bound to this owner).  Never leave the
                // previous opportunity visible while retrying: the old
                // Drawer/Overlay is stale relative to this fully observed
                // actionable state. Hide is owner-scoped, so a delayed old
                // owner cannot affect the current screen.
                RouteAdviceController.Hide(screen);
                // Preserve no committed fingerprint so a later _Process
                // retries the same observation safely.
                return;
            }
            _lastCommittedObservationFingerprint = actionableFingerprint;
            RouteAdviceController.Show(screen, pending);
        }
    }

    internal static void OnSelected(NMapScreen screen, NMapPoint point)
        => Safely(screen, () =>
        {
            lock (Gate)
            {
                // A delayed callback from a disposed/replaced map owner must
                // not close or clear the current owner's route observation.
                if (!ReferenceEquals(_screen, screen))
                {
                    return;
                }
                // A native map-point selection is the verified transition
                // into a later room.  Any prior event child must have opened
                // before this action; clearing here cannot race a map screen
                // that was merely shown behind a reward overlay.
                EventDecisionObserver.OnNonEventDecisionBoundary();
                var nodeId = $"{point.Point.coord.row}:{point.Point.coord.col}";
                if (StateEventWriter.EmitRouteSelected(screen, nodeId))
                {
                    RouteAdviceController.Hide(screen);
                    ClearStagedObservation();
                    _lastCommittedObservationFingerprint = null;
                }
                else
                {
                    // Keep the writer pending decision intact for its own
                    // retry/recovery path, but never leave advice visible
                    // once the player has made the native selection.
                    FailClosed(screen);
                    _lastCommittedObservationFingerprint = null;
                }
            }
        });

    internal static void OnClosed(NMapScreen screen)
        => Safely(screen, () =>
        {
            lock (Gate)
            {
                RouteAdviceController.Hide(screen);
                if (ReferenceEquals(_screen, screen))
                {
                    _screen = null;
                    ClearStagedObservation();
                    _lastCommittedObservationFingerprint = null;
                }
            }
        });

    internal static void OnOwnerGone(NMapScreen screen)
        => Safely(screen, () =>
        {
            lock (Gate)
            {
                StateEventWriter.UnbindDecisionScreen(screen);
                RouteAdviceController.Hide(screen);
                if (ReferenceEquals(_screen, screen))
                {
                    _screen = null;
                    ClearStagedObservation();
                    _lastCommittedObservationFingerprint = null;
                }
            }
        });

    private static void ClearStagedObservation()
    {
        _stagedActionableFingerprint = null;
        _stagedAtTicks = 0;
    }

    // Call only while Gate is held.  It intentionally does not unbind or
    // mutate StateEventWriter's pending decision: persistence may retry or
    // recover independently, whereas the screen must immediately stop
    // displaying an unverified recommendation.
    private static void FailClosed(NMapScreen screen)
    {
        if (ReferenceEquals(_screen, screen))
        {
            ClearStagedObservation();
        }
        RouteAdviceController.Hide(screen);
    }

    private static bool TryReadVisualTravelableIds(
        NMapScreen screen,
        out List<string> candidates)
    {
        candidates = [];
        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var point in EnumerateDescendants(screen).OfType<NMapPoint>())
        {
            if (!GodotObject.IsInstanceValid(point)
                || point.State != MapPointState.Travelable)
            {
                continue;
            }
            var nodeId = $"{point.Point.coord.row}:{point.Point.coord.col}";
            if (!ids.Add(nodeId))
            {
                return false;
            }
        }
        candidates = ids.OrderBy(id => id, StringComparer.Ordinal).ToList();
        return candidates.Count > 0;
    }

    private static IEnumerable<Node> EnumerateDescendants(Node root)
    {
        foreach (var child in root.GetChildren(includeInternal: true))
        {
            yield return child;
            foreach (var descendant in EnumerateDescendants(child))
            {
                yield return descendant;
            }
        }
    }

    private static void Safely(NMapScreen screen, Action action)
    {
        try
        {
            action();
        }
        catch (Exception exception)
        {
            TryLogRouteFailure(
                "[STS2-Guide] Route observer failed closed: "
                + exception.Message
            );
            try
            {
                lock (Gate)
                {
                    // A callback from a discarded owner must never mutate
                    // the new screen's staging state, even on its exception
                    // path.
                    if (ReferenceEquals(_screen, screen))
                    {
                        ClearStagedObservation();
                    }
                }
            }
            catch
            {
                // Cleanup must not become another Postfix failure.
            }
            try
            {
                RouteAdviceController.Hide(screen);
            }
            catch (Exception cleanupException)
            {
                TryLogRouteFailure(
                    "[STS2-Guide] Route observer cleanup failed: "
                    + cleanupException.Message
                );
            }
        }
    }

    private static void TryLogRouteFailure(string message)
    {
        try
        {
            Log.Error(message);
        }
        catch
        {
            // Logging is diagnostic only and must not escape a Harmony
            // callback after the primary failure has already been contained.
        }
    }
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.Open))]
internal static class RouteMapOpenPatch
{
    [HarmonyPostfix]
    internal static void AfterOpen(NMapScreen __instance, bool isOpenedFromTopBar)
        => RouteChoiceObserver.OnOpened(__instance, isOpenedFromTopBar);
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.SetTravelEnabled))]
internal static class RouteMapTravelPatch
{
    [HarmonyPostfix]
    internal static void AfterSetTravelEnabled(NMapScreen __instance, bool enabled)
        => RouteChoiceObserver.TryObserve(__instance);
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen._Process))]
internal static class RouteMapProcessPatch
{
    [HarmonyPostfix]
    internal static void AfterProcess(NMapScreen __instance)
        => RouteChoiceObserver.TryObserve(__instance);
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.OnMapPointSelectedLocally))]
internal static class RouteMapSelectedPatch
{
    [HarmonyPostfix]
    internal static void AfterSelected(NMapScreen __instance, NMapPoint point)
        => RouteChoiceObserver.OnSelected(__instance, point);
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.Close))]
internal static class RouteMapClosePatch
{
    [HarmonyPostfix]
    internal static void AfterClose(NMapScreen __instance, bool animateOut)
        => RouteChoiceObserver.OnClosed(__instance);
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.CleanUp))]
internal static class RouteMapCleanupPatch
{
    [HarmonyPostfix]
    internal static void AfterCleanup(NMapScreen __instance)
        => RouteChoiceObserver.OnOwnerGone(__instance);
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen._ExitTree))]
internal static class RouteMapExitPatch
{
    [HarmonyPostfix]
    internal static void AfterExitTree(NMapScreen __instance)
        => RouteChoiceObserver.OnOwnerGone(__instance);
}
