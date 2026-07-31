using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// Strict route adapter for the shared drawer.  It owns no lifecycle state:
/// the writer supplies one immutable pending view after a successful commit.
/// </summary>
internal static class RouteAdviceController
{
    private static readonly object Gate = new();
    private static NMapScreen? _screen;
    private static Godot.Timer? _timer;
    private static PendingDecisionView? _pending;
    private static ContextDrawerHandle _drawer;
    private static long _lastWriteTicks;

    internal static void Show(NMapScreen screen, PendingDecisionView pending)
    {
        lock (Gate)
        {
            if (!ReleaseCapabilityGate.IsEnabled("route_choice")
                || pending.EventType != "route_choice"
                || pending.Candidates.Count == 0
                || pending.RouteContext is null)
            {
                HideInternal();
                return;
            }
            if (ReferenceEquals(_screen, screen)
                && _drawer.IsValid
                && _pending?.EventId == pending.EventId)
            {
                return;
            }
            if (ReferenceEquals(_screen, screen)
                && _drawer.IsValid
                && _pending is not null
                && CanReuseDrawer(_pending, pending))
            {
                _pending = pending;
                ContextDrawer.SetRouteMode(
                    _drawer,
                    pending.RouteMode
                );
                _lastWriteTicks = 0;
                Invalidate();
                return;
            }
            HideInternal();
            _pending = pending;
            _screen = screen;
            var captions = pending.Candidates.Select(candidate => candidate.Label).ToList();
            _drawer = ContextDrawer.Show(
                screen,
                null,
                "路线建议",
                captions,
                pending.RouteMode,
                OnRouteModeRequested
            );
            if (!_drawer.IsValid)
            {
                HideInternal();
                return;
            }
            _timer = new Godot.Timer { WaitTime = 0.1, OneShot = false, Autostart = true };
            _timer.Timeout += Poll;
            screen.AddChild(_timer);
            _lastWriteTicks = 0;
            Invalidate();
        }
    }

    private static bool CanReuseDrawer(
        PendingDecisionView current,
        PendingDecisionView next)
    {
        // ContextDrawer builds one immutable row Control per candidate.  A
        // route UPDATED may change the canonical candidate set, its ordering
        // or its captions; any such shape change must rebuild the drawer.
        // A different decision ID must also hide the old owner-scoped overlay
        // before a new one is ever rendered.
        return current.DecisionId == next.DecisionId
            && current.Candidates.Count == next.Candidates.Count
            && current.Candidates.Zip(next.Candidates).All(pair =>
                pair.First.CandidateId == pair.Second.CandidateId
                && pair.First.DisplayIndex == pair.Second.DisplayIndex
                && pair.First.Label == pair.Second.Label
            );
    }

    internal static void Hide(NMapScreen? owner = null)
    {
        lock (Gate)
        {
            if (owner is not null && !ReferenceEquals(owner, _screen))
            {
                return;
            }
            HideInternal();
        }
    }

    private static void HideInternal()
    {
        if (_timer is not null && GodotObject.IsInstanceValid(_timer))
        {
            _timer.Stop();
            _timer.Timeout -= Poll;
            _timer.QueueFree();
        }
        ContextDrawer.Hide(_drawer);
        RouteMapOverlay.Hide(_screen, _pending?.DecisionId);
        _screen = null;
        _timer = null;
        _pending = null;
        _drawer = default;
        _lastWriteTicks = 0;
    }

    private static void Poll()
    {
        lock (Gate)
        {
            if (_screen is null || _pending is null
                || !GodotObject.IsInstanceValid(_screen)
                || !ContextDrawer.IsVisible(_drawer))
            {
                HideInternal();
                return;
            }
            ContextDrawer.Reposition(_drawer);
            var path = ProjectSettings.GlobalizePath("user://STS2Guide/advice-event.json");
            if (!File.Exists(path))
            {
                _lastWriteTicks = 0;
                Invalidate();
                return;
            }
            long ticks;
            try
            {
                ticks = File.GetLastWriteTimeUtc(path).Ticks;
            }
            catch
            {
                _lastWriteTicks = 0;
                Invalidate();
                return;
            }
            if (ticks == _lastWriteTicks)
            {
                return;
            }
            try
            {
                using var document = JsonDocument.Parse(File.ReadAllText(path));
                var matched = ReadMatching(document.RootElement, _pending);
                if (matched is null)
                {
                    _lastWriteTicks = ticks;
                    Invalidate();
                    return;
                }
                ContextDrawer.Render(_drawer, matched.Rows);
                RouteMapOverlay.Render(
                    _screen,
                    _pending.DecisionId,
                    matched.Presentation,
                    _pending.RouteContext!
                );
                _lastWriteTicks = ticks;
            }
            catch (Exception exception)
            {
                _lastWriteTicks = 0;
                Log.Error("[STS2-Guide] Route advice read failed: " + exception.Message);
                Invalidate();
            }
        }
    }

    private static bool OnRouteModeRequested(string routeMode)
    {
        lock (Gate)
        {
            if (_screen is null
                || _pending is null
                || !GodotObject.IsInstanceValid(_screen)
                || !ContextDrawer.IsVisible(_drawer))
            {
                return false;
            }
            var updated = StateEventWriter.EmitRouteModeUpdated(
                _screen,
                _pending.DecisionId,
                routeMode
            );
            if (updated is null)
            {
                return false;
            }
            _pending = updated;
            ContextDrawer.SetRouteMode(_drawer, updated.RouteMode);
            _lastWriteTicks = 0;
            Invalidate();
            return true;
        }
    }

    private static RouteAdvice? ReadMatching(
        JsonElement root,
        PendingDecisionView pending)
    {
        if (!AdviceCompatibilityReader.MatchesCurrentRuntime(root)
            || !GuidePreferencesMatch(root, pending.RouteMode)
            || !StringEquals(root, "run_id", pending.RunId)
            || !StringEquals(root, "event_id", pending.EventId)
            || !StringEquals(root, "decision_id", pending.DecisionId)
            || !StringEquals(root, "event_type", "route_choice")
            || !StringEquals(root, "status", "processed")
            || !IntEquals(root, "sequence", pending.Sequence)
            || !root.TryGetProperty("advice_disposition", out var disposition)
            || !StringEquals(disposition, "action", "publish")
            || !StringEquals(disposition, "run_id", pending.RunId)
            || !StringEquals(disposition, "decision_id", pending.DecisionId)
            || !root.TryGetProperty("recommendation", out var recommendation)
            || recommendation.ValueKind != JsonValueKind.Object
            || !AdviceContractReader.HasRecommendationMetadata(
                recommendation
            )
            || !IntEquals(recommendation, "contract_version", 2)
            || !StringEquals(recommendation, "decision_id", pending.DecisionId)
            || !StringEquals(recommendation, "decision_type", "route_choice")
            || !IntEquals(recommendation, "world_sequence", pending.Sequence)
            || !recommendation.TryGetProperty("candidates", out var candidates)
            || candidates.ValueKind != JsonValueKind.Array
            || !recommendation.TryGetProperty("presentation", out var rawPresentation)
            || rawPresentation.ValueKind != JsonValueKind.Object)
        {
            return null;
        }
        var rows = ReadRows(candidates, recommendation, pending);
        var presentation = ReadPresentation(rawPresentation, recommendation, pending);
        return rows is null || presentation is null
            ? null
            : new RouteAdvice(rows, presentation);
    }

    private static IReadOnlyList<ContextDrawerRow>? ReadRows(
        JsonElement candidates,
        JsonElement recommendation,
        PendingDecisionView pending)
    {
        var expected = pending.Candidates
            .OrderBy(candidate => candidate.DisplayIndex)
            .ToList();
        if (candidates.GetArrayLength() != expected.Count)
        {
            return null;
        }
        if (!TryReadNullableString(
                recommendation,
                "recommended_candidate_id",
                out var recommended)
            || !AdviceContractReader.StatusMatchesRecommendation(
                recommendation,
                recommended
            ))
        {
            return null;
        }
        if (recommended is not null && !expected.Any(candidate => candidate.CandidateId == recommended))
        {
            return null;
        }
        var rows = new ContextDrawerRow[expected.Count];
        var matched = new bool[expected.Count];
        var recommendedMatched = recommended is null;
        foreach (var candidate in candidates.EnumerateArray())
        {
            if (!StringProperty(candidate, "candidate_id", out var candidateId)
                || !StringProperty(candidate, "label", out var label)
                || !candidate.TryGetProperty("display_index", out var indexElement)
                || !indexElement.TryGetInt32(out var index)
                || index < 0 || index >= expected.Count
                || matched[index]
                || expected[index].CandidateId != candidateId
                || !AdviceContractReader.HasCandidateMetadata(candidate)
                || !BoolEquals(candidate, "eligible", true))
            {
                return null;
            }
            double? score = null;
            if (candidate.TryGetProperty("score", out var scoreElement))
            {
                if (scoreElement.ValueKind == JsonValueKind.Number
                    && scoreElement.TryGetDouble(out var numeric)
                    && double.IsFinite(numeric)
                    && numeric >= 0 && numeric <= 100)
                {
                    score = numeric;
                }
                else if (scoreElement.ValueKind != JsonValueKind.Null)
                {
                    return null;
                }
            }
            else
            {
                return null;
            }
            var isRecommended = candidateId == recommended;
            if (isRecommended)
            {
                if (!score.HasValue)
                {
                    return null;
                }
                recommendedMatched = true;
            }
            rows[index] = new ContextDrawerRow(
                label,
                score,
                isRecommended
            );
            matched[index] = true;
        }
        return matched.All(value => value) && recommendedMatched ? rows : null;
    }

    private static RoutePresentation? ReadPresentation(
        JsonElement presentation,
        JsonElement recommendation,
        PendingDecisionView pending)
    {
        var context = pending.RouteContext;
        if (context is null
            || !StringEquals(presentation, "kind", "route_paths")
            || !StringEquals(presentation, "origin_node_id", context.OriginNodeId)
            || !TryReadNodeIds(presentation, "primary_path_node_ids", out var primary)
            || !TryReadNodeIds(presentation, "backup_path_node_ids", out var backup)
            || !presentation.TryGetProperty("paths", out var paths)
            || paths.ValueKind != JsonValueKind.Array)
        {
            return null;
        }
        var expected = pending.Candidates
            .OrderBy(candidate => candidate.DisplayIndex)
            .Select(candidate => candidate.CandidateId)
            .ToList();
        var knownNodes = context.Nodes.ToDictionary(node => node.NodeId, StringComparer.Ordinal);
        if (knownNodes.Count != context.Nodes.Count
            || !knownNodes.ContainsKey(context.OriginNodeId ?? "")
            || !expected.SequenceEqual(context.AvailableNextNodeIds))
        {
            return null;
        }
        var pathByCandidate = new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal);
        foreach (var row in paths.EnumerateArray())
        {
            if (!StringProperty(row, "candidate_id", out var candidateId)
                || !TryReadNodeIds(row, "node_ids", out var nodeIds)
                || !row.TryGetProperty("score", out var score)
                || (score.ValueKind != JsonValueKind.Null
                    && (!score.TryGetDouble(out var numeric)
                        || !double.IsFinite(numeric)
                        || numeric < 0
                        || numeric > 100))
                || pathByCandidate.ContainsKey(candidateId)
                || !expected.Contains(candidateId)
                || !ValidatePath(candidateId, nodeIds, knownNodes, context.BossNodeIds))
            {
                return null;
            }
            pathByCandidate[candidateId] = nodeIds;
        }
        if (!TryReadNullableString(
                recommendation,
                "recommended_candidate_id",
                out var recommended))
        {
            return null;
        }
        if (recommended is null)
        {
            return primary.Count == 0 && backup.Count == 0 && pathByCandidate.Count == 0
                ? new RoutePresentation(null, primary)
                : null;
        }
        if (!expected.Contains(recommended)
            || primary.Count == 0
            || primary[0] != recommended
            || !pathByCandidate.TryGetValue(recommended, out var expectedPrimary)
            || !primary.SequenceEqual(expectedPrimary)
            || !knownNodes[context.OriginNodeId!].Edges.Contains(
                recommended,
                StringComparer.Ordinal
            )
            || pathByCandidate.Count != expected.Count
            || !expected.All(pathByCandidate.ContainsKey))
        {
            return null;
        }
        if (backup.Count > 0)
        {
            return null;
        }
        return new RoutePresentation(recommended, primary);
    }

    private static bool GuidePreferencesMatch(
        JsonElement root,
        string expectedMode)
        => root.TryGetProperty(
                "guide_preferences",
                out var preferences
            )
            && preferences.ValueKind == JsonValueKind.Object
            && preferences.EnumerateObject().Count() == 1
            && StringEquals(
                preferences,
                "route_mode",
                expectedMode
            );

    private static bool ValidatePath(
        string candidateId,
        IReadOnlyList<string> nodeIds,
        IReadOnlyDictionary<string, MapNodeState> nodes,
        IReadOnlyList<string> bossIds)
    {
        if (nodeIds.Count == 0 || nodeIds[0] != candidateId)
        {
            return false;
        }
        for (var index = 0; index < nodeIds.Count; index++)
        {
            if (!nodes.TryGetValue(nodeIds[index], out var node))
            {
                return false;
            }
            if (index + 1 < nodeIds.Count
                && !node.Edges.Contains(nodeIds[index + 1], StringComparer.Ordinal))
            {
                return false;
            }
        }
        var terminal = nodeIds[^1];
        return bossIds.Contains(terminal, StringComparer.Ordinal)
            && nodes[terminal].Kind == "BOSS";
    }

    private static bool TryReadNodeIds(
        JsonElement parent,
        string property,
        out IReadOnlyList<string> ids)
    {
        ids = [];
        if (!parent.TryGetProperty(property, out var value)
            || value.ValueKind != JsonValueKind.Array)
        {
            return false;
        }
        var parsed = new List<string>();
        foreach (var item in value.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String
                || string.IsNullOrWhiteSpace(item.GetString()))
            {
                return false;
            }
            parsed.Add(item.GetString()!);
        }
        ids = parsed;
        return true;
    }

    private static void Invalidate()
    {
        if (_pending is null || !_drawer.IsValid)
        {
            return;
        }
        ContextDrawer.Render(_drawer, _pending.Candidates.Select(candidate =>
            new ContextDrawerRow(candidate.Label, null, false)).ToList());
        RouteMapOverlay.Hide(_screen, _pending.DecisionId);
    }

    private static bool StringEquals(JsonElement root, string property, string? expected)
        => expected is not null
            && root.TryGetProperty(property, out var value)
            && value.ValueKind == JsonValueKind.String
            && value.GetString() == expected;

    private static bool TryReadNullableString(
        JsonElement root,
        string property,
        out string? value)
    {
        value = null;
        if (!root.TryGetProperty(property, out var element))
        {
            return false;
        }
        if (element.ValueKind == JsonValueKind.Null)
        {
            return true;
        }
        if (element.ValueKind != JsonValueKind.String
            || string.IsNullOrWhiteSpace(element.GetString()))
        {
            return false;
        }
        // An omitted or wrong-typed recommended_candidate_id is not an
        // uncertain route result and must clear the current presentation.
        value = element.GetString();
        return true;
    }

    private static bool StringProperty(JsonElement root, string property, out string value)
    {
        value = "";
        if (!root.TryGetProperty(property, out var child)
            || child.ValueKind != JsonValueKind.String
            || string.IsNullOrWhiteSpace(child.GetString()))
        {
            return false;
        }
        value = child.GetString()!;
        return true;
    }

    private static bool IntEquals(JsonElement root, string property, long expected)
        => root.TryGetProperty(property, out var value)
            && value.TryGetInt64(out var actual)
            && actual == expected;

    private static bool BoolEquals(JsonElement root, string property, bool expected)
        => root.TryGetProperty(property, out var value)
            && (value.ValueKind == JsonValueKind.True || value.ValueKind == JsonValueKind.False)
            && value.GetBoolean() == expected;

    private sealed record RouteAdvice(
        IReadOnlyList<ContextDrawerRow> Rows,
        RoutePresentation Presentation
    );
}

internal sealed record RoutePresentation(
    string? RecommendedCandidateId,
    IReadOnlyList<string> PrimaryPathNodeIds
);
