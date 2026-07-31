using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Logging;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// One owner-scoped reader for every non-card, non-route v9 decision.
/// It never interprets game text and never calls a game action.
/// </summary>
internal static class GenericAdviceController
{
    private static readonly object Gate = new();
    private static Control? _owner;
    private static Godot.Timer? _timer;
    private static PendingDecisionView? _pending;
    private static ContextDrawerHandle _drawer;
    private static long _lastWriteTicks;

    internal static void Show(
        Control owner,
        PendingDecisionView pending,
        string title)
    {
        lock (Gate)
        {
            if (!ReleaseCapabilityGate.IsEnabled(pending.EventType)
                || !IsGenericEvent(pending.EventType)
                || pending.Candidates.Count == 0)
            {
                HideInternal();
                return;
            }
            if (ReferenceEquals(owner, _owner)
                && _pending is not null
                && CanReuseDrawer(_pending, pending)
                && ContextDrawer.IsVisible(_drawer))
            {
                _pending = pending;
                StateEventWriter.BindDecisionScreen(
                    owner,
                    pending.DecisionId
                );
                _lastWriteTicks = 0;
                Invalidate();
                return;
            }

            HideInternal();
            if (!StateEventWriter.BindDecisionScreen(
                owner,
                pending.DecisionId
            ))
            {
                return;
            }
            _owner = owner;
            _pending = pending;
            _drawer = ContextDrawer.Show(
                owner,
                null,
                title,
                pending.Candidates
                    .OrderBy(candidate => candidate.DisplayIndex)
                    .Select(candidate => candidate.Label)
                    .ToList()
            );
            if (!_drawer.IsValid)
            {
                HideInternal();
                return;
            }
            _timer = new Godot.Timer
            {
                WaitTime = 0.1,
                OneShot = false,
                Autostart = true,
            };
            _timer.Timeout += Poll;
            owner.AddChild(_timer);
            _lastWriteTicks = 0;
            Invalidate();
        }
    }

    internal static void Hide(Control? owner = null)
    {
        lock (Gate)
        {
            if (owner is not null && !ReferenceEquals(owner, _owner))
            {
                return;
            }
            HideInternal();
        }
    }

    private static bool CanReuseDrawer(
        PendingDecisionView current,
        PendingDecisionView next)
        => current.DecisionId == next.DecisionId
            && current.EventType == next.EventType
            && current.Candidates.Count == next.Candidates.Count
            && current.Candidates.Zip(next.Candidates).All(pair =>
                pair.First.CandidateId == pair.Second.CandidateId
                && pair.First.Label == pair.Second.Label
                && pair.First.DisplayIndex == pair.Second.DisplayIndex
                && pair.First.Eligible == pair.Second.Eligible
            );

    private static void HideInternal()
    {
        if (_owner is not null)
        {
            StateEventWriter.UnbindDecisionScreen(_owner);
        }
        if (_timer is not null && GodotObject.IsInstanceValid(_timer))
        {
            _timer.Stop();
            _timer.Timeout -= Poll;
            _timer.QueueFree();
        }
        ContextDrawer.Hide(_drawer);
        _owner = null;
        _timer = null;
        _pending = null;
        _drawer = default;
        _lastWriteTicks = 0;
    }

    private static void Poll()
    {
        lock (Gate)
        {
            if (_owner is null || _pending is null
                || !GodotObject.IsInstanceValid(_owner)
                || !ContextDrawer.IsVisible(_drawer))
            {
                HideInternal();
                return;
            }
            ContextDrawer.Reposition(_drawer);
            var path = ProjectSettings.GlobalizePath(
                "user://STS2Guide/advice-event.json"
            );
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
                using var document = JsonDocument.Parse(
                    File.ReadAllText(path)
                );
                var rows = ReadMatchingRows(
                    document.RootElement,
                    _pending
                );
                if (rows is null)
                {
                    _lastWriteTicks = ticks;
                    Invalidate();
                    return;
                }
                ContextDrawer.Render(_drawer, rows);
                _lastWriteTicks = ticks;
            }
            catch (Exception exception)
            {
                _lastWriteTicks = 0;
                Log.Error(
                    "[STS2-Guide] Generic advice read failed: "
                    + exception.Message
                );
                Invalidate();
            }
        }
    }

    private static IReadOnlyList<ContextDrawerRow>? ReadMatchingRows(
        JsonElement root,
        PendingDecisionView pending)
    {
        if (!AdviceCompatibilityReader.MatchesCurrentRuntime(root)
            || !StringEquals(root, "run_id", pending.RunId)
            || !StringEquals(root, "event_id", pending.EventId)
            || !StringEquals(root, "decision_id", pending.DecisionId)
            || !StringEquals(root, "event_type", pending.EventType)
            || !StringEquals(root, "status", "processed")
            || !IntEquals(root, "sequence", pending.Sequence)
            || !GuidePreferencesMatch(root, pending.RouteMode)
            || !root.TryGetProperty(
                "advice_disposition",
                out var disposition
            )
            || !StringEquals(disposition, "action", "publish")
            || !StringEquals(disposition, "run_id", pending.RunId)
            || !StringEquals(
                disposition,
                "decision_id",
                pending.DecisionId
            )
            || !root.TryGetProperty(
                "recommendation",
                out var recommendation
            )
            || recommendation.ValueKind != JsonValueKind.Object
            || !AdviceContractReader.HasRecommendationMetadata(
                recommendation
            )
            || !IntEquals(recommendation, "contract_version", 1)
            || !StringEquals(
                recommendation,
                "decision_id",
                pending.DecisionId
            )
            || !StringEquals(
                recommendation,
                "decision_type",
                DecisionType(pending.EventType)
            )
            || !IntEquals(
                recommendation,
                "world_sequence",
                pending.Sequence
            )
            || !recommendation.TryGetProperty(
                "candidates",
                out var candidates
            )
            || candidates.ValueKind != JsonValueKind.Array)
        {
            return null;
        }
        var expected = pending.Candidates
            .OrderBy(candidate => candidate.DisplayIndex)
            .ToList();
        if (candidates.GetArrayLength() != expected.Count
            || !TryNullableString(
                recommendation,
                "recommended_candidate_id",
                out var recommended
            )
            || !AdviceContractReader.StatusMatchesRecommendation(
                recommendation,
                recommended
            ))
        {
            return null;
        }
        var rows = new ContextDrawerRow[expected.Count];
        var matched = new bool[expected.Count];
        foreach (var candidate in candidates.EnumerateArray())
        {
            if (!StringProperty(
                    candidate,
                    "candidate_id",
                    out var candidateId
                )
                || !candidate.TryGetProperty(
                    "display_index",
                    out var display
                )
                || !display.TryGetInt32(out var index)
                || index < 0
                || index >= expected.Count
                || matched[index]
                || expected[index].CandidateId != candidateId
                || !BoolEquals(
                    candidate,
                    "eligible",
                    expected[index].Eligible
                )
                || !AdviceContractReader.HasCandidateMetadata(candidate))
            {
                return null;
            }
            double? score = null;
            if (!candidate.TryGetProperty("score", out var rawScore))
            {
                return null;
            }
            if (rawScore.ValueKind == JsonValueKind.Number
                && rawScore.TryGetDouble(out var numeric)
                && double.IsFinite(numeric)
                && numeric >= 0
                && numeric <= 100)
            {
                score = numeric;
            }
            else if (rawScore.ValueKind != JsonValueKind.Null)
            {
                return null;
            }
            var isRecommended = candidateId == recommended;
            if (isRecommended
                && (!expected[index].Eligible || !score.HasValue))
            {
                return null;
            }
            rows[index] = new ContextDrawerRow(
                expected[index].Label,
                score,
                isRecommended,
                ReadReasons(candidate)
            );
            matched[index] = true;
        }
        if (matched.Any(value => !value)
            || (
                recommended is not null
                && !expected.Any(candidate =>
                    candidate.CandidateId == recommended
                    && candidate.Eligible
                )
            ))
        {
            return null;
        }
        return rows;
    }

    private static IReadOnlyList<string> ReadReasons(
        JsonElement candidate)
    {
        if (!candidate.TryGetProperty("factors", out var factors)
            || factors.ValueKind != JsonValueKind.Array)
        {
            return [];
        }
        var reasons = new List<(double Weight, int Index, string Message)>();
        var index = 0;
        foreach (var factor in factors.EnumerateArray())
        {
            if (factor.ValueKind == JsonValueKind.Object
                && StringProperty(factor, "message", out var message)
                && factor.TryGetProperty("delta", out var delta)
                && delta.TryGetDouble(out var numeric)
                && double.IsFinite(numeric))
            {
                reasons.Add((Math.Abs(numeric), index, message));
            }
            index++;
        }
        return reasons
            .OrderByDescending(reason => reason.Weight)
            .ThenBy(reason => reason.Index)
            .Take(3)
            .Select(reason => reason.Message)
            .ToList();
    }

    private static void Invalidate()
    {
        if (_pending is null)
        {
            return;
        }
        ContextDrawer.Render(
            _drawer,
            _pending.Candidates
                .OrderBy(candidate => candidate.DisplayIndex)
                .Select(candidate => new ContextDrawerRow(
                    candidate.Label,
                    null,
                    false
                ))
                .ToList()
        );
    }

    private static bool IsGenericEvent(string eventType)
        => eventType is (
            "merchant"
            or "rest_site"
            or "neow_choice"
            or "event_choice"
            or "deck_edit"
        );

    private static string DecisionType(string eventType)
        => eventType switch
        {
            "merchant" => "merchant_choice",
            "rest_site" => "campfire_action",
            "neow_choice" => "neow_blessing",
            "event_choice" => "event_option",
            "deck_edit" => "deck_edit",
            _ => "",
        };

    private static bool GuidePreferencesMatch(
        JsonElement root,
        string routeMode)
        => root.TryGetProperty(
                "guide_preferences",
                out var preferences
            )
            && preferences.ValueKind == JsonValueKind.Object
            && StringEquals(
                preferences,
                "route_mode",
                routeMode
            );

    private static bool TryNullableString(
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
        value = element.GetString();
        return true;
    }

    private static bool StringProperty(
        JsonElement root,
        string property,
        out string value)
    {
        value = "";
        if (!root.TryGetProperty(property, out var element)
            || element.ValueKind != JsonValueKind.String
            || string.IsNullOrWhiteSpace(element.GetString()))
        {
            return false;
        }
        value = element.GetString()!;
        return true;
    }

    private static bool StringEquals(
        JsonElement root,
        string property,
        string expected)
        => root.TryGetProperty(property, out var value)
            && value.ValueKind == JsonValueKind.String
            && value.GetString() == expected;

    private static bool IntEquals(
        JsonElement root,
        string property,
        long expected)
        => root.TryGetProperty(property, out var value)
            && value.TryGetInt64(out var actual)
            && actual == expected;

    private static bool BoolEquals(
        JsonElement root,
        string property,
        bool expected)
        => root.TryGetProperty(property, out var value)
            && (
                expected
                    ? value.ValueKind == JsonValueKind.True
                    : value.ValueKind == JsonValueKind.False
            );
}
