using System.Reflection;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// Card Reward adapter for the canonical recommendation envelope.
/// </summary>
internal static class CardRewardAdvicePanel
{
    private static readonly object Gate = new();
    private static readonly FieldInfo? CardRowField =
        typeof(NCardRewardSelectionScreen).GetField(
            "_cardRow",
            BindingFlags.NonPublic | BindingFlags.Instance
        );

    private static NCardRewardSelectionScreen? _screen;
    private static Godot.Timer? _timer;
    private static PendingDecisionView? _pending;
    private static List<string> _captions = [];
    private static ContextDrawerHandle _drawerHandle;
    private static long _lastWriteTicks;
    private static readonly BackendLiveness Backend = new();
    private static SpecialRewardExpectation? _specialExpectation;

    internal static void PrepareSpecialReward(
        string capability,
        string sourceType)
    {
        lock (Gate)
        {
            _specialExpectation = new SpecialRewardExpectation(
                capability,
                sourceType
            );
        }
    }

    internal static void ClearSpecialRewardExpectation()
    {
        lock (Gate)
        {
            _specialExpectation = null;
        }
    }

    internal static void Show(NCardRewardSelectionScreen screen)
        => ShowCore(screen, allowDeferredSpecialRetry: true);

    private static void ShowCore(
        NCardRewardSelectionScreen screen,
        bool allowDeferredSpecialRetry)
    {
        lock (Gate)
        {
            HideInternal();
            if (!GodotObject.IsInstanceValid(screen)
                || !ReleaseCapabilityGate.IsEnabled("card_reward"))
            {
                return;
            }
            var special = _specialExpectation;
            if (special is not null
                && (string.IsNullOrWhiteSpace(special.Capability)
                    || string.IsNullOrWhiteSpace(special.SourceType)
                    || !ReleaseCapabilityGate.IsEnabled(
                        special.Capability
                    )
                    || !StateEventWriter.HasPendingCardRewardParent(
                        special.SourceType
                    )))
            {
                if (allowDeferredSpecialRetry
                    && ReleaseCapabilityGate.IsEnabled(
                        special.Capability
                    ))
                {
                    Callable.From(() => ObserverSafety.Run(
                        "card_reward.deferred_panel",
                        () => ShowCore(
                            screen,
                            allowDeferredSpecialRetry: false
                        )
                    )).CallDeferred();
                }
                else
                {
                    _specialExpectation = null;
                }
                return;
            }
            _specialExpectation = null;
            try
            {
                _pending = StateEventWriter.GetPendingDecision();
                if (_pending is null || _pending.Options.Count == 0)
                {
                    Log.Info(
                        "[STS2-Guide] Advice panel skipped: no pending card options."
                    );
                    return;
                }
                if (!PendingMatchesScreen(screen, _pending, out var captions))
                {
                    Log.Info(
                        "[STS2-Guide] Advice panel skipped: pending decision "
                        + "does not match visible screen candidates."
                    );
                    _pending = null;
                    return;
                }
                if (!StateEventWriter.BindCardRewardScreen(
                    screen,
                    _pending.DecisionId
                ))
                {
                    Log.Info(
                        "[STS2-Guide] Advice panel skipped: pending "
                        + "decision changed before screen binding."
                    );
                    _pending = null;
                    return;
                }

                _pending = StateEventWriter.RefreshCardRewardScreen(
                    screen,
                    _pending.DecisionId
                );
                if (_pending is null)
                {
                    StateEventWriter.UnbindCardRewardScreen(screen);
                    Log.Info(
                        "[STS2-Guide] Advice panel skipped: fresh card "
                        + "snapshot could not be committed."
                    );
                    return;
                }

                _screen = screen;
                if (_pending.CanSkip)
                {
                    captions.Add("跳过");
                }
                _captions = captions;
                _drawerHandle = ContextDrawer.Show(
                    screen,
                    CardRowField?.GetValue(screen) as Control,
                    "选牌建议",
                    captions
                );
                if (!_drawerHandle.IsValid)
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
                _timer.Timeout += PollAdvice;
                screen.AddChild(_timer);
                _lastWriteTicks = 0;
                Invalidate();
            }
            catch (Exception exception)
            {
                Log.Error(
                    "[STS2-Guide] Advice panel show failed: "
                    + exception.Message
                );
                HideInternal();
            }
        }
    }

    internal static void Hide(NCardRewardSelectionScreen? owner = null)
    {
        lock (Gate)
        {
            if (owner is not null && !ReferenceEquals(owner, _screen))
            {
                return;
            }
            HideInternal();
            _specialExpectation = null;
        }
    }

    private static void HideInternal()
    {
        if (_timer is not null && GodotObject.IsInstanceValid(_timer))
        {
            _timer.Stop();
            _timer.Timeout -= PollAdvice;
            _timer.QueueFree();
        }
        ContextDrawer.Hide(_drawerHandle);
        _screen = null;
        _timer = null;
        _pending = null;
        _captions = [];
        _drawerHandle = default;
        _lastWriteTicks = 0;
    }

    private sealed record SpecialRewardExpectation(
        string Capability,
        string SourceType
    );

    private static void PollAdvice()
    {
        lock (Gate)
        {
            if (_screen is null
                || _pending is null
                || !GodotObject.IsInstanceValid(_screen)
                || !ContextDrawer.IsVisible(_drawerHandle))
            {
                HideInternal();
                return;
            }
            ContextDrawer.Reposition(_drawerHandle);
            var backendStatus = Backend.Read(ProjectSettings.GlobalizePath(
                "user://STS2Guide/runtime-status.json"), StateEvent.ReleaseFingerprint);
            if (backendStatus != AdviceAvailability.Ready)
            {
                _lastWriteTicks = 0;
                Invalidate(backendStatus);
                return;
            }
            var advicePath = ProjectSettings.GlobalizePath(
                "user://STS2Guide/advice-event.json"
            );
            if (!File.Exists(advicePath))
            {
                _lastWriteTicks = 0;
                Invalidate();
                return;
            }
            long writeTicks;
            try
            {
                writeTicks = File.GetLastWriteTimeUtc(advicePath).Ticks;
            }
            catch
            {
                _lastWriteTicks = 0;
                Invalidate(AdviceAvailability.Unavailable);
                return;
            }
            if (writeTicks == _lastWriteTicks)
            {
                return;
            }
            try
            {
                var result = AdviceFileReader.Read(advicePath, root => ReadMatchingRows(root, _pending));
                var rows = result.Value;
                if (rows is null)
                {
                    _lastWriteTicks = 0; // transient read failures must retry even with unchanged mtime
                    Invalidate(result.Availability);
                    return;
                }
                ContextDrawer.Render(_drawerHandle, rows, result.Availability);
                _lastWriteTicks = writeTicks;
            }
            catch (Exception exception)
            {
                _lastWriteTicks = 0;
                Log.Error(
                    "[STS2-Guide] Advice panel read failed: "
                    + exception.Message
                );
                Invalidate(AdviceAvailability.Unavailable);
            }
        }
    }

    private static IReadOnlyList<ContextDrawerRow>? ReadMatchingRows(
        JsonElement root,
        PendingDecisionView pending)
    {
        if (!AdviceCompatibilityReader.MatchesCurrentRuntime(root)
            || !GuidePreferencesMatch(root, pending.RouteMode)
            || !TryReadString(root, "run_id", out var runId)
            || !TryReadString(root, "event_id", out var eventId)
            || !TryReadString(root, "decision_id", out var rootDecisionId)
            || runId != pending.RunId
            || eventId != pending.EventId
            || rootDecisionId != pending.DecisionId
            || !root.TryGetProperty("sequence", out var sequenceElement)
            || !sequenceElement.TryGetInt64(out var sequence)
            || sequence != pending.Sequence
            || !TryReadString(root, "event_type", out var eventType)
            || eventType != "card_reward"
            || !TryReadString(root, "status", out var status)
            || status != "processed"
            || !root.TryGetProperty(
                "advice_disposition",
                out var disposition
            )
            || disposition.ValueKind != JsonValueKind.Object
            || !TryReadString(disposition, "action", out var action)
            || action != "publish"
            || !TryReadString(
                disposition,
                "run_id",
                out var dispositionRunId
            )
            || dispositionRunId != pending.RunId
            || !TryReadString(
                disposition,
                "decision_id",
                out var dispositionDecisionId
            )
            || dispositionDecisionId != pending.DecisionId
            || !root.TryGetProperty("recommendation", out var recommendation)
            || recommendation.ValueKind != JsonValueKind.Object
            || !AdviceContractReader.HasRecommendationMetadata(
                recommendation
            )
            || !recommendation.TryGetProperty(
                "contract_version",
                out var contractVersion
            )
            || !contractVersion.TryGetInt32(out var version)
            || (version != 1 && version != 2)
            || !TryReadString(
                recommendation,
                "decision_id",
                out var recommendationDecisionId
            )
            || recommendationDecisionId != pending.DecisionId
            || !TryReadString(
                recommendation,
                "decision_type",
                out var decisionType
            )
            || decisionType != "card_reward"
            || !recommendation.TryGetProperty(
                "world_sequence",
                out var worldSequenceElement
            )
            || !worldSequenceElement.TryGetInt64(out var worldSequence)
            || worldSequence != pending.Sequence
            || !recommendation.TryGetProperty("candidates", out var candidates)
            || candidates.ValueKind != JsonValueKind.Array)
        {
            return null;
        }

        var expected = pending.Options
            .Select(option => option.CandidateId)
            .ToList();
        if (pending.CanSkip)
        {
            expected.Add("skip");
        }
        if (candidates.GetArrayLength() != expected.Count)
        {
            return null;
        }
        if (!TryReadNullableString(
            recommendation,
            "recommended_candidate_id",
            out var recommendedId
        ) || !AdviceContractReader.StatusMatchesRecommendation(
            recommendation,
            recommendedId
        ) || (recommendedId is not null
            && !expected.Contains(recommendedId)))
        {
            return null;
        }
        var rows = new ContextDrawerRow[expected.Count];
        var matched = new bool[expected.Count];
        var recommendedMatched = recommendedId is null;
        foreach (var candidate in candidates.EnumerateArray())
        {
            if (!TryReadString(candidate, "candidate_id", out var candidateId)
                || !TryReadString(candidate, "label", out var label)
                || !candidate.TryGetProperty(
                    "display_index",
                    out var indexElement
                )
                || !indexElement.TryGetInt32(out var displayIndex)
                || displayIndex < 0
                || displayIndex >= expected.Count
                || expected[displayIndex] != candidateId
                || matched[displayIndex]
                || !AdviceContractReader.HasCandidateMetadata(candidate)
                || !candidate.TryGetProperty("eligible", out var eligibleElement)
                || eligibleElement.ValueKind != JsonValueKind.True)
            {
                return null;
            }
            double? score = null;
            if (!candidate.TryGetProperty("score", out var scoreElement))
            {
                return null;
            }
            if (scoreElement.ValueKind == JsonValueKind.Number)
            {
                if (!scoreElement.TryGetDouble(out var numericScore)
                    || !double.IsFinite(numericScore)
                    || numericScore < 0
                    || numericScore > 100)
                {
                    return null;
                }
                score = numericScore;
            }
            else if (scoreElement.ValueKind != JsonValueKind.Null)
            {
                return null;
            }
            var isRecommended = candidateId == recommendedId;
            if (isRecommended)
            {
                if (!score.HasValue)
                {
                    return null;
                }
                recommendedMatched = true;
            }
            rows[displayIndex] = new ContextDrawerRow(
                label,
                score,
                isRecommended,
                AdviceReasonReader.Read(candidate)
            );
            matched[displayIndex] = true;
        }
        if (!matched.All(value => value) || !recommendedMatched)
            return null;
        return rows;
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
            && TryReadString(
                preferences,
                "route_mode",
                out var routeMode
            )
            && routeMode == expectedMode;

    private static void Invalidate(AdviceAvailability availability = AdviceAvailability.Waiting)
    {
        if (_pending is null || !_drawerHandle.IsValid)
        {
            return;
        }
        var rows = _captions
            .Select(caption => new ContextDrawerRow(
                caption,
                null,
                false
            ))
            .ToList();
        ContextDrawer.Render(_drawerHandle, rows, availability);
    }

    private static bool TryReadString(
        JsonElement element,
        string property,
        out string value)
    {
        value = "";
        if (!element.TryGetProperty(property, out var child)
            || child.ValueKind != JsonValueKind.String)
        {
            return false;
        }
        value = child.GetString() ?? "";
        return value.Length > 0;
    }

    private static bool TryReadNullableString(
        JsonElement element,
        string property,
        out string? value)
    {
        value = null;
        if (!element.TryGetProperty(property, out var child))
        {
            return false;
        }
        if (child.ValueKind == JsonValueKind.Null)
        {
            return true;
        }
        if (child.ValueKind != JsonValueKind.String)
        {
            return false;
        }
        value = child.GetString();
        return !string.IsNullOrWhiteSpace(value);
    }

    private static bool PendingMatchesScreen(
        NCardRewardSelectionScreen screen,
        PendingDecisionView pending,
        out List<string> captions)
    {
        captions = [];
        var visible = new List<DecisionOption>();
        try
        {
            var cardRow = CardRowField?.GetValue(screen) as Control;
            if (cardRow is null)
            {
                return false;
            }
            foreach (var child in cardRow.GetChildren())
            {
                if (child is NGridCardHolder { CardModel: { } model })
                {
                    visible.Add(RunStateReader.ReadDecisionOption(model));
                    // The visible game model owns the localized title even
                    // when the backend has never produced an advice file.
                    captions.Add(string.IsNullOrWhiteSpace(model.Title)
                        ? $"卡牌 {visible.Count}"
                        : model.Title);
                }
            }
            if (visible.Count != pending.Options.Count)
            {
                return false;
            }
            for (var index = 0; index < visible.Count; index++)
            {
                var actual = visible[index];
                var expected = pending.Options[index];
                if (expected.CandidateId != $"{index}:{actual.Card}"
                    || actual.Card != expected.CardId
                    || actual.Upgrades != expected.Upgrades
                    || actual.Enchantment != expected.Enchantment
                    || actual.EnchantmentAmount != expected.EnchantmentAmount
                    || actual.Affliction != expected.Affliction
                    || actual.AfflictionAmount != expected.AfflictionAmount)
                {
                    return false;
                }
            }
            return true;
        }
        catch (Exception exception)
        {
            Log.Info(
                "[STS2-Guide] PendingMatchesScreen failed: "
                + exception.Message
            );
            return false;
        }
    }
}
