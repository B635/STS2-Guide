using System.Security.Cryptography;
using System.Runtime.CompilerServices;
using System.Text;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace STS2Guide.ReadOnlyExporter;

internal static class EventDecisionObserver
{
    private static readonly object Gate = new();
    private static readonly Dictionary<EventModel, EventSession> Sessions =
        new(ReferenceEqualityComparer.Instance);
    private static readonly Dictionary<EventOption, OptionBinding> Bindings =
        new(ReferenceEqualityComparer.Instance);
    private static ConditionalWeakTable<object, CapturedCardRewardContext>
        CapturedCardRewards = new();

    internal static void OnCreated(
        NEventRoom owner,
        EventModel model)
    {
        lock (Gate)
        {
            // Creating a new verified event room is a semantic boundary for
            // any older event lifecycle.  It is safe to discard old owners;
            // a child CardReward already captured from them keeps its own
            // weak, reward-object-scoped context below.
            Sessions.Clear();
            Bindings.Clear();
            // Neow and the other ancients (AncientEventModel, verified in the
            // 0.111.0 assembly) offer relic options: one relic-choice decision.
            var capability = model is AncientEventModel
                ? "neow_choice"
                : "event_choice";
            // Keep the verified owner/model lifecycle even when the
            // recommendation capability is disabled. CardReward uses this
            // context to prevent a special reward from masquerading as an
            // ordinary combat reward.
            Sessions[model] = new EventSession(owner);
            if (!ReleaseCapabilityGate.IsEnabled(capability))
            {
                GenericAdviceController.Hide(owner);
                return;
            }
            TryObserveLocked(model);
        }
    }

    internal static void TryObserve(EventModel model)
    {
        lock (Gate)
        {
            TryObserveLocked(model);
        }
    }

    internal static void ScheduleObserve(EventModel model)
        => Callable.From(() => ObserverSafety.Run(
            "event.deferred_state",
            () => TryObserve(model)
        )).CallDeferred();

    internal static void ScheduleFinished(EventModel model)
        => Callable.From(() => ObserverSafety.Run(
            "event.deferred_finished",
            () => OnFinished(model)
        )).CallDeferred();

    internal static bool TryCaptureCardRewardContext(
        object reward,
        out string capability,
        out string sourceType)
    {
        lock (Gate)
        {
            if (CapturedCardRewards.TryGetValue(
                    reward,
                    out var captured
                ))
            {
                capability = captured.Capability;
                sourceType = captured.SourceType;
                return true;
            }
            capability = "";
            sourceType = "";
            // Presence, not IsFinished, is the ownership boundary. A chosen
            // option may mark the event finished synchronously before the
            // nested reward screen is populated.
            var active = Sessions.ToList();
            if (active.Count == 0)
            {
                return false;
            }
            if (active.Count != 1)
            {
                // Ambiguous special ownership is still special. Blank
                // outputs force the CardReward observer to fail closed.
                CapturedCardRewards.Add(
                    reward,
                    new CapturedCardRewardContext(null, "", "")
                );
                return true;
            }
            var isAncient = active[0].Key is AncientEventModel;
            capability = isAncient ? "neow_choice" : "event_choice";
            sourceType = capability;
            CapturedCardRewards.Add(
                reward,
                new CapturedCardRewardContext(
                    active[0].Value,
                    capability,
                    sourceType
                )
            );
            return true;
        }
    }

    internal static bool TryResolveCapturedCardRewardParent(
        object reward,
        out DecisionParentContext? decisionParent)
    {
        lock (Gate)
        {
            decisionParent = null;
            if (!CapturedCardRewards.TryGetValue(reward, out var captured))
            {
                return false;
            }
            decisionParent = captured.Session?.CardRewardParent;
            return true;
        }
    }

    internal static void OnNonEventDecisionBoundary()
    {
        lock (Gate)
        {
            Sessions.Clear();
            Bindings.Clear();
            StateEventWriter.ClearUnconsumedDecisionParentAtBoundary();
        }
    }

    internal static void ResetRunLifecycle()
    {
        lock (Gate)
        {
            Sessions.Clear();
            Bindings.Clear();
            CapturedCardRewards = new();
        }
    }

    internal static void OnChosen(EventOption option)
    {
        lock (Gate)
        {
            if (!Bindings.TryGetValue(option, out var binding))
            {
                return;
            }
            var selected = StateEventWriter.EmitGenericSelected(
                binding.Owner,
                binding.CandidateId
            );
            if (selected)
            {
                foreach (var pair in Sessions.Where(pair =>
                    ReferenceEquals(pair.Value.Owner, binding.Owner)))
                {
                    var sourceType = pair.Key is AncientEventModel
                        ? "neow_choice"
                        : "event_choice";
                    pair.Value.CardRewardParent =
                        StateEventWriter.PeekDecisionParent(
                            "card_reward",
                            sourceType
                        );
                }
            }
            GenericAdviceController.Hide(binding.Owner);
            RemoveBindingsForOwner(binding.Owner);
        }
    }

    internal static void OnFinished(EventModel model)
    {
        lock (Gate)
        {
            if (!Sessions.TryGetValue(model, out var session))
            {
                return;
            }
            SchedulePresentationClose(model, session.Owner);
        }
    }

    private static void SchedulePresentationClose(
        EventModel model,
        NEventRoom owner)
        => Callable.From(() => ObserverSafety.Run(
            "event.deferred_presentation_close",
            () =>
            {
                lock (Gate)
                {
                    if (!Sessions.TryGetValue(model, out var session)
                        || !ReferenceEquals(session.Owner, owner))
                    {
                        return;
                    }
                    StateEventWriter.EmitGenericOwnerClosed(owner);
                    GenericAdviceController.Hide(owner);
                    RemoveBindingsForOwner(owner);
                }
            }
        )).CallDeferred();

    internal static void OnExited(NEventRoom owner)
    {
        lock (Gate)
        {
            GenericAdviceController.Hide(owner);
            foreach (var model in Sessions
                .Where(pair => ReferenceEquals(
                    pair.Value.Owner,
                    owner
                ))
                .Select(pair => pair.Key)
                .ToList())
            {
                // Exit may happen inside EventOption.Chosen before its
                // Harmony Postfix.  Delay presentation closure once so the
                // real selection can establish a typed child marker first.
                // The session itself is a semantic tombstone: it remains
                // across any captured child and is cleared only when a
                // verified next-decision boundary appears or the run resets.
                SchedulePresentationClose(model, owner);
            }
        }
    }

    private static void TryObserveLocked(EventModel model)
    {
        var capability = model is AncientEventModel
            ? "neow_choice"
            : "event_choice";
        if (!ReleaseCapabilityGate.IsEnabled(capability))
        {
            if (Sessions.TryGetValue(model, out var disabledSession))
            {
                GenericAdviceController.Hide(disabledSession.Owner);
            }
            return;
        }
        if (!Sessions.TryGetValue(model, out var session)
            || model.IsFinished
            || model.CurrentOptions.Count == 0)
        {
            return;
        }
        RunStateReader.Observe(model.Owner);
        var eventId = StableIds.FromType(
            model.CanonicalInstance.GetType(),
            "Event",
            "Model"
        );
        var optionIds = new List<string>();
        foreach (var option in model.CurrentOptions)
        {
            var optionId = option.TextKey?.Trim();
            if (string.IsNullOrWhiteSpace(optionId)
                || optionId.Length > 200)
            {
                Log.Error(
                    "[STS2-Guide] Event observation stopped: "
                    + "EventOption.TextKey is blank or exceeds the "
                    + "v9 protocol boundary."
                );
                GenericAdviceController.Hide(session.Owner);
                return;
            }
            optionIds.Add(optionId);
        }
        var pageId = PageId(eventId, optionIds);
        if (session.PageId != pageId)
        {
            session.PageId = pageId;
            session.PageSource = new object();
        }
        RemoveBindingsForOwner(session.Owner);
        var isAncient = model is AncientEventModel;
        var eventType = isAncient
            ? "neow_choice"
            : "event_choice";
        var candidates = new List<DecisionCandidateEnvelope>();
        for (var index = 0; index < model.CurrentOptions.Count; index++)
        {
            var option = model.CurrentOptions[index];
            var optionId = optionIds[index];
            var candidateId = CandidateId(
                eventId,
                pageId,
                index,
                optionId
            );
            var candidate = BuildCandidate(
                model,
                option,
                candidateId,
                eventId,
                pageId,
                optionId,
                isAncient
            );
            candidates.Add(candidate);
            Bindings[option] = new OptionBinding(
                session.Owner,
                candidateId
            );
        }
        if (!candidates.Any(candidate => candidate.Eligible))
        {
            GenericAdviceController.Hide(session.Owner);
            return;
        }
        var pending = StateEventWriter.EmitGenericDecision(
            eventType,
            candidates,
            new DecisionContext
            {
                CanSkip = false,
                CanReroll = false,
                RewardSource = isAncient ? "NEOW" : "EVENT",
            },
            session.PageSource
        );
        if (pending is null)
        {
            GenericAdviceController.Hide(session.Owner);
            return;
        }
        GenericAdviceController.Show(
            session.Owner,
            pending,
            model is Neow ? "祝福建议" : isAncient ? "古神建议" : "事件建议"
        );
    }

    private static DecisionCandidateEnvelope BuildCandidate(
        EventModel model,
        EventOption option,
        string candidateId,
        string eventId,
        string pageId,
        string optionId,
        bool isAncient)
    {
        var label = option.Title.GetFormattedText();
        if (string.IsNullOrWhiteSpace(label))
        {
            label = optionId;
        }
        var effect = VerifiedEventEffectReader.Read(model, option);
        // The relic the option gives (EventOption.Relic, verified in the
        // 0.111.0 assembly); its effects come from the reviewed facts.
        var relicId = option.Relic is null
            ? null
            : StableIds.FromType(option.Relic.GetType(), "Relic", "Model");
        if (isAncient)
        {
            return new DecisionCandidateEnvelope
            {
                CandidateId = candidateId,
                Kind = "neow_blessing",
                EntityId = relicId,
                Label = label,
                Eligible = !option.IsLocked,
                UnavailableReason = option.IsLocked
                    ? "locked"
                    : null,
                Costs = [],
                Payload = new CandidatePayload
                {
                    BlessingId = optionId,
                    StageId = pageId,
                    OptionId = optionId,
                    Effects = [effect],
                },
            };
        }
        return new DecisionCandidateEnvelope
        {
            CandidateId = candidateId,
            Kind = "event_option",
            EntityId = relicId,
            Label = label,
            Eligible = !option.IsLocked,
            UnavailableReason = option.IsLocked
                ? "locked"
                : null,
            Costs = [],
            Payload = new CandidatePayload
            {
                EventId = eventId,
                PageId = pageId,
                OptionId = optionId,
                Effects = [effect],
            },
        };
    }

    private static string PageId(
        string eventId,
        IReadOnlyList<string> optionIds)
    {
        var canonical = eventId + "\n" + string.Join(
            "\n",
            optionIds.Select((value, index) => $"{index}:{value}")
        );
        return "PAGE_"
            + Convert.ToHexString(
                SHA256.HashData(
                    Encoding.UTF8.GetBytes(canonical)
                )
            )[..16];
    }

    private static string CandidateId(
        string eventId,
        string pageId,
        int optionIndex,
        string optionId)
    {
        var canonical =
            $"{eventId}\n{pageId}\n{optionIndex}\n{optionId}";
        return "event:"
            + Convert.ToHexString(
                SHA256.HashData(
                    Encoding.UTF8.GetBytes(canonical)
                )
            )[..24];
    }

    private static void RemoveBindingsForOwner(NEventRoom owner)
    {
        foreach (var option in Bindings
            .Where(pair => ReferenceEquals(
                pair.Value.Owner,
                owner
            ))
            .Select(pair => pair.Key)
            .ToList())
        {
            Bindings.Remove(option);
        }
    }

    private sealed class EventSession(NEventRoom owner)
    {
        internal NEventRoom Owner { get; } = owner;
        internal string? PageId { get; set; }
        internal object PageSource { get; set; } = new();
        internal DecisionParentContext? CardRewardParent { get; set; }
    }

    private sealed record CapturedCardRewardContext(
        EventSession? Session,
        string Capability,
        string SourceType
    );

    private sealed record OptionBinding(
        NEventRoom Owner,
        string CandidateId
    );
}

[HarmonyPatch(
    typeof(NEventRoom),
    nameof(NEventRoom.Create)
)]
internal static class EventRoomCreatedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(
        EventModel eventModel,
        NEventRoom __result)
        => ObserverSafety.Run(
            "event.create",
            () => EventDecisionObserver.OnCreated(__result, eventModel)
        );
}

[HarmonyPatch(typeof(EventModel), "SetEventState")]
internal static class EventStateUpdatedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(EventModel __instance)
        => ObserverSafety.Run(
            "event.set_state",
            () => EventDecisionObserver.ScheduleObserve(__instance)
        );
}

[HarmonyPatch(
    typeof(EventOption),
    nameof(EventOption.Chosen)
)]
internal static class EventOptionChosenObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(EventOption __instance)
        => ObserverSafety.Run(
            "event.option_chosen",
            () => EventDecisionObserver.OnChosen(__instance)
        );
}

[HarmonyPatch(
    typeof(NEventRoom),
    nameof(NEventRoom.OptionButtonClicked)
)]
internal static class EventOptionClickedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(EventOption option)
        => ObserverSafety.Run(
            "event.option_clicked",
            () => EventDecisionObserver.OnChosen(option)
        );
}

[HarmonyPatch(typeof(EventModel), "SetEventFinished")]
internal static class EventFinishedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(EventModel __instance)
        => ObserverSafety.Run(
            "event.finished",
            () => EventDecisionObserver.ScheduleFinished(__instance)
        );
}

[HarmonyPatch(
    typeof(NEventRoom),
    nameof(NEventRoom._ExitTree)
)]
internal static class EventRoomExitedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(NEventRoom __instance)
        => ObserverSafety.Run(
            "event.exit_tree",
            () => EventDecisionObserver.OnExited(__instance)
        );
}
