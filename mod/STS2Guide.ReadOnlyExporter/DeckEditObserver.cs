using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Runs;

namespace STS2Guide.ReadOnlyExporter;

internal static class DeckEditObserver
{
    private static readonly object Gate = new();
    private static readonly Dictionary<
        NCardGridSelectionScreen,
        DeckEditSession
    > Sessions = new(ReferenceEqualityComparer.Instance);

    internal static void TryObserve(
        NCardGridSelectionScreen owner,
        IReadOnlyList<CardModel> cards,
        CardSelectorPrefs preferences,
        string operation)
    {
        lock (Gate)
        {
            if (!ReleaseCapabilityGate.IsEnabled("deck_edit"))
            {
                GenericAdviceController.Hide(owner);
                return;
            }
            if (operation is not ("upgrade" or "remove" or "transform")
                || cards.Count == 0
                || preferences.MinSelect != 1
                || preferences.MaxSelect != 1)
            {
                // Combination scoring is not implemented.  Multi-card
                // operations must not masquerade as independent choices.
                return;
            }
            var parent = StateEventWriter.PeekDecisionParent(
                $"deck_edit:{operation}",
                "rest_site",
                "merchant",
                "event_choice",
                "neow_choice"
            ) ?? StateEventWriter.ResolveDeckEditParent(operation);
            if (parent is null)
            {
                Log.Info(
                    "[STS2-Guide] Deck edit ignored because no verified "
                    + "parent decision is active."
                );
                return;
            }
            var candidates = new List<DecisionCandidateEnvelope>();
            var byCard = new Dictionary<CardModel, string>(
                ReferenceEqualityComparer.Instance
            );
            for (var index = 0; index < cards.Count; index++)
            {
                var card = cards[index];
                var cardId = StableIds.FromType(
                    card.GetType(),
                    "Model",
                    "Card"
                );
                var candidateId =
                    $"{operation}:{index}:{cardId}:"
                    + card.CurrentUpgradeLevel;
                var eligible = operation switch
                {
                    "upgrade" => card.IsUpgradable,
                    "remove" => card.IsRemovable,
                    "transform" => card.IsTransformable,
                    _ => false,
                };
                candidates.Add(new DecisionCandidateEnvelope
                {
                    CandidateId = candidateId,
                    Kind = "deck_edit",
                    EntityId = cardId,
                    Label = string.IsNullOrWhiteSpace(card.Title)
                        ? cardId
                        : card.Title,
                    Eligible = eligible,
                    UnavailableReason = eligible
                        ? null
                        : "unavailable",
                    Costs = [],
                    Payload = new CandidatePayload
                    {
                        Operation = operation,
                        TargetCandidateIds = [],
                    },
                });
                byCard[card] = candidateId;
            }
            if (!candidates.Any(candidate => candidate.Eligible))
            {
                return;
            }
            var pending = StateEventWriter.EmitGenericDecision(
                "deck_edit",
                candidates,
                new DecisionContext
                {
                    CanSkip = preferences.Cancelable,
                    CanReroll = false,
                    RewardSource = operation.ToUpperInvariant(),
                },
                owner,
                parent
            );
            if (pending is null)
            {
                return;
            }
            Sessions[owner] = new DeckEditSession(byCard);
            GenericAdviceController.Show(
                owner,
                pending,
                operation switch
                {
                    "upgrade" => "升级建议",
                    "remove" => "移除建议",
                    _ => "变化建议",
                }
            );
        }
    }

    internal static void ObserveSelectionTask(
        NCardGridSelectionScreen owner,
        Task<IEnumerable<CardModel>> selection)
    {
        _ = ObserveSelectionAsync(
            owner,
            selection
        );
    }

    private static async Task ObserveSelectionAsync(
        NCardGridSelectionScreen owner,
        Task<IEnumerable<CardModel>> selection)
    {
        try
        {
            var selected = (await selection).ToList();
            ScheduleSelectionCompletion(owner, selected);
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Deck-edit selection observation failed: "
                + exception.Message
            );
            ScheduleSelectionCompletion(owner, null);
        }
    }

    private static void ScheduleSelectionCompletion(
        NCardGridSelectionScreen owner,
        IReadOnlyList<CardModel>? selected)
        => Callable.From(() => ObserverSafety.Run(
            "deck_edit.selection_completed",
            () => CompleteSelection(owner, selected)
        )).CallDeferred();

    private static void CompleteSelection(
        NCardGridSelectionScreen owner,
        IReadOnlyList<CardModel>? selected)
    {
        lock (Gate)
        {
            if (!Sessions.TryGetValue(owner, out var session))
            {
                return;
            }
            if (selected?.Count == 1
                && session.Candidates.TryGetValue(
                    selected[0],
                    out var candidateId
                ))
            {
                StateEventWriter.EmitGenericSelected(owner, candidateId);
            }
            else
            {
                StateEventWriter.EmitGenericOwnerClosed(owner);
            }
            GenericAdviceController.Hide(owner);
            Sessions.Remove(owner);
        }
    }

    internal static void OnExited(NCardGridSelectionScreen owner)
    {
        lock (Gate)
        {
            StateEventWriter.EmitGenericOwnerClosed(owner);
            GenericAdviceController.Hide(owner);
            Sessions.Remove(owner);
        }
    }

    internal static string? InferBaseOperation(
        NDeckCardSelectScreen owner,
        CardSelectorPrefs preferences)
    {
        if (owner.GetType() != typeof(NDeckCardSelectScreen))
        {
            return null;
        }
        var removePrompt = CardSelectorPrefs.RemoveSelectionPrompt;
        if (
            preferences.Prompt.LocTable == removePrompt.LocTable
            && preferences.Prompt.LocEntryKey
                == removePrompt.LocEntryKey
        )
        {
            return "remove";
        }
        return null;
    }

    private sealed record DeckEditSession(
        Dictionary<CardModel, string> Candidates
    );
}

[HarmonyPatch(
    typeof(NDeckUpgradeSelectScreen),
    nameof(NDeckUpgradeSelectScreen.ShowScreen)
)]
internal static class DeckUpgradeOpenedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(
        IReadOnlyList<CardModel> cards,
        CardSelectorPrefs prefs,
        NDeckUpgradeSelectScreen __result)
        => ObserverSafety.Run(
            "deck_edit.upgrade_open",
            () => DeckEditObserver.TryObserve(
                __result,
                cards,
                prefs,
                "upgrade"
            )
        );
}

[HarmonyPatch(
    typeof(NDeckTransformSelectScreen),
    nameof(NDeckTransformSelectScreen.ShowScreen)
)]
internal static class DeckTransformOpenedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(
        IReadOnlyList<CardModel> cards,
        CardSelectorPrefs prefs,
        NDeckTransformSelectScreen __result)
        => ObserverSafety.Run(
            "deck_edit.transform_open",
            () => DeckEditObserver.TryObserve(
                __result,
                cards,
                prefs,
                "transform"
            )
        );
}

[HarmonyPatch(
    typeof(NDeckCardSelectScreen),
    nameof(NDeckCardSelectScreen.Create)
)]
internal static class DeckRemoveOpenedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(
        IReadOnlyList<CardModel> cards,
        CardSelectorPrefs prefs,
        NDeckCardSelectScreen __result)
        => ObserverSafety.Run(
            "deck_edit.remove_open",
            () =>
            {
                var operation = DeckEditObserver.InferBaseOperation(
                    __result,
                    prefs
                );
                if (operation is not null)
                {
                    DeckEditObserver.TryObserve(
                        __result,
                        cards,
                        prefs,
                        operation
                    );
                }
            }
        );
}

[HarmonyPatch(
    typeof(NCardGridSelectionScreen),
    nameof(NCardGridSelectionScreen.CardsSelected)
)]
internal static class DeckEditSelectedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(
        NCardGridSelectionScreen __instance,
        Task<IEnumerable<CardModel>> __result)
        => ObserverSafety.Run(
            "deck_edit.cards_selected",
            () => DeckEditObserver.ObserveSelectionTask(
                __instance,
                __result
            )
        );
}

[HarmonyPatch(
    typeof(NCardGridSelectionScreen),
    nameof(NCardGridSelectionScreen._ExitTree)
)]
internal static class DeckEditExitedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(
        NCardGridSelectionScreen __instance)
        => ObserverSafety.Run(
            "deck_edit.exit_tree",
            () => DeckEditObserver.OnExited(__instance)
        );
}
