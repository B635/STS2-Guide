using System.Reflection;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Rewards;

namespace STS2Guide.ReadOnlyExporter;

[HarmonyPatch(typeof(Player), nameof(Player.PopulateCombatState))]
internal static class PlayerObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterPopulateCombatState(Player __instance)
    {
        try
        {
            RunStateReader.Observe(__instance);
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Player observation failed: "
                + exception.Message
            );
        }
    }
}

[HarmonyPatch(typeof(CardReward), nameof(CardReward.Populate))]
internal static class CardRewardObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterPopulate(CardReward __instance)
        => ObserverSafety.Run(
            "card_reward.schedule_populate",
            () => ScheduleOrObserve(__instance)
        );

    private static void ScheduleOrObserve(CardReward reward)
    {
        if (!EventDecisionObserver.TryCaptureCardRewardContext(
                reward,
                out var parentCapability,
                out var parentSourceType
            ))
        {
            CardRewardAdvicePanel.ClearSpecialRewardExpectation();
            Observe(reward, requiredParentSourceType: null);
            return;
        }
        CardRewardAdvicePanel.PrepareSpecialReward(
            parentCapability,
            parentSourceType
        );
        if (string.IsNullOrWhiteSpace(parentCapability)
            || string.IsNullOrWhiteSpace(parentSourceType)
            || !ReleaseCapabilityGate.IsEnabled(parentCapability))
        {
            CardRewardAdvicePanel.Hide();
            CardRewardAdvicePanel.PrepareSpecialReward(
                parentCapability,
                parentSourceType
            );
            return;
        }
        Callable.From(() => ObserverSafety.Run(
            "card_reward.deferred_special_populate",
            () => Observe(reward, parentSourceType)
        )).CallDeferred();
    }

    private static void Observe(
        CardReward __instance,
        string? requiredParentSourceType)
    {
        try
        {
            if (!ReleaseCapabilityGate.IsEnabled("card_reward"))
            {
                CardRewardAdvicePanel.Hide();
                return;
            }
            if (requiredParentSourceType is not null)
            {
                if (!ReleaseCapabilityGate.IsEnabled(
                        requiredParentSourceType
                    ))
                {
                    CardRewardAdvicePanel.Hide();
                    Log.Info(
                        "[STS2-Guide] Special Card Reward ignored because "
                        + "its parent capability is unavailable."
                    );
                    return;
                }
                if (!EventDecisionObserver
                        .TryResolveCapturedCardRewardParent(
                            __instance,
                            out var requiredParent
                        )
                    || requiredParent is null)
                {
                    CardRewardAdvicePanel.Hide();
                    Log.Info(
                        "[STS2-Guide] Special Card Reward ignored because "
                        + "its exact selected parent is unavailable."
                    );
                    return;
                }
                ObserveCaptured(
                    __instance,
                    requiredParentSourceType,
                    requiredParent
                );
                return;
            }
            ObserveCaptured(__instance, null, null);
        }
        catch (Exception exception)
        {
            CardRewardAdvicePanel.Hide();
            Log.Error(
                $"[STS2-Guide] Card reward observation failed: "
                + exception.Message
            );
        }
    }

    private static void ObserveCaptured(
        CardReward __instance,
        string? requiredParentSourceType,
        DecisionParentContext? requiredParent)
    {
        try
        {
            Log.Info("[STS2-Guide] CardReward.Populate triggered.");

            var p = __instance.Player;
            if (p is not null)
            {
                Log.Info("[STS2-Guide] CardReward.Populate: __instance.Player type=" + p.GetType().FullName);
                RunStateReader.Observe(p);
            }
            else
            {
                Log.Info("[STS2-Guide] CardReward.Populate: __instance.Player is null, trying TryObserveFromRunManager.");
                RunStateReader.TryObserveFromRunManager();
            }

            var options = CardOptionReader.Read(__instance);
            if (options.Count == 0)
            {
                Log.Info("[STS2-Guide] CardReward.Populate: 0 options, returning.");
                return;
            }
            var rewardSource = requiredParentSourceType switch
            {
                "neow_choice" => "NEOW",
                "event_choice" => "EVENT",
                // This hook observes CardReward.Populate itself.  The v9
                // protocol therefore defines the parentless semantic source
                // as CARD; no nullable reflected property is needed.
                _ => "CARD",
            };
            StateEventWriter.EmitCardReward(
                options,
                new DecisionContext
                {
                    CanSkip = __instance.CanSkip,
                    CanReroll = __instance.CanReroll,
                    // CardReward.RewardType is CARD for both ordinary and
                    // nested rewards. A verified typed parent supplies the
                    // more precise semantic source without guessing UI text.
                    RewardSource = rewardSource,
                },
                __instance,
                requiredParentSourceType,
                requiredParent
            );
        }
        catch (Exception exception)
        {
            CardRewardAdvicePanel.Hide();
            Log.Error(
                $"[STS2-Guide] Card reward observation failed: "
                + exception.Message
            );
        }
    }
}

[HarmonyPatch(typeof(NCardRewardSelectionScreen), "SelectCard")]
internal static class CardSelectedObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterSelectCard(
        NCardRewardSelectionScreen __instance,
        NCardHolder cardHolder)
    {
        try
        {
            if (cardHolder.CardModel is { } card)
            {
                StateEventWriter.EmitCardSelected(card, __instance);
                CardRewardAdvicePanel.Hide(__instance);
            }
        }
        catch (Exception exception)
        {
            Log.Error(
                $"[STS2-Guide] Card selection observation failed: "
                + exception.Message
            );
        }
    }
}

[HarmonyPatch(typeof(CardReward), nameof(CardReward.OnSkipped))]
internal static class CardSkippedObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterSkipped(CardReward __instance)
    {
        try
        {
            if (StateEventWriter.EmitCardSkipped(__instance))
            {
                CardRewardAdvicePanel.Hide();
            }
        }
        catch (Exception exception)
        {
            Log.Error("[STS2-Guide] Skip observation failed: " + exception.Message);
        }
    }
}

[HarmonyPatch(
    typeof(NCardRewardSelectionScreen),
    "OnAlternateRewardSelected"
)]
internal static class CardRewardAlternativeSelectedObservationPatch
{
    private static readonly FieldInfo? ExtraOptionsField =
        typeof(NCardRewardSelectionScreen).GetField(
            "_extraOptions",
            BindingFlags.NonPublic | BindingFlags.Instance
        );

    [HarmonyPostfix]
    internal static void AfterAlternateRewardSelected(
        NCardRewardSelectionScreen __instance,
        int index)
    {
        try
        {
            var alternatives = ExtraOptionsField?.GetValue(__instance)
                as IReadOnlyList<CardRewardAlternative>;
            if (alternatives is null
                || index < 0
                || index >= alternatives.Count)
            {
                Log.Error(
                    "[STS2-Guide] Card reward alternative could not be "
                    + "resolved from the verified screen API; leaving "
                    + "the decision open."
                );
                return;
            }
            var alternative = alternatives[index];
            if (!string.Equals(
                alternative.OptionId,
                "Skip",
                StringComparison.Ordinal
            ))
            {
                return;
            }
            if (StateEventWriter.EmitCardSkippedFromScreen(__instance))
            {
                CardRewardAdvicePanel.Hide(__instance);
            }
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Card reward alternative skip "
                + "observation failed: "
                + exception.Message
            );
        }
    }
}

[HarmonyPatch(
    typeof(NCardRewardSelectionScreen),
    nameof(NCardRewardSelectionScreen.ShowScreen)
)]
internal static class CardRewardAdvicePanelOpenPatch
{
    [HarmonyPostfix]
    internal static void AfterShowScreen(
        NCardRewardSelectionScreen __result
    )
    {
        if (__result is not null)
        {
            try
            {
                CardRewardAdvicePanel.Show(__result);
            }
            catch (Exception exception)
            {
                Log.Error("[STS2-Guide] Advice panel open failed: " + exception.Message);
            }
        }
    }
}

[HarmonyPatch(
    typeof(NCardRewardSelectionScreen),
    nameof(NCardRewardSelectionScreen._ExitTree)
)]
internal static class CardRewardAdvicePanelExitPatch
{
    [HarmonyPostfix]
    internal static void AfterExitTree(
        NCardRewardSelectionScreen __instance)
    {
        try
        {
            // Tree exit is presentation lifecycle only. Saving to the menu
            // also removes this screen, while the game decision remains
            // unfinished and must be recoverable on Continue.
            StateEventWriter.UnbindCardRewardScreen(__instance);
            CardRewardAdvicePanel.Hide(__instance);
        }
        catch (Exception exception)
        {
            Log.Error("[STS2-Guide] Advice panel cleanup failed: " + exception.Message);
        }
    }
}

internal static class CardOptionReader
{
    internal static List<DecisionOption> Read(CardReward reward)
    {
        return reward.Cards
            .Select(RunStateReader.ReadDecisionOption)
            .Take(10)
            .ToList();
    }
}
