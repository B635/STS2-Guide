using System.Reflection;
using HarmonyLib;
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
        RunStateReader.Observe(__instance);
    }
}

[HarmonyPatch(typeof(CardReward), nameof(CardReward.Populate))]
internal static class CardRewardObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterPopulate(CardReward __instance)
    {
        try
        {
            RunStateReader.Observe(__instance.Player);
            var options = CardOptionReader.Read(__instance);
            if (options.Count == 0)
            {
                return;
            }
            StateEventWriter.EmitCardReward(
                options,
                new DecisionContext
                {
                    CanSkip = __instance.CanSkip,
                    CanReroll = __instance.CanReroll,
                    RewardSource = CardOptionReader.ReadRewardType(
                        __instance
                    ),
                }
            );
        }
        catch (Exception exception)
        {
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
    internal static void AfterSelectCard(NCardHolder cardHolder)
    {
        try
        {
            if (cardHolder.CardModel is { } card)
            {
                StateEventWriter.EmitCardSelected(card);
                CardRewardAdvicePanel.Hide();
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
    internal static void AfterSkipped()
    {
        StateEventWriter.EmitCardSkipped();
        CardRewardAdvicePanel.Hide();
    }
}

[HarmonyPatch(
    typeof(NRewardsScreen),
    nameof(NRewardsScreen.AfterOverlayClosed)
)]
internal static class RewardClosedObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterOverlayClosed()
    {
        StateEventWriter.EmitDecisionClosed();
        CardRewardAdvicePanel.Hide();
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
            CardRewardAdvicePanel.Show(__result);
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
    internal static void AfterExitTree()
    {
        CardRewardAdvicePanel.Hide();
    }
}

internal static class CardOptionReader
{
    private static readonly PropertyInfo? RewardTypeProperty =
        typeof(CardReward).GetProperty(
            "RewardType",
            BindingFlags.Public
            | BindingFlags.NonPublic
            | BindingFlags.Instance
        );

    internal static List<DecisionOption> Read(CardReward reward)
    {
        return reward.Cards
            .Select(RunStateReader.ReadDecisionOption)
            .Take(10)
            .ToList();
    }

    internal static string? ReadRewardType(CardReward reward)
    {
        return RewardTypeProperty?
            .GetValue(reward)?
            .ToString()?
            .ToUpperInvariant();
    }
}
