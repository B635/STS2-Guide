using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace STS2Guide.ReadOnlyExporter;

internal static class RestSiteObserver
{
    private static readonly object Gate = new();

    internal static void TryObserve(NRestSiteRoom owner)
    {
        lock (Gate)
        {
            if (!ReleaseCapabilityGate.IsEnabled("rest_site"))
            {
                GenericAdviceController.Hide(owner);
                return;
            }
            TryObserveLocked(owner);
        }
    }

    private static void TryObserveLocked(NRestSiteRoom owner)
    {
        var player = RunStateReader.GetObservedPlayer();
        if (player is null
            && !RunStateReader.TryObserveFromRunManager())
        {
            return;
        }
        player = RunStateReader.GetObservedPlayer();
        if (player is null || owner.Options.Count == 0)
        {
            return;
        }
        var candidates = new List<DecisionCandidateEnvelope>();
        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var option in owner.Options)
        {
            var optionId = option.OptionId?.Trim();
            if (string.IsNullOrWhiteSpace(optionId)
                || !ids.Add(optionId))
            {
                Log.Error(
                    "[STS2-Guide] Rest-site observation stopped: "
                    + "option IDs are blank or duplicated."
                );
                GenericAdviceController.Hide(owner);
                return;
            }
            candidates.Add(BuildCandidate(option, player));
        }
        if (!candidates.Any(candidate => candidate.Eligible))
        {
            GenericAdviceController.Hide(owner);
            return;
        }
        var pending = StateEventWriter.EmitGenericDecision(
            "rest_site",
            candidates,
            new DecisionContext
            {
                CanSkip = false,
                CanReroll = false,
                RewardSource = "REST_SITE",
            },
            owner
        );
        if (pending is null)
        {
            GenericAdviceController.Hide(owner);
            return;
        }
        GenericAdviceController.Show(
            owner,
            pending,
            "篝火建议"
        );
    }

    internal static void OnSelected(
        NRestSiteRoom owner,
        RestSiteOption option)
    {
        lock (Gate)
        {
            StateEventWriter.EmitGenericSelected(
                owner,
                CandidateId(option)
            );
            GenericAdviceController.Hide(owner);
        }
    }

    internal static void OnClosed(NRestSiteRoom owner)
    {
        lock (Gate)
        {
            StateEventWriter.EmitGenericOwnerClosed(owner);
            GenericAdviceController.Hide(owner);
        }
    }

    private static DecisionCandidateEnvelope BuildCandidate(
        RestSiteOption option,
        MegaCrit.Sts2.Core.Entities.Players.Player player)
    {
        var optionId = option.OptionId.Trim();
        return new DecisionCandidateEnvelope
        {
            CandidateId = CandidateId(option),
            Kind = "rest_action",
            EntityId = null,
            Label = LocalizedLabel(
                option.Title.GetFormattedText(),
                optionId
            ),
            Eligible = option.IsEnabled,
            UnavailableReason = option.IsEnabled
                ? null
                : "unavailable",
            Costs = [],
            Payload = new CandidatePayload
            {
                ActionId = optionId,
                Effects = [ReadEffect(option, player)],
            },
        };
    }

    private static ChoiceEffect ReadEffect(
        RestSiteOption option,
        MegaCrit.Sts2.Core.Entities.Players.Player player)
    {
        if (option is HealRestSiteOption heal)
        {
            var amount = HealRestSiteOption.GetHealAmount(player);
            if (amount == decimal.Truncate(amount)
                && amount >= int.MinValue
                && amount <= int.MaxValue)
            {
                return new ChoiceEffect
                {
                    Kind = "hp_delta",
                    Amount = (int)amount,
                    TargetMode = "none",
                    Certainty = "exact",
                    SourceCode = "game_api:rest_heal_amount",
                };
            }
            return new ChoiceEffect
            {
                Kind = "hp_delta",
                MinAmount = (int)decimal.Floor(amount),
                MaxAmount = (int)decimal.Ceiling(amount),
                TargetMode = "none",
                Certainty = "bounded",
                SourceCode = "game_api:rest_heal_amount",
            };
        }
        if (option is SmithRestSiteOption)
        {
            return new ChoiceEffect
            {
                Kind = "upgrade_card",
                EntityType = "cards",
                TargetMode = "choose",
                Certainty = "exact",
                SourceCode = "game_api:rest_smith_option",
            };
        }
        return new ChoiceEffect
        {
            Kind = "followup_choice",
            TargetMode = "none",
            Certainty = "unknown",
            SourceCode = (
                "game_api:rest_action_unmodeled:"
                + StableIds.FromType(
                    option.GetType(),
                    "RestSiteOption"
                )
            ),
        };
    }

    private static string CandidateId(RestSiteOption option)
        => $"rest:{option.OptionId.Trim()}";

    private static string LocalizedLabel(
        string? label,
        string fallback)
        => string.IsNullOrWhiteSpace(label) ? fallback : label;
}

[HarmonyPatch(
    typeof(NRestSiteRoom),
    "OnAfterPlayerSelectedRestSiteOption"
)]
internal static class RestSiteSelectionCompletedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(
        NRestSiteRoom __instance,
        RestSiteOption option,
        bool success)
        => ObserverSafety.Run(
            "rest_site.selection_completed",
            () =>
            {
                if (success)
                {
                    RestSiteObserver.OnSelected(__instance, option);
                }
            }
        );
}

[HarmonyPatch(
    typeof(NRestSiteRoom),
    nameof(NRestSiteRoom.Create)
)]
internal static class RestSiteCreatedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(NRestSiteRoom __result)
        => ObserverSafety.Run(
            "rest_site.create",
            () => RestSiteObserver.TryObserve(__result)
        );
}

[HarmonyPatch(
    typeof(NRestSiteRoom),
    nameof(NRestSiteRoom.EnableOptions)
)]
internal static class RestSiteEnabledObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(NRestSiteRoom __instance)
        => ObserverSafety.Run(
            "rest_site.enable_options",
            () => RestSiteObserver.TryObserve(__instance)
        );
}

[HarmonyPatch(
    typeof(NRestSiteRoom),
    nameof(NRestSiteRoom.BeforeExitingRoom)
)]
internal static class RestSiteClosingObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(NRestSiteRoom __instance)
        => ObserverSafety.Run(
            "rest_site.before_exit",
            () => RestSiteObserver.OnClosed(__instance)
        );
}

[HarmonyPatch(
    typeof(NRestSiteRoom),
    nameof(NRestSiteRoom._ExitTree)
)]
internal static class RestSiteExitedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(NRestSiteRoom __instance)
        => ObserverSafety.Run(
            "rest_site.exit_tree",
            () => RestSiteObserver.OnClosed(__instance)
        );
}
