using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;

namespace STS2Guide.ReadOnlyExporter;

internal static class MerchantObserver
{
    private static readonly object Gate = new();
    private static NMerchantInventory? _owner;

    internal static void TryObserve(NMerchantInventory owner)
    {
        lock (Gate)
        {
            if (!ReleaseCapabilityGate.IsEnabled("merchant"))
            {
                GenericAdviceController.Hide(owner);
                return;
            }
            if (!owner.IsOpen || owner.Inventory is null)
            {
                return;
            }
            RunStateReader.Observe(owner.Inventory.Player);
            if (!TryBuildCandidates(
                owner.Inventory,
                out var candidates
            ))
            {
                GenericAdviceController.Hide(owner);
                return;
            }
            var pending = StateEventWriter.EmitGenericDecision(
                "merchant",
                candidates,
                new DecisionContext
                {
                    CanSkip = false,
                    CanReroll = false,
                    RewardSource = "MERCHANT",
                },
                owner.Inventory
            );
            if (pending is null)
            {
                GenericAdviceController.Hide(owner);
                return;
            }
            _owner = owner;
            GenericAdviceController.Show(
                owner,
                pending,
                "商店建议"
            );
        }
    }

    internal static void OnInventoryUpdated(MerchantEntry entry)
    {
        lock (Gate)
        {
            var owner = _owner;
            var inventory = owner?.Inventory;
            if (owner is null
                || inventory is null
                || !owner.IsOpen
                || !inventory.AllEntries.Any(candidate =>
                    ReferenceEquals(candidate, entry)
                ))
            {
                return;
            }
            TryObserve(owner);
        }
    }

    internal static void OnClosed(NMerchantInventory owner)
    {
        lock (Gate)
        {
            StateEventWriter.EmitGenericOwnerClosed(owner);
            GenericAdviceController.Hide(owner);
            if (ReferenceEquals(owner, _owner))
            {
                _owner = null;
            }
        }
    }

    private static bool TryBuildCandidates(
        MerchantInventory inventory,
        out List<DecisionCandidateEnvelope> candidates)
    {
        candidates = [];
        var entries = inventory.AllEntries.ToList();
        for (var index = 0; index < entries.Count; index++)
        {
            var entry = entries[index];
            if (!TryBuildCandidate(
                entry,
                index,
                out var candidate
            ))
            {
                Log.Error(
                    "[STS2-Guide] Merchant observation stopped: "
                    + "an entry did not expose a supported public model."
                );
                candidates = [];
                return false;
            }
            candidates.Add(candidate);
        }
        candidates.Add(new DecisionCandidateEnvelope
        {
            CandidateId = "leave",
            Kind = "leave",
            EntityId = null,
            Label = "离开",
            Eligible = true,
            UnavailableReason = null,
            Costs = [],
            Payload = new CandidatePayload(),
        });
        return true;
    }

    private static bool TryBuildCandidate(
        MerchantEntry entry,
        int index,
        out DecisionCandidateEnvelope candidate)
    {
        var stocked = entry.IsStocked;
        var eligible = stocked && entry.EnoughGold;
        var unavailableReason = eligible
            ? null
            : stocked
                ? "insufficient_gold"
                : "sold_out";
        string offerKind;
        string? entityId;
        string label;
        ChoiceEffect effect;

        switch (entry)
        {
            case MerchantCardEntry cardEntry
                when cardEntry.CreationResult?.Card is not null:
            {
                var model = cardEntry.CreationResult.Card;
                offerKind = "card";
                entityId = StableIds.FromType(
                    model.GetType(),
                    "Model",
                    "Card"
                );
                label = string.IsNullOrWhiteSpace(model.Title)
                    ? entityId
                    : model.Title;
                effect = EntityEffect(
                    "add_card",
                    "cards",
                    entityId,
                    "game_api:merchant_card_entry"
                );
                break;
            }
            case MerchantRelicEntry relicEntry
                when relicEntry.Model is not null:
            {
                var model = relicEntry.Model;
                offerKind = "relic";
                entityId = StableIds.FromType(
                    model.GetType(),
                    "Model",
                    "Relic"
                );
                label = LocalizedLabel(
                    model.Title.GetFormattedText(),
                    entityId
                );
                effect = EntityEffect(
                    "add_relic",
                    "relics",
                    entityId,
                    "game_api:merchant_relic_entry"
                );
                break;
            }
            case MerchantPotionEntry potionEntry
                when potionEntry.Model is not null:
            {
                var model = potionEntry.Model;
                offerKind = "potion";
                entityId = StableIds.FromType(
                    model.GetType(),
                    "Model",
                    "Potion"
                );
                label = LocalizedLabel(
                    model.Title.GetFormattedText(),
                    entityId
                );
                effect = EntityEffect(
                    "add_potion",
                    "potions",
                    entityId,
                    "game_api:merchant_potion_entry"
                );
                break;
            }
            case MerchantCardRemovalEntry:
                offerKind = "card_removal";
                entityId = null;
                label = "移除一张牌";
                effect = new ChoiceEffect
                {
                    Kind = "remove_card",
                    EntityType = "cards",
                    EntityId = null,
                    TargetMode = "choose",
                    Certainty = "exact",
                    SourceCode =
                        "game_api:merchant_card_removal_entry",
                };
                break;
            default:
                candidate = null!;
                return false;
        }

        var slotId = $"merchant:{offerKind}:{index}";
        candidate = new DecisionCandidateEnvelope
        {
            CandidateId = entityId is null
                ? slotId
                : $"{slotId}:{entityId}",
            Kind = "merchant_offer",
            EntityId = entityId,
            Label = label,
            Eligible = eligible,
            UnavailableReason = unavailableReason,
            Costs =
            [
                new DecisionCost
                {
                    Kind = "gold",
                    Amount = Math.Max(0, entry.Cost),
                    ResourceId = null,
                },
            ],
            Payload = new CandidatePayload
            {
                SlotId = slotId,
                OfferKind = offerKind,
                IsStocked = stocked,
                Effects = [effect],
            },
        };
        return true;
    }

    private static ChoiceEffect EntityEffect(
        string kind,
        string entityType,
        string entityId,
        string sourceCode)
        => new()
        {
            Kind = kind,
            EntityType = entityType,
            EntityId = entityId,
            TargetMode = "specific",
            Certainty = "exact",
            SourceCode = sourceCode,
        };

    private static string LocalizedLabel(
        string? label,
        string fallback)
        => string.IsNullOrWhiteSpace(label) ? fallback : label;
}

[HarmonyPatch(
    typeof(NMerchantInventory),
    nameof(NMerchantInventory.Open)
)]
internal static class MerchantOpenedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(NMerchantInventory __instance)
        => ObserverSafety.Run(
            "merchant.open",
            () => MerchantObserver.TryObserve(__instance)
        );
}

[HarmonyPatch(
    typeof(MerchantEntry),
    nameof(MerchantEntry.OnMerchantInventoryUpdated)
)]
internal static class MerchantUpdatedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(MerchantEntry __instance)
        => ObserverSafety.Run(
            "merchant.inventory_updated",
            () => MerchantObserver.OnInventoryUpdated(__instance)
        );
}

[HarmonyPatch(
    typeof(MerchantEntry),
    nameof(MerchantEntry.InvokePurchaseCompleted)
)]
internal static class MerchantPurchaseCompletedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(MerchantEntry entry)
        => ObserverSafety.Run(
            "merchant.purchase_completed",
            () => MerchantObserver.OnInventoryUpdated(entry)
        );
}

[HarmonyPatch(
    typeof(NMerchantInventory),
    nameof(NMerchantInventory.OnCardRemovalUsed)
)]
internal static class MerchantRemovalUpdatedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(NMerchantInventory __instance)
        => ObserverSafety.Run(
            "merchant.removal_updated",
            () => MerchantObserver.TryObserve(__instance)
        );
}

[HarmonyPatch(typeof(NMerchantInventory), "Close")]
internal static class MerchantClosingObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(NMerchantInventory __instance)
        => ObserverSafety.Run(
            "merchant.close",
            () => MerchantObserver.OnClosed(__instance)
        );
}

[HarmonyPatch(
    typeof(NMerchantInventory),
    nameof(NMerchantInventory._ExitTree)
)]
internal static class MerchantExitedObservationPatch
{
    [HarmonyPostfix]
    private static void Postfix(NMerchantInventory __instance)
        => ObserverSafety.Run(
            "merchant.exit_tree",
            () => MerchantObserver.OnClosed(__instance)
        );
}
