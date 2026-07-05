using System.Reflection;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;

namespace STS2Guide.ReadOnlyExporter;

internal static class RunStateReader
{
    private static readonly object Gate = new();
    private static Player? _observedPlayer;

    /// <summary>
    /// Returns a clone of the observed player for map reading etc.
    /// External callers must not modify the player.
    /// </summary>
    internal static Player? GetObservedPlayer()
    {
        lock (Gate)
        {
            return _observedPlayer;
        }
    }

    internal static void Observe(Player player)
    {
        lock (Gate)
        {
            if (_observedPlayer is null || IsLocalPlayer(player))
            {
                _observedPlayer = player;
            }
        }
    }

    internal static void Clear()
    {
        lock (Gate)
        {
            _observedPlayer = null;
        }
    }

    internal static bool TryCapture(out RunStateSnapshot? snapshot)
    {
        Player? player;
        lock (Gate)
        {
            player = _observedPlayer;
        }
        if (player is null)
        {
            Log.Info("[STS2-Guide] No observed player; state event skipped.");
            snapshot = null;
            return false;
        }

        try
        {
            var runState = player.RunState;
            var warnings = new List<string>();
            snapshot = new RunStateSnapshot
            {
                Character = StableIds.FromType(
                    player.Character.GetType(),
                    "Model",
                    "Character"
                ),
                Ascension = Math.Max(0, runState.AscensionLevel),
                Act = Math.Max(1, runState.CurrentActIndex + 1),
                Floor = Math.Max(0, runState.TotalFloor),
                Hp = player.Creature.CurrentHp,
                MaxHp = player.Creature.MaxHp,
                Gold = Math.Max(0, player.Gold),
                Energy = Math.Max(0, player.PlayerCombatState?.Energy ?? 3),
                Deck = ReadDeck(player, warnings),
                Relics = player.Relics
                    .Select(relic => StableIds.FromType(
                        relic.GetType(),
                        "Relic",
                        "Model"
                    ))
                    .ToList(),
                RelicStates = player.Relics
                    .Select(relic => new RelicState
                    {
                        Relic = StableIds.FromType(
                            relic.GetType(),
                            "Relic",
                            "Model"
                        ),
                        DisplayAmount = relic.ShowCounter
                            ? relic.DisplayAmount
                            : null,
                        StackCount = Math.Max(0, relic.StackCount),
                        Status = relic.Status.ToString().ToUpperInvariant(),
                    })
                    .ToList(),
                Potions = ReadPotions(player),
                MaxPotionSlots = Math.Max(0, player.MaxPotionCount),
                Modifiers = runState.Modifiers
                    .Select(modifier => StableIds.FromType(
                        modifier.GetType(),
                        "Modifier",
                        "Model"
                    ))
                    .ToList(),
                CaptureWarnings = warnings,
            };
            return true;
        }
        catch (Exception exception)
        {
            Log.Error(
                $"[STS2-Guide] State capture failed: {exception.Message}"
            );
            snapshot = null;
            return false;
        }
    }

    private static List<DeckCardState> ReadDeck(
        Player player,
        List<string> warnings
    )
    {
        var cards = new List<DeckCardState>();
        foreach (var model in player.Deck.Cards)
        {
            cards.Add(ReadDeckCard(model));
        }
        if (cards.Count == 0)
        {
            warnings.Add("deck_empty_or_unavailable");
        }
        return cards
            .GroupBy(card => new
            {
                card.Card,
                card.Upgrades,
                card.Enchantment,
                card.EnchantmentAmount,
                card.Affliction,
                card.AfflictionAmount,
            })
            .Select(group => new DeckCardState
            {
                Card = group.Key.Card,
                Upgrades = group.Key.Upgrades,
                Count = group.Count(),
                Enchantment = group.Key.Enchantment,
                EnchantmentAmount = group.Key.EnchantmentAmount,
                Affliction = group.Key.Affliction,
                AfflictionAmount = group.Key.AfflictionAmount,
            })
            .OrderBy(card => card.Card, StringComparer.Ordinal)
            .ThenBy(card => card.Upgrades)
            .ToList();
    }

    internal static DecisionOption ReadDecisionOption(CardModel model)
    {
        return new DecisionOption
        {
            Card = StableIds.FromType(
                model.GetType(),
                "Model",
                "Card"
            ),
            Upgrades = Math.Max(0, model.CurrentUpgradeLevel),
            Enchantment = model.Enchantment is null
                ? null
                : StableIds.FromType(
                    model.Enchantment.GetType(),
                    "Enchantment",
                    "Model"
                ),
            EnchantmentAmount = model.Enchantment is null
                ? null
                : Math.Max(0, model.Enchantment.Amount),
            Affliction = model.Affliction is null
                ? null
                : StableIds.FromType(
                    model.Affliction.GetType(),
                    "Affliction",
                    "Model"
                ),
            AfflictionAmount = model.Affliction is null
                ? null
                : Math.Max(0, model.Affliction.Amount),
        };
    }

    private static DeckCardState ReadDeckCard(CardModel model)
    {
        var option = ReadDecisionOption(model);
        return new DeckCardState
        {
            Card = option.Card,
            Count = 1,
            Upgrades = option.Upgrades,
            Enchantment = option.Enchantment,
            EnchantmentAmount = option.EnchantmentAmount,
            Affliction = option.Affliction,
            AfflictionAmount = option.AfflictionAmount,
        };
    }

    private static List<PotionState> ReadPotions(Player player)
    {
        var potions = new List<PotionState>();
        for (var slot = 0; slot < player.PotionSlots.Count; slot++)
        {
            var potion = player.PotionSlots[slot];
            if (potion is null)
            {
                continue;
            }
            potions.Add(new PotionState
            {
                Potion = StableIds.FromType(
                    potion.GetType(),
                    "Potion",
                    "Model"
                ),
                Slot = slot,
            });
        }
        return potions;
    }

    private static bool IsLocalPlayer(Player player)
    {
        foreach (var name in new[] {
            "IsLocalPlayer",
            "IsLocal",
            "IsControlledByLocalUser",
        })
        {
            var property = player.GetType().GetProperty(
                name,
                BindingFlags.Public
                | BindingFlags.NonPublic
                | BindingFlags.Instance
            );
            try
            {
                if (property?.PropertyType == typeof(bool)
                    && property.GetValue(player) is true)
                {
                    return true;
                }
            }
            catch
            {
                // A missing version-specific property is harmless.
            }
        }
        return false;
    }
}
