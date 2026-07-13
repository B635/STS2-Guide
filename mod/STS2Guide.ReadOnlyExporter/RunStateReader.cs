using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

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

    internal static void Observe(Player? player)
    {
        if (player is null) return;
        lock (Gate)
        {
            // P0 supports single-player only.  Every verified observation
            // source (CardReward.Player, Player.PopulateCombatState and
            // RunState.Players[0]) therefore refers to the one active player.
            _observedPlayer = player;
        }
    }

    internal static void Clear()
    {
        lock (Gate)
        {
            _observedPlayer = null;
        }
    }

    /// <summary>
    /// Try to observe the local player via RunManager.Instance.DebugOnlyGetState()
    /// (public API), then reads the first player from the public
    /// RunState.Players collection.  P0 supports single-player only.
    ///
    /// This works before the first combat (e.g. after Neow, when the map opens)
    /// because RunManager holds state from Launch time onward.
    ///
    /// </summary>
    internal static bool TryObserveFromRunManager()
    {
        if (GetObservedPlayer() is not null)
        {
            Log.Info("[STS2-Guide] TryObserveFromRunManager: already have observed player, skipping.");
            return true;
        }
        try
        {
            // Stage 1: RunManager.Instance
            var manager = RunManager.Instance;
            if (manager is null)
            {
                Log.Info("[STS2-Guide] TryObserveFromRunManager: RunManager.Instance is null.");
                return false;
            }
            Log.Info("[STS2-Guide] TryObserveFromRunManager: RunManager.Instance OK.");

            // Stage 2: DebugOnlyGetState() — public API (verified by STS2MCP)
            var state = manager.DebugOnlyGetState();
            if (state is null)
            {
                Log.Info("[STS2-Guide] TryObserveFromRunManager: DebugOnlyGetState() returned null.");
                return false;
            }
            Log.Info("[STS2-Guide] TryObserveFromRunManager: DebugOnlyGetState() OK. Type=" + state.GetType().FullName);

            // Stage 3: RunState.Players is a public IReadOnlyList<Player> in
            // the current game assembly.  Do not reflect or guess a
            // LocalContext namespace for the single-player P0 path.
            var player = state.Players.FirstOrDefault();
            Log.Info(
                "[STS2-Guide] TryObserveFromRunManager: Players count="
                + state.Players.Count
                + " firstPlayer="
                + (player is not null ? "found" : "null")
            );

            if (player is not null)
            {
                Observe(player);
                Log.Info("[STS2-Guide] TryObserveFromRunManager: Player observed successfully via DebugOnlyGetState.");
                return true;
            }
            Log.Info("[STS2-Guide] TryObserveFromRunManager: No player found after all stages.");
        }
        catch (Exception ex)
        {
            Log.Error(
                "[STS2-Guide] TryObserveFromRunManager: exception " + ex.GetType().Name + " — " + ex.Message);
        }
        return false;
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
            Log.Info("[STS2-Guide] TryCapture: no observed player, trying TryObserveFromRunManager...");
            if (TryObserveFromRunManager())
            {
                lock (Gate)
                {
                    player = _observedPlayer;
                }
            }
        }
        if (player is null)
        {
            Log.Info("[STS2-Guide] TryCapture: still no observed player after retry; state event skipped.");
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

}
