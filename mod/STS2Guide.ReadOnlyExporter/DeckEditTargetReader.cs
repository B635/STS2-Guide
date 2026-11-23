using MegaCrit.Sts2.Core.Entities.Players;

namespace STS2Guide.ReadOnlyExporter;

internal static class DeckEditTargetReader
{
    // Current game: FromDeckForRemoval uses IsRemovable with null merchant filter;
    // FromDeckForUpgrade uses IsUpgradable. No selection/action method is called.
    internal static DeckEditTargetSet? Read(Player player, string operation)
    {
        if (operation is not ("upgrade" or "remove"))
            return null;
        try
        {
            return Build(player.Deck.Cards.Select(card => (
                Card: StableIds.FromType(card.GetType(), "Model", "Card"),
                Upgrades: card.CurrentUpgradeLevel,
                Eligible: operation == "upgrade" ? card.IsUpgradable : card.IsRemovable
            )), operation);
        }
        catch
        {
            // An incomplete list must not be presented as the complete legal set.
            return null;
        }
    }

    internal static DeckEditTargetSet? Build(
        IEnumerable<(string Card, int Upgrades, bool Eligible)> cards, string operation)
    {
        if (operation is not ("upgrade" or "remove"))
            return null;
        var targets = new List<DeckEditTarget>();
        var index = 0;
        foreach (var card in cards)
        {
            var position = index++;
            if (!card.Eligible)
                continue;
            if (string.IsNullOrWhiteSpace(card.Card) || card.Card.Length > 200
                || card.Upgrades is < 0 or > 9 || targets.Count >= 200)
                return null;
            targets.Add(new DeckEditTarget
            {
                TargetId = $"{operation}:{position}:{card.Card}:{card.Upgrades}",
                Card = card.Card,
                Upgrades = card.Upgrades,
            });
        }
        return new DeckEditTargetSet
        {
            Operation = operation,
            SourceCode = operation == "upgrade"
                ? "game_api:smith_upgrade_targets" : "game_api:merchant_removal_targets",
            Targets = targets,
        };
    }
}
