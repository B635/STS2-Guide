using System.Reflection;
using System.Text.Json;
using STS2Guide.ReadOnlyExporter;

internal static class DeckEditTargetTests
{
    internal static void Run(Assembly game)
    {
        var rows = new[] { ("STRIKE_IRONCLAD", 0, true), ("WOUND", 0, false),
                           ("STRIKE_IRONCLAD", 0, true), ("STRIKE_IRONCLAD", 1, true) };
        var result = DeckEditTargetReader.Build(rows, "upgrade")!;
        if (result.Targets.Count != 3 || result.Targets.Select(t => t.TargetId).Distinct().Count() != 3
            || result.Targets[2].Upgrades != 1)
            throw new Exception("Production target projection lost eligibility or instance identity.");
        if (DeckEditTargetReader.Build(Array.Empty<(string, int, bool)>(), "remove")?.Targets.Count != 0
            || DeckEditTargetReader.Build(rows, "transform") is not null
            || DeckEditTargetReader.Build(new[] { ("", 0, true) }, "remove") is not null
            || DeckEditTargetReader.Build(new[] { ("CARD", 10, true) }, "remove") is not null
            || DeckEditTargetReader.Build(Enumerable.Repeat(("CARD", 0, true), 201), "remove") is not null)
            throw new Exception("Production target projection must distinguish empty/invalid/oversized sets.");
        using var json = JsonDocument.Parse(JsonSerializer.Serialize(result));
        if (json.RootElement.GetProperty("targets")[0].GetProperty("card").GetString() != "STRIKE_IRONCLAD")
            throw new Exception("Production target wire shape changed.");

        const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;
        Type Type(string name) => game.GetType("MegaCrit.Sts2.Core." + name, true)!;
        void Method(string type, string name, params Type[] parameters)
        {
            if (Type(type).GetMethod(name, flags, parameters) is null)
                throw new Exception($"Current game callback missing: {type}.{name}");
        }
        var card = Type("Models.CardModel");
        if (card.GetProperty("CurrentUpgradeLevel", flags)?.PropertyType != typeof(int))
            throw new Exception("Current game card upgrade getter missing.");
        foreach (var name in new[] { "IsUpgradable", "IsRemovable" })
            if (card.GetProperty(name, flags)?.PropertyType != typeof(bool))
                throw new Exception("Current game target getter missing: " + name);
        if (Type("Entities.RestSite.SmithRestSiteOption").GetProperty("SmithCount", flags)?.PropertyType != typeof(int))
            throw new Exception("Smith count contract changed.");
        foreach (var name in new[] { "Open", "Close", "OnCardRemovalUsed", "_ExitTree" })
            Method("Nodes.Screens.Shops.NMerchantInventory", name);
        Method("Entities.Merchant.MerchantEntry", "InvokePurchaseCompleted", Type("Entities.Merchant.MerchantEntry"));
        Method("Nodes.Rooms.NRestSiteRoom", "OnAfterPlayerSelectedRestSiteOption",
               Type("Entities.RestSite.RestSiteOption"), typeof(bool), typeof(ulong));
    }
}
