using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using STS2Guide.ReadOnlyExporter;

// Pure managed validation against the local assembly. No Godot host or game is started.
var directory = Path.GetFullPath(args[0]);
AdviceAvailabilityTests.Run();
AdviceReasonTests.Run();
BackendLivenessTests.Run();
AssemblyLoadContext.Default.Resolving += (context, name) =>
{
    var path = Path.Combine(directory, name.Name + ".dll");
    return File.Exists(path) ? context.LoadFromAssemblyPath(path) : null;
};
var game = AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.Combine(directory, "sts2.dll"));
DeckEditTargetTests.Run(game);
VerifiedEventEffectTests.Run(game);
var cases = new Dictionary<string, string>
{
    ["Potions.SpeedPotion"] = "SPEED_POTION",
    ["Potions.FairyInABottle"] = "FAIRY_IN_A_BOTTLE",
    ["Cards.ExpectAFight"] = "EXPECT_A_FIGHT",
    ["Cards.IAmInvincible"] = "I_AM_INVINCIBLE",
};
foreach (var (name, expected) in cases)
{
    var type = game.GetType("MegaCrit.Sts2.Core.Models." + name, true)!;
    var actual = StableIds.FromType(type, "Potion", "Card", "Model");
    if (actual != expected)
        throw new Exception($"Production identity {name}: expected {expected}, got {actual}");
}
var model = game.GetType("MegaCrit.Sts2.Core.Models.AbstractModel", true)!;
var entries = game.GetTypes()
    .Where(type => !type.IsAbstract && !type.IsNested && model.IsAssignableFrom(type))
    .Where(type => new[] { "Cards", "Potions", "Relics", "Characters" }
        .Any(kind => type.Namespace == "MegaCrit.Sts2.Core.Models." + kind))
    .Select(type => new
    {
        kind = type.Namespace!.Split('.').Last().ToLowerInvariant(),
        type = type.Name,
        id = StableIds.FromType(type, "Potion", "Card", "Relic", "Character", "Model"),
    }).ToArray();
Console.WriteLine(JsonSerializer.Serialize(entries));
