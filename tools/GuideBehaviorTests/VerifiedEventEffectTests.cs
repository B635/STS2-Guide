using System.Reflection;
using System.Text.Json;
using STS2Guide.ReadOnlyExporter;

internal static class VerifiedEventEffectTests
{
    internal static void Run(Assembly game)
    {
        const string pearl = "NEOW.pages.INITIAL.options.GOLDEN_PEARL";
        const string abstain = "ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN";
        const string exit = "ABYSSAL_BATHS.pages.ALL.options.EXIT_BATHS";
        void Check(bool condition, string message)
        {
            if (!condition) throw new Exception("Verified event effects: " + message);
        }
        var gold = VerifiedEventEffectReader.FromObservedFacts("NEOW", pearl, "GOLDEN_PEARL", 173m);
        Check(gold.Kind == "gold_delta" && gold.Amount == 173 && gold.Certainty == "exact",
            "use observed gold rather than a hard-coded catalog amount");
        var heal = VerifiedEventEffectReader.FromObservedFacts("ABYSSAL_BATHS", abstain, null, 12m);
        Check(heal.Kind == "hp_delta" && heal.Amount == 12, "use observed event healing");
        var leave = VerifiedEventEffectReader.FromObservedFacts("ABYSSAL_BATHS", exit, null, null);
        Check(leave.Kind == "no_op" && leave.Amount is null, "only verified exit is no-op");
        foreach (var effect in new[]
        {
            VerifiedEventEffectReader.FromObservedFacts("OTHER", pearl, "GOLDEN_PEARL", 150m),
            VerifiedEventEffectReader.FromObservedFacts("NEOW", pearl, "NUTRITIOUS_OYSTER", 150m),
            VerifiedEventEffectReader.FromObservedFacts("NEOW", pearl + ".EXTRA", "GOLDEN_PEARL", 150m),
            VerifiedEventEffectReader.FromObservedFacts("NEOW", "NEOW.pages.UNKNOWN.options.GOLDEN_PEARL", "GOLDEN_PEARL", 150m),
            VerifiedEventEffectReader.FromObservedFacts("NEOW", pearl, null, 150m),
            VerifiedEventEffectReader.FromObservedFacts("NEOW", pearl, "GOLDEN_PEARL", null),
            VerifiedEventEffectReader.FromObservedFacts("NEOW", pearl, "GOLDEN_PEARL", -1m),
            VerifiedEventEffectReader.FromObservedFacts("NEOW", pearl, "GOLDEN_PEARL", 1.5m),
            VerifiedEventEffectReader.FromObservedFacts("NEOW", pearl, "GOLDEN_PEARL", decimal.MaxValue),
            VerifiedEventEffectReader.FromObservedFacts("ABYSSAL_BATHS", abstain, "GOLDEN_PEARL", 10m),
            VerifiedEventEffectReader.FromObservedFacts("ABYSSAL_BATHS", "ABYSSAL_BATHS.pages.INITIAL.options.IMMERSE", null, 2m),
            VerifiedEventEffectReader.FromObservedFacts("ABYSSAL_BATHS", exit, null, 3m),
            VerifiedEventEffectReader.FromObservedFacts("NEOW", "NEOW.pages.INITIAL.options.NUTRITIOUS_OYSTER", "NUTRITIOUS_OYSTER", 11m),
        })
            Check(effect.Certainty == "unknown" && effect.Amount is null
                && effect.ChildDecisionType is null, "unverified option/effect must remain unknown");
        using var wire = JsonDocument.Parse(JsonSerializer.Serialize(gold));
        Check(wire.RootElement.GetProperty("amount").GetDouble() == 173
            && wire.RootElement.GetProperty("child_decision_type").ValueKind == JsonValueKind.Null,
            "effect wire shape must preserve actual amount and no invented child");

        Type Type(string name) => game.GetType("MegaCrit.Sts2.Core." + name, true)!;
        void Getter(string type, string property, Type expected)
        {
            var info = Type(type).GetProperty(property, BindingFlags.Public | BindingFlags.Instance);
            Check(info?.GetMethod?.IsPublic == true && info.PropertyType == expected,
                "current game getter changed: " + type + "." + property);
        }
        Getter("Events.EventOption", "Relic", Type("Models.RelicModel"));
        Getter("Models.EventModel", "DynamicVars", Type("Localization.DynamicVars.DynamicVarSet"));
        Getter("Models.RelicModel", "DynamicVars", Type("Localization.DynamicVars.DynamicVarSet"));
        Getter("Localization.DynamicVars.DynamicVarSet", "Gold", Type("Localization.DynamicVars.GoldVar"));
        Getter("Localization.DynamicVars.DynamicVarSet", "Heal", Type("Localization.DynamicVars.HealVar"));
        Getter("Localization.DynamicVars.DynamicVar", "BaseValue", typeof(decimal));
        Getter("Localization.DynamicVars.DynamicVar", "IntValue", typeof(int));
        Check(Type("Models.Relics.GoldenPearl").GetMethod("AfterObtained")?.ReturnType == typeof(Task),
            "verified pickup method changed");
    }
}
