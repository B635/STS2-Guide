using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Relics;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// Narrow, version-verified observations, never an interpreter for event text.
/// Only these complete option effects were checked against the current game IL.
/// </summary>
internal static class VerifiedEventEffectReader
{
    private const string PearlKey = "NEOW.pages.INITIAL.options.GOLDEN_PEARL";
    private const string AbstainKey = "ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN";
    private const string ExitKey = "ABYSSAL_BATHS.pages.ALL.options.EXIT_BATHS";

    internal static ChoiceEffect Read(EventModel model, EventOption option)
    {
        try
        {
            // Exact types prevent another event or third-party subclass from
            // inheriting a forecast for a different callback/side effect.
            if (model.GetType() == typeof(Neow)
                && option.TextKey == PearlKey
                && option.Relic?.GetType() == typeof(GoldenPearl))
            {
                return FromObservedFacts("NEOW", option.TextKey, "GOLDEN_PEARL",
                    option.Relic.DynamicVars.Gold.BaseValue);
            }
            if (model.GetType() == typeof(AbyssalBaths))
            {
                if (option.TextKey == AbstainKey && option.Relic is null)
                    return FromObservedFacts("ABYSSAL_BATHS", option.TextKey, null,
                        model.DynamicVars.Heal.IntValue);
                if (option.TextKey == ExitKey && option.Relic is null)
                    return FromObservedFacts("ABYSSAL_BATHS", option.TextKey, null, null);
            }
        }
        catch
        {
            // A missing getter/value must not produce a partially known effect.
            return Unknown("game_api:event_effect_read_failed");
        }
        return Unknown();
    }

    // This projection is shared with managed tests; callers must supply facts
    // selected above from verified runtime model types, not labels or prose.
    internal static ChoiceEffect FromObservedFacts(
        string eventId, string optionKey, string? relicId, decimal? amount)
    {
        if (eventId == "ABYSSAL_BATHS" && optionKey == ExitKey
            && relicId is null && amount is null)
            return Exact("no_op", null, "game_api:abyssal_baths_exit");
        if (amount is null || amount < 0 || amount > int.MaxValue
            || amount != decimal.Truncate(amount.Value))
            return Unknown();
        if (eventId == "NEOW" && optionKey == PearlKey && relicId == "GOLDEN_PEARL")
            return Exact("gold_delta", (int)amount, "game_api:golden_pearl_gold");
        if (eventId == "ABYSSAL_BATHS" && optionKey == AbstainKey && relicId is null)
            return Exact("hp_delta", (int)amount, "game_api:abyssal_baths_abstain_heal");
        return Unknown();
    }

    private static ChoiceEffect Exact(string kind, int? amount, string sourceCode)
        => new()
        {
            Kind = kind, Amount = amount, TargetMode = "none",
            Certainty = "exact", SourceCode = sourceCode,
        };

    private static ChoiceEffect Unknown(string sourceCode = "game_api:event_effect_not_exposed")
        => new()
        {
            Kind = "followup_choice", TargetMode = "none",
            Certainty = "unknown", SourceCode = sourceCode,
        };
}
