using System.Text.Json;

namespace STS2Guide.ReadOnlyExporter;

internal enum AdviceAvailability { Ready, Waiting, DataMissing, Unavailable }

/// <summary>Presentation only; callers must validate identity and the full advice first.</summary>
internal static class AdviceAvailabilityText
{
    internal static string Message(AdviceAvailability state) => state switch
    {
        AdviceAvailability.Ready => "",
        AdviceAvailability.Waiting => "等待当前建议",
        AdviceAvailability.DataMissing => "评分所需数据不足",
        _ => "建议暂不可用",
    };

    internal static AdviceAvailability FromValidatedRecommendation(JsonElement recommendation)
    {
        var candidates = recommendation.GetProperty("candidates").EnumerateArray()
            .Where(candidate => !candidate.TryGetProperty("eligible", out var eligible)
                || eligible.ValueKind != JsonValueKind.False).ToArray();
        if (candidates.Length > 0 && candidates.All(candidate => candidate.GetProperty("score").ValueKind == JsonValueKind.Number))
            return AdviceAvailability.Ready;
        var hasGap = HasBlockingDataGap(recommendation) || candidates.Any(candidate =>
            candidate.GetProperty("score").ValueKind == JsonValueKind.Null
            && (HasBlockingDataGap(candidate) || (candidate.TryGetProperty("data_gaps", out var gaps)
                && gaps.ValueKind == JsonValueKind.Array
                && gaps.EnumerateArray().Any(gap => gap.GetString()?.StartsWith(
                    "unknown_card:", StringComparison.Ordinal) == true))));
        return hasGap ? AdviceAvailability.DataMissing : AdviceAvailability.Unavailable;
    }

    private static bool HasBlockingDataGap(JsonElement value) =>
        value.TryGetProperty("data_gaps", out var gaps)
        && gaps.ValueKind == JsonValueKind.Array
        && gaps.EnumerateArray().Any(gap => gap.GetString() is string code
            && (code is "hp_missing" or "max_hp_missing" or "map_context_missing"
                or "map_nodes_missing" or "origin_missing" or "route_candidates_missing"
                or "boss_nodes_missing"
                || code.StartsWith("risk_profile_missing:", StringComparison.Ordinal)));
}
