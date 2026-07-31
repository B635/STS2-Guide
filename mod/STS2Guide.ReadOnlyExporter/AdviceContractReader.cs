using System.Text.Json;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// Shared strict checks for the policy-neutral part of advice contract v1/v2.
/// Decision-specific readers still validate owner identity, candidate order,
/// scores and route topology.
/// </summary>
internal static class AdviceContractReader
{
    private static readonly HashSet<string> AllowedDimensions =
    [
        "immediate_power",
        "survival",
        "long_term_growth",
        "resource_efficiency",
        "deck_burden",
        "synergy",
        "route_fit",
        "data_completeness",
    ];

    internal static bool HasRecommendationMetadata(
        JsonElement recommendation)
    {
        return HasNonBlankString(recommendation, "policy_version")
            && StringIn(
                recommendation,
                "status",
                "recommend",
                "skip",
                "uncertain"
            )
            && StringIn(
                recommendation,
                "confidence",
                "low",
                "medium",
                "high"
            )
            && HasUniqueStringArray(recommendation, "data_gaps");
    }

    internal static bool HasCandidateMetadata(JsonElement candidate)
    {
        if (!candidate.TryGetProperty("rank", out var rank)
            || (rank.ValueKind != JsonValueKind.Null
                && (!rank.TryGetInt32(out var rankValue)
                    || rankValue < 1))
            || !candidate.TryGetProperty("factors", out var factors)
            || factors.ValueKind != JsonValueKind.Array
            || factors.EnumerateArray().Any(
                factor => factor.ValueKind != JsonValueKind.Object
            )
            || !candidate.TryGetProperty("dimensions", out var dimensions)
            || dimensions.ValueKind != JsonValueKind.Object
            || !HasValidDimensions(dimensions)
            || !HasUniqueStringArray(candidate, "data_gaps"))
        {
            return false;
        }
        return true;
    }

    internal static bool StatusMatchesRecommendation(
        JsonElement recommendation,
        string? recommendedCandidateId)
    {
        if (!recommendation.TryGetProperty("status", out var status)
            || status.ValueKind != JsonValueKind.String)
        {
            return false;
        }
        return status.GetString() switch
        {
            "uncertain" => recommendedCandidateId is null,
            "recommend" => recommendedCandidateId is not null
                && recommendedCandidateId != "skip",
            "skip" => recommendedCandidateId == "skip",
            _ => false,
        };
    }

    private static bool HasValidDimensions(JsonElement dimensions)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in dimensions.EnumerateObject())
        {
            if (!names.Add(property.Name)
                || !AllowedDimensions.Contains(property.Name))
            {
                return false;
            }
            if (property.Value.ValueKind == JsonValueKind.Null)
            {
                continue;
            }
            if (property.Value.ValueKind != JsonValueKind.Number
                || !property.Value.TryGetDouble(out var numeric)
                || !double.IsFinite(numeric))
            {
                return false;
            }
        }
        return true;
    }

    private static bool HasUniqueStringArray(
        JsonElement root,
        string property)
    {
        if (!root.TryGetProperty(property, out var values)
            || values.ValueKind != JsonValueKind.Array)
        {
            return false;
        }
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var value in values.EnumerateArray())
        {
            if (value.ValueKind != JsonValueKind.String
                || string.IsNullOrWhiteSpace(value.GetString())
                || !seen.Add(value.GetString()!))
            {
                return false;
            }
        }
        return true;
    }

    private static bool HasNonBlankString(
        JsonElement root,
        string property)
        => root.TryGetProperty(property, out var value)
            && value.ValueKind == JsonValueKind.String
            && !string.IsNullOrWhiteSpace(value.GetString());

    private static bool StringIn(
        JsonElement root,
        string property,
        params string[] expected)
        => root.TryGetProperty(property, out var value)
            && value.ValueKind == JsonValueKind.String
            && expected.Contains(value.GetString(), StringComparer.Ordinal);
}
