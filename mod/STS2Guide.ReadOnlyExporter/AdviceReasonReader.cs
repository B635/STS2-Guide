using System.Text.Json;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>Presentation projection only, after the caller's full identity/contract validation.</summary>
internal static class AdviceReasonReader
{
    internal static IReadOnlyList<string> Read(JsonElement candidate)
    {
        if (candidate.ValueKind != JsonValueKind.Object
            || !candidate.TryGetProperty("score", out var score)
            || score.ValueKind != JsonValueKind.Number || !score.TryGetDouble(out var value)
            || !double.IsFinite(value) || value is < 0 or > 100
            || !candidate.TryGetProperty("factors", out var factors)
            || factors.ValueKind != JsonValueKind.Array)
            return [];
        var reasons = new List<(bool Limitation, double Weight, int Index, string Message)>();
        var index = 0;
        foreach (var factor in factors.EnumerateArray())
        {
            if (factor.ValueKind == JsonValueKind.Object
                && factor.TryGetProperty("message", out var message)
                && message.ValueKind == JsonValueKind.String
                && !string.IsNullOrWhiteSpace(message.GetString())
                && factor.TryGetProperty("delta", out var delta)
                && delta.ValueKind == JsonValueKind.Number && delta.TryGetDouble(out var numeric)
                && double.IsFinite(numeric))
            {
                var limitation = numeric == 0
                    && factor.TryGetProperty("code", out var code)
                    && code.ValueKind == JsonValueKind.String
                    && code.GetString() is "unmodeled_mechanic" or "estimate_scope";
                if (numeric != 0 || limitation)
                    reasons.Add((limitation, Math.Abs(numeric), index, message.GetString()!.Trim()));
            }
            index++;
        }
        return reasons.OrderByDescending(reason => reason.Limitation)
            .ThenByDescending(reason => reason.Weight).ThenBy(reason => reason.Index)
            .Select(reason => reason.Message.Length > 240 ? reason.Message[..240] + "…" : reason.Message)
            .Distinct(StringComparer.Ordinal).Take(3).ToArray();
    }
}
