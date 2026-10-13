using System.Globalization;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>How a recommendation score is written in the drawer: one decimal, culture-independent.</summary>
internal static class ScoreText
{
    internal const string Missing = "--";

    internal static bool IsValid(double? score) =>
        score.HasValue && double.IsFinite(score.Value) && score.Value is >= 0 and <= 100;

    /// <summary>One decimal separates the small differences one card makes to a whole deck;
    /// it does not claim more precision. The dot is fixed so the text is the same on every system locale.</summary>
    internal static string Format(double? score) =>
        IsValid(score) ? score!.Value.ToString("0.0", CultureInfo.InvariantCulture) : Missing;
}
