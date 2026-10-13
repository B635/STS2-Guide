using System.Globalization;
using STS2Guide.ReadOnlyExporter;

internal static class ScoreTextTests
{
    internal static void Run()
    {
        var cases = new (double? Score, string Expected)[]
        {
            (46.1, "46.1"), (46.0, "46.0"), (0, "0.0"), (100, "100.0"), (57.25, "57.3"), (57.24, "57.2"),
            (null, "--"), (double.NaN, "--"), (double.PositiveInfinity, "--"), (-0.1, "--"), (100.1, "--"),
        };
        foreach (var (score, expected) in cases)
            if (ScoreText.Format(score) != expected)
                throw new Exception($"Score {score}: expected {expected}, got {ScoreText.Format(score)}");
        // A comma-decimal system locale must not change what the drawer shows.
        var previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("de-DE");
            if (ScoreText.Format(46.1) != "46.1") throw new Exception("Score text depends on the system locale.");
        }
        finally { CultureInfo.CurrentCulture = previous; }
        if (!ScoreText.IsValid(50) || ScoreText.IsValid(null) || ScoreText.IsValid(101))
            throw new Exception("Score validity changed.");
    }
}
