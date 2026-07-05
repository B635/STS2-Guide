using System.Text.RegularExpressions;

namespace STS2Guide.ReadOnlyExporter;

internal static partial class StableIds
{
    [GeneratedRegex("([a-z0-9])([A-Z])")]
    private static partial Regex WordBoundary();

    internal static string FromType(Type type, params string[] suffixes)
    {
        var name = type.Name;
        var removed = true;
        while (removed)
        {
            removed = false;
            foreach (var suffix in suffixes)
            {
                if (name.EndsWith(suffix, StringComparison.Ordinal))
                {
                    name = name[..^suffix.Length];
                    removed = true;
                    break;
                }
            }
        }
        return WordBoundary()
            .Replace(name, "$1_$2")
            .Replace(' ', '_')
            .ToUpperInvariant();
    }
}
