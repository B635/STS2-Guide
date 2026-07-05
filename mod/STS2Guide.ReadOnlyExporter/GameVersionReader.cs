using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Logging;

namespace STS2Guide.ReadOnlyExporter;

internal static class GameVersionReader
{
    private static readonly Lazy<string?> CachedVersion = new(ReadOnce);

    internal static string? Read()
    {
        return CachedVersion.Value;
    }

    private static string? ReadOnce()
    {
        var candidates = new HashSet<string>(
            StringComparer.OrdinalIgnoreCase
        );
        AddCandidate(candidates, Path.GetDirectoryName(OS.GetExecutablePath()));
        AddCandidate(candidates, AppContext.BaseDirectory);
        AddCandidate(
            candidates,
            Directory.GetParent(AppContext.BaseDirectory)?.FullName
        );

        foreach (var directory in candidates)
        {
            var releaseInfoPath = Path.Combine(
                directory,
                "release_info.json"
            );
            if (!File.Exists(releaseInfoPath))
            {
                continue;
            }
            try
            {
                using var document = JsonDocument.Parse(
                    File.ReadAllText(releaseInfoPath)
                );
                if (document.RootElement.TryGetProperty(
                        "version",
                        out var versionElement
                    ))
                {
                    var version = versionElement.GetString()?.Trim();
                    if (!string.IsNullOrWhiteSpace(version))
                    {
                        return version.TrimStart('v', 'V');
                    }
                }
            }
            catch (Exception exception)
            {
                Log.Error(
                    "[STS2-Guide] release_info.json could not be read: "
                    + exception.Message
                );
            }
        }

        Log.Info(
            "[STS2-Guide] release_info.json was not found; "
            + "falling back to the game assembly version."
        );
        return typeof(Player).Assembly.GetName().Version?.ToString();
    }

    private static void AddCandidate(
        ISet<string> candidates,
        string? path
    )
    {
        if (!string.IsNullOrWhiteSpace(path))
        {
            candidates.Add(Path.GetFullPath(path));
        }
    }
}
