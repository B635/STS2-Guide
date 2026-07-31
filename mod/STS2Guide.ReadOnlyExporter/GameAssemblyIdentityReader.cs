using System.Security.Cryptography;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Runs;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// Reads the identity of the exact sts2 assembly loaded by the game process.
/// The Mod reports this fact only; the Host owns compatibility decisions.
/// </summary>
internal static class GameAssemblyIdentityReader
{
    private static readonly Lazy<string?> CachedSha256 = new(ReadOnce);

    internal static string? ReadSha256()
    {
        return CachedSha256.Value;
    }

    private static string? ReadOnce()
    {
        try
        {
            var assembly = typeof(RunManager).Assembly;
            var location = assembly.Location;
            if (string.IsNullOrWhiteSpace(location) || !File.Exists(location))
            {
                Log.Error(
                    "[STS2-Guide] Loaded sts2 assembly has no readable "
                    + "location; compatibility identity is unavailable."
                );
                return null;
            }

            using var stream = new FileStream(
                location,
                FileMode.Open,
                FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete
            );
            return Convert.ToHexString(SHA256.HashData(stream))
                .ToLowerInvariant();
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Loaded sts2 assembly SHA-256 could not be "
                + "read: "
                + exception.Message
            );
            return null;
        }
    }
}
