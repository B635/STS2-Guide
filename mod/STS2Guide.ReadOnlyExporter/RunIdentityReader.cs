using System.Security.Cryptography;
using System.Reflection;
using System.Text;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Runs;

namespace STS2Guide.ReadOnlyExporter;

internal sealed record RunIdentity(
    string RunId,
    string StartedAt,
    bool IsStable
);

internal static class RunIdentityReader
{
    private static readonly FieldInfo? StartTimeField =
        typeof(RunManager).GetField(
            "_startTime",
            BindingFlags.NonPublic | BindingFlags.Instance
        );

    internal static RunIdentity Read()
    {
        try
        {
            var manager = RunManager.Instance;
            var history = manager.History;
            var seed = history?.Seed;
            if (string.IsNullOrWhiteSpace(seed))
            {
                seed = RunStateReader
                    .GetObservedPlayer()?
                    .RunState
                    .Rng
                    .StringSeed;
            }
            var startTime = history?.StartTime ?? 0;
            if (startTime <= 0
                && StartTimeField?.GetValue(manager) is long internalStartTime)
            {
                startTime = internalStartTime;
            }
            if (!string.IsNullOrWhiteSpace(seed) && startTime > 0)
            {
                var identityBytes = SHA256.HashData(
                    Encoding.UTF8.GetBytes($"{seed}|{startTime}")
                );
                var digest = Convert
                    .ToHexString(identityBytes)
                    .ToLowerInvariant();
                return new RunIdentity(
                    $"sts2-{digest[..24]}",
                    DateTimeOffset
                        .FromUnixTimeSeconds(startTime)
                        .ToString("O"),
                    true
                );
            }
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Stable run identity read failed: "
                + exception.Message
            );
        }

        Log.Info(
            "[STS2-Guide] Stable seed/start_time is not available at this "
            + "observation point; using a temporary identity and retrying "
            + "when the current player state becomes available."
        );
        return new RunIdentity(
            $"temporary-{Guid.NewGuid():N}",
            DateTimeOffset.UtcNow.ToString("O"),
            false
        );
    }
}
