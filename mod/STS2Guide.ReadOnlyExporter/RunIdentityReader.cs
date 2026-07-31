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
            var historySeed = history?.Seed;
            string? currentRunSeed = null;
            try
            {
                currentRunSeed = manager
                    .DebugOnlyGetState()?
                    .Players
                    .FirstOrDefault()?
                    .RunState
                    .Rng
                    .StringSeed;
            }
            catch (Exception exception)
            {
                Log.Info(
                    "[STS2-Guide] Current RunState seed is not available "
                    + "yet: " + exception.Message
                );
            }

            // History is useful for the run start time, but History.Seed can
            // still describe the run that just ended.  Never promote an
            // identity to stable unless the current RunState/player supplies
            // the seed used by that identity.
            if (string.IsNullOrWhiteSpace(currentRunSeed))
            {
                return Temporary(
                    "Current RunState seed is not available at this "
                    + "observation point"
                );
            }
            if (!string.IsNullOrWhiteSpace(historySeed)
                && !string.Equals(
                    historySeed,
                    currentRunSeed,
                    StringComparison.Ordinal
                ))
            {
                Log.Info(
                    "[STS2-Guide] History seed does not match the current "
                    + "RunState seed; identity remains provisional."
                );
                return Temporary(
                    "History and current RunState seeds do not agree"
                );
            }

            var startTime = history?.StartTime ?? 0;
            if (startTime <= 0
                && StartTimeField?.GetValue(manager) is long internalStartTime)
            {
                startTime = internalStartTime;
            }
            if (startTime > 0)
            {
                var identityBytes = SHA256.HashData(
                    Encoding.UTF8.GetBytes($"{currentRunSeed}|{startTime}")
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

        return Temporary(
            "Stable current RunState seed/start_time is not available at "
            + "this observation point"
        );
    }

    private static RunIdentity Temporary(string reason)
    {
        Log.Info(
            $"[STS2-Guide] {reason}; using a temporary identity and "
            + "retrying before the first event emission."
        );
        return new RunIdentity(
            $"temporary-{Guid.NewGuid():N}",
            DateTimeOffset.UtcNow.ToString("O"),
            false
        );
    }
}
