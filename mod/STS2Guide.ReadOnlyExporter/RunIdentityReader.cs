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

            // Both inputs must belong to the active RunManager. History is an
            // ended-run record and can still describe the previous run while
            // the first actionable map is already open.
            if (string.IsNullOrWhiteSpace(currentRunSeed))
            {
                return Temporary(
                    "Current RunState seed is not available at this "
                    + "observation point"
                );
            }
            if (StartTimeField?.GetValue(manager) is not long internalStartTime
                || internalStartTime <= 0)
            {
                return Temporary(
                    "Current RunManager start time is not available at this "
                    + "observation point"
                );
            }

            var startTime = internalStartTime;
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
