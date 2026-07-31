namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// Pure state transition for protecting a newly launched run from a stale
/// stable identity left behind by the most recently ended run.
/// </summary>
internal static class RunIdentityGuard
{
    internal static bool ShouldCloseActiveRun(
        bool hasUnendedState,
        string currentRunId,
        bool currentIsStable,
        string candidateRunId,
        bool candidateIsStable)
    {
        return hasUnendedState
            && currentIsStable
            && candidateIsStable
            && !string.Equals(
                currentRunId,
                candidateRunId,
                StringComparison.Ordinal
            );
    }

    internal static bool ShouldRetainCurrentIdentity(
        bool hasUnendedState,
        bool currentIsStable,
        bool candidateIsStable)
    {
        return hasUnendedState
            && currentIsStable
            && !candidateIsStable;
    }

    internal static void RememberEndedStableIdentity(
        ref string? lastEndedStableRunId,
        string runId,
        bool isStable)
    {
        if (isStable)
        {
            lastEndedStableRunId = runId;
        }
    }

    internal static bool TryAcceptStableIdentity(
        ref string? lastEndedStableRunId,
        string candidateRunId,
        bool candidateIsStable)
    {
        if (!candidateIsStable)
        {
            return false;
        }
        if (!string.IsNullOrWhiteSpace(lastEndedStableRunId)
            && string.Equals(
                lastEndedStableRunId,
                candidateRunId,
                StringComparison.Ordinal
            ))
        {
            // Keep the guard armed. The current player's seed may still be
            // the ended run's stale value and can be retried later.
            return false;
        }

        // Only a genuinely different stable identity disarms the guard.
        lastEndedStableRunId = null;
        return true;
    }
}
