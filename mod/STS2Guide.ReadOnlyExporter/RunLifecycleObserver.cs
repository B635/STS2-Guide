using HarmonyLib;
using MegaCrit.Sts2.Core.Runs;

namespace STS2Guide.ReadOnlyExporter;

[HarmonyPatch(typeof(RunManager), nameof(RunManager.Launch))]
internal static class RunStartedObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterLaunch()
    {
        StateEventWriter.BeginRun();
    }
}

[HarmonyPatch(typeof(RunManager), nameof(RunManager.OnEnded))]
internal static class RunEndedObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterRunEnded(bool isVictory)
    {
        var outcome = isVictory
            ? "win"
            : RunManager.Instance.IsAbandoned
                ? "abandon"
                : "loss";
        StateEventWriter.EmitRunEnded(outcome);
    }
}

[HarmonyPatch(typeof(RunManager), nameof(RunManager.Abandon))]
internal static class RunAbandonedObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterAbandon()
    {
        StateEventWriter.EmitRunEnded("abandon");
    }
}
