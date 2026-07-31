using HarmonyLib;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Runs;

namespace STS2Guide.ReadOnlyExporter;

[HarmonyPatch(typeof(RunManager), nameof(RunManager.Launch))]
internal static class RunStartedObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterLaunch()
    {
        try
        {
            EventDecisionObserver.ResetRunLifecycle();
            StateEventWriter.BeginRun();
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Run launch observation failed: "
                + exception.Message
            );
        }
    }
}

[HarmonyPatch(typeof(RunManager), nameof(RunManager.OnEnded))]
internal static class RunEndedObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterRunEnded(bool isVictory)
    {
        try
        {
            var outcome = isVictory
                ? "win"
                : RunManager.Instance.IsAbandoned
                    ? "abandon"
                    : "loss";
            StateEventWriter.EmitRunEnded(outcome);
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Run end observation failed: "
                + exception.Message
            );
        }
        finally
        {
            EventDecisionObserver.ResetRunLifecycle();
        }
    }
}

[HarmonyPatch(typeof(RunManager), nameof(RunManager.Abandon))]
internal static class RunAbandonedObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterAbandon()
    {
        try
        {
            StateEventWriter.EmitRunEnded("abandon");
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Run abandon observation failed: "
                + exception.Message
            );
        }
        finally
        {
            EventDecisionObserver.ResetRunLifecycle();
        }
    }
}
