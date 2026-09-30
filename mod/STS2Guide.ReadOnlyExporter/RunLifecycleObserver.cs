using HarmonyLib;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Runs;

namespace STS2Guide.ReadOnlyExporter;

[HarmonyPatch(typeof(RunManager), nameof(RunManager.CleanUp))]
internal static class RunObservationClosedPatch
{
    [HarmonyPostfix]
    internal static void AfterCleanUp()
    {
        // CleanUp also occurs on save-to-menu, not only on run completion.
        ObserverSafety.Run("run.observation_closed", StateEventWriter.EmitRunObservationClosed);
        ObserverSafety.Run("run.card_hide", () => CardRewardAdvicePanel.Hide());
        ObserverSafety.Run("run.route_hide", () => RouteAdviceController.Hide());
    }
}

[HarmonyPatch(typeof(NMainMenu), nameof(NMainMenu._Ready))]
internal static class MainMenuObservationClosedPatch
{
    [HarmonyPostfix]
    internal static void AfterReady()
        // Idempotent retry if the cleanup callback could not commit to disk.
        => ObserverSafety.Run("menu.observation_closed", StateEventWriter.EmitRunObservationClosed);
}

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

[HarmonyPatch(typeof(NMainMenu), nameof(NMainMenu.AbandonRun))]
internal static class SavedRunAbandonedObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterSavedRunAbandon()
    {
        try
        {
            StateEventWriter.EmitSavedRunEnded("abandon");
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Saved run abandon observation failed: "
                + exception.Message
            );
        }
        finally
        {
            EventDecisionObserver.ResetRunLifecycle();
        }
    }
}
