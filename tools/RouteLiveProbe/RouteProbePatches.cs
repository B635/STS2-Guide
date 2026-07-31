using HarmonyLib;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Runs;

namespace STS2Guide.RouteLiveProbe;

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.Initialize))]
internal static class MapScreenInitializedProbePatch
{
    [HarmonyPostfix]
    internal static void AfterInitialize(NMapScreen __instance, RunState runState)
    {
        ProbeRecorder.Capture(
            "map_screen_initialize_postfix",
            __instance,
            new Dictionary<string, object?>
            {
                ["run_runtime_object_id"] = ProbeRecorder.RuntimeObjectId(runState),
            }
        );
    }
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.SetMap))]
internal static class MapScreenSetMapProbePatch
{
    [HarmonyPostfix]
    internal static void AfterSetMap(NMapScreen __instance, bool clearDrawings)
    {
        ProbeRecorder.Capture(
            "map_screen_set_map_postfix",
            __instance,
            new Dictionary<string, object?>
            {
                ["clear_drawings"] = clearDrawings,
            }
        );
    }
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.Open))]
internal static class MapScreenOpenedProbePatch
{
    [HarmonyPostfix]
    internal static void AfterOpen(
        NMapScreen __instance,
        bool isOpenedFromTopBar
    )
    {
        ProbeRecorder.Capture(
            "map_open_postfix",
            __instance,
            new Dictionary<string, object?>
            {
                ["is_opened_from_top_bar"] = isOpenedFromTopBar,
            }
        );
    }
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.SetTravelEnabled))]
internal static class MapTravelEnabledProbePatch
{
    [HarmonyPostfix]
    internal static void AfterSetTravelEnabled(
        NMapScreen __instance,
        bool enabled
    )
    {
        ProbeRecorder.Capture(
            "map_set_travel_enabled_postfix",
            __instance,
            new Dictionary<string, object?>
            {
                ["requested_enabled"] = enabled,
            }
        );
    }
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.OnMapPointSelectedLocally))]
internal static class MapPointSelectedProbePatch
{
    [HarmonyPostfix]
    internal static void AfterMapPointSelected(
        NMapScreen __instance,
        NMapPoint point
    )
    {
        ProbeRecorder.CaptureSelection(__instance, point);
    }
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.Close))]
internal static class MapScreenClosedProbePatch
{
    [HarmonyPostfix]
    internal static void AfterClose(NMapScreen __instance, bool animateOut)
    {
        ProbeRecorder.Capture(
            "map_close_postfix",
            __instance,
            new Dictionary<string, object?>
            {
                ["animate_out"] = animateOut,
            }
        );
    }
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.CleanUp))]
internal static class MapScreenCleanedUpProbePatch
{
    [HarmonyPostfix]
    internal static void AfterCleanUp(NMapScreen __instance)
    {
        ProbeRecorder.Capture("map_cleanup_postfix", __instance);
    }
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen._ExitTree))]
internal static class MapScreenExitedTreeProbePatch
{
    [HarmonyPostfix]
    internal static void AfterExitTree(NMapScreen __instance)
    {
        ProbeRecorder.Capture("map_exit_tree_postfix", __instance);
    }
}

[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen._Process))]
internal static class MapScreenProcessProbePatch
{
    [HarmonyPostfix]
    internal static void AfterProcess(NMapScreen __instance)
    {
        ProbeRecorder.SampleIfChanged(__instance);
    }
}

[HarmonyPatch(typeof(RunManager), nameof(RunManager.SetUpNewSingleplayer))]
internal static class NewSingleplayerSetupProbePatch
{
    [HarmonyPostfix]
    internal static void AfterSetUpNewSingleplayer(RunState state)
    {
        ProbeRecorder.CaptureRunLifecycle(
            "new_singleplayer_setup_postfix",
            state
        );
    }
}

[HarmonyPatch(typeof(RunManager), nameof(RunManager.SetUpSavedSingleplayer))]
internal static class SavedSingleplayerSetupProbePatch
{
    [HarmonyPostfix]
    internal static void AfterSetUpSavedSingleplayer(
        RunState state,
        Task __result
    )
    {
        ProbeRecorder.CaptureRunLifecycle(
            "saved_singleplayer_setup_postfix",
            state,
            new Dictionary<string, object?>
            {
                ["returned_task_status"] = __result.Status.ToString(),
            }
        );
    }
}

[HarmonyPatch(typeof(RunManager), nameof(RunManager.Launch))]
internal static class RunLaunchedProbePatch
{
    [HarmonyPostfix]
    internal static void AfterLaunch(RunState __result)
    {
        ProbeRecorder.CaptureRunLifecycle("run_launch_postfix", __result);
    }
}

[HarmonyPatch(typeof(RunManager), nameof(RunManager.CleanUp))]
internal static class RunCleanedUpProbePatch
{
    [HarmonyPostfix]
    internal static void AfterRunCleanUp(bool graceful)
    {
        ProbeRecorder.CaptureRunLifecycle(
            "run_cleanup_postfix",
            null,
            new Dictionary<string, object?>
            {
                ["graceful"] = graceful,
            }
        );
    }
}
