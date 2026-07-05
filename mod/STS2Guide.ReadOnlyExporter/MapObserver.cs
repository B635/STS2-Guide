using HarmonyLib;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// Captures a verified map snapshot when the player opens the map.  P0 stores
/// the graph in the active checkpoint but does not calculate route advice.
/// </summary>
[HarmonyPatch(typeof(NMapScreen), nameof(NMapScreen.Open))]
internal static class MapScreenOpenedObservationPatch
{
    [HarmonyPostfix]
    internal static void AfterMapOpened()
    {
        try
        {
            EmitMapEvent();
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Map observation failed: "
                + exception.Message
            );
        }
    }

    private static void EmitMapEvent()
    {
        if (!RunStateReader.TryCapture(out var state) || state is null)
        {
            return;
        }
        var player = RunStateReader.GetObservedPlayer();
        if (player is null)
        {
            Log.Info("[STS2-Guide] No observed player for map event.");
            return;
        }

        var snapshot = MapNodeReader.Read(player);
        if (snapshot.Nodes.Count == 0)
        {
            Log.Info(
                "[STS2-Guide] Map opened, but the verified API returned "
                + "no nodes; no guessed graph was emitted."
            );
            return;
        }
        StateEventWriter.EmitMapChoice(state, snapshot);
    }
}
