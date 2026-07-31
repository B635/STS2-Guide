using HarmonyLib;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;

namespace STS2Guide.RouteLiveProbe;

[ModInitializer("Initialize")]
public static class ModEntry
{
    public static void Initialize()
    {
        ProbeRecorder.Initialize();
        var harmony = new Harmony("sts2-guide.dev.route-live-probe");
        harmony.PatchAll(typeof(ModEntry).Assembly);
        Log.Info(
            "[STS2-Guide Route Probe] Development-only read-only probe "
            + "initialized. It records observations and never selects nodes."
        );
    }
}
