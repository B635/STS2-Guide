using HarmonyLib;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;

namespace STS2Guide.ReadOnlyExporter;

[ModInitializer("Initialize")]
public static class ModEntry
{
    public static void Initialize()
    {
        var harmony = new Harmony("sts2-guide.readonly-exporter");
        harmony.PatchAll(typeof(ModEntry).Assembly);
        Log.Info(
            "[STS2-Guide] Read-only exporter initialized. "
            + "Only Harmony postfix observers are registered."
        );
    }
}
