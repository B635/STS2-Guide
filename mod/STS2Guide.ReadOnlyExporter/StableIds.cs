using MegaCrit.Sts2.Core.Models;

namespace STS2Guide.ReadOnlyExporter;

internal static class StableIds
{
    internal static string FromType(Type type, params string[] legacySuffixes)
    {
        // The game owns identity, including meaningful suffixes such as POTION.
        // Keep the call signature for existing observers; suffixes no longer alter IDs.
        return ModelDb.GetEntry(type);
    }
}
