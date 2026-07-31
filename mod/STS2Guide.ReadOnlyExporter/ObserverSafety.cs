using MegaCrit.Sts2.Core.Logging;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// A Harmony observation must never make the original game call fail.
/// Every new Postfix enters through this boundary.
/// </summary>
internal static class ObserverSafety
{
    internal static void Run(string context, Action observation)
    {
        try
        {
            observation();
        }
        catch (Exception exception)
        {
            try
            {
                Log.Error(
                    $"[STS2-Guide] {context} observation failed: "
                    + exception.Message
                );
            }
            catch
            {
                // Logging must not turn a read-only observer failure into a
                // gameplay failure either.
            }
        }
    }
}
