using System.Reflection;
using System.Text.Json;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// Reads the release manifest embedded in this exact Mod DLL. A capability
/// is observable only when both the release and that capability were enabled
/// before this DLL was built. Missing or malformed data always fails closed.
/// </summary>
internal static class ReleaseCapabilityGate
{
    private const string ResourceName =
        "STS2Guide.CompatibilityManifest";

    private static readonly Lazy<CapabilitySnapshot> Snapshot =
        new(LoadSnapshot);

    internal static bool IsEnabled(string capability)
    {
        var snapshot = Snapshot.Value;
        return snapshot.ReleaseEnabled
            && snapshot.ReleaseFingerprint
                == StateEvent.ReleaseFingerprint
            && snapshot.EnabledCapabilities.Contains(capability);
    }

    private static CapabilitySnapshot LoadSnapshot()
    {
        try
        {
            using var stream = Assembly.GetExecutingAssembly()
                .GetManifestResourceStream(ResourceName);
            if (stream is null)
            {
                return CapabilitySnapshot.Disabled;
            }
            using var document = JsonDocument.Parse(stream);
            var root = document.RootElement;
            if (!TryExactString(root, "status", out var status)
                || status != "enabled"
                || !TryExactString(
                    root,
                    "release_fingerprint",
                    out var fingerprint
                )
                || !root.TryGetProperty(
                    "capabilities",
                    out var capabilities
                )
                || capabilities.ValueKind != JsonValueKind.Object)
            {
                return CapabilitySnapshot.Disabled;
            }
            var enabled = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in capabilities.EnumerateObject())
            {
                if (property.Value.ValueKind == JsonValueKind.String
                    && property.Value.GetString() == "enabled")
                {
                    enabled.Add(property.Name);
                }
            }
            return new CapabilitySnapshot(true, fingerprint, enabled);
        }
        catch
        {
            return CapabilitySnapshot.Disabled;
        }
    }

    private static bool TryExactString(
        JsonElement root,
        string property,
        out string value)
    {
        value = "";
        if (!root.TryGetProperty(property, out var element)
            || element.ValueKind != JsonValueKind.String)
        {
            return false;
        }
        value = element.GetString() ?? "";
        return !string.IsNullOrWhiteSpace(value)
            && value == value.Trim();
    }

    private sealed record CapabilitySnapshot(
        bool ReleaseEnabled,
        string ReleaseFingerprint,
        HashSet<string> EnabledCapabilities)
    {
        internal static CapabilitySnapshot Disabled { get; } = new(
            false,
            "",
            new HashSet<string>(StringComparer.Ordinal)
        );
    }
}
