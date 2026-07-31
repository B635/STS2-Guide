using System.Text.Json;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// Re-checks the Host's compatibility marker before any advice reaches the
/// game UI.  The Host owns the release manifest; the Mod independently
/// compares the reported live producer/game facts with what is loaded now.
/// </summary>
internal static class AdviceCompatibilityReader
{
    internal static bool MatchesCurrentRuntime(JsonElement root)
    {
        if (!root.TryGetProperty("compatibility", out var compatibility)
            || compatibility.ValueKind != JsonValueKind.Object
            || !StringEquals(compatibility, "status", "compatible")
            || !IntEquals(compatibility, "manifest_version", 2)
            || !StringEquals(
                compatibility,
                "release_fingerprint",
                StateEvent.ReleaseFingerprint
            )
            || !HasNonBlankString(compatibility, "guide_version")
            || !IntEquals(
                compatibility,
                "state_event_schema_version",
                StateEvent.CurrentSchemaVersion
            )
            || !StringEquals(
                compatibility,
                "producer_id",
                StateEvent.ProducerId
            )
            || !StringEquals(
                compatibility,
                "producer_version",
                StateEvent.ProducerVersion
            )
            || !StringEquals(
                compatibility,
                "capability_status",
                "enabled"
            )
            || !root.TryGetProperty(
                "event_type",
                out var eventType
            )
            || eventType.ValueKind != JsonValueKind.String
            || !StringEquals(
                compatibility,
                "capability",
                eventType.GetString() ?? ""
            ))
        {
            return false;
        }

        var gameVersion = GameVersionReader.Read();
        var gameAssemblySha256 =
            GameAssemblyIdentityReader.ReadSha256();
        return !string.IsNullOrWhiteSpace(gameVersion)
            && !string.IsNullOrWhiteSpace(gameAssemblySha256)
            && StringEquals(
                compatibility,
                "game_version",
                gameVersion
            )
            && StringEquals(
                compatibility,
                "game_assembly_sha256",
                gameAssemblySha256
            );
    }

    private static bool HasNonBlankString(
        JsonElement root,
        string property)
        => root.TryGetProperty(property, out var value)
            && value.ValueKind == JsonValueKind.String
            && !string.IsNullOrWhiteSpace(value.GetString());

    private static bool StringEquals(
        JsonElement root,
        string property,
        string expected)
        => root.TryGetProperty(property, out var value)
            && value.ValueKind == JsonValueKind.String
            && value.GetString() == expected;

    private static bool IntEquals(
        JsonElement root,
        string property,
        int expected)
        => root.TryGetProperty(property, out var value)
            && value.TryGetInt32(out var actual)
            && actual == expected;
}
