using System.Diagnostics;
using System.Text.Json;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>Observe heartbeat progress, never trust a residual file or wall-clock age.</summary>
internal sealed class BackendLiveness
{
    internal const double TimeoutSeconds = 5;
    private string? _session;
    private long _sequence;
    private double _lastProgress;
    private bool _confirmed;

    internal void Reset()
    {
        _session = null;
        _sequence = 0;
        _confirmed = false;
    }

    internal AdviceAvailability Read(string path, string releaseFingerprint) =>
        Read(path, releaseFingerprint, (double)Stopwatch.GetTimestamp() / Stopwatch.Frequency);

    internal AdviceAvailability Read(string path, string releaseFingerprint, double now)
    {
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(path));
            var root = document.RootElement;
            var names = root.EnumerateObject().Select(property => property.Name).ToArray();
            string[] expected = ["schema_version", "session_id", "heartbeat_sequence", "release_fingerprint",
                "state", "game_running", "worker_running", "compatibility_reasons", "activation_count"];
            if (names.Length != expected.Length || names.Distinct().Count() != names.Length
                || expected.Any(name => !root.TryGetProperty(name, out _))
                || root.GetProperty("schema_version").GetInt32() != 2
                || root.GetProperty("release_fingerprint").GetString() != releaseFingerprint
                || !root.GetProperty("activation_count").TryGetInt64(out var activations) || activations < 0)
                return Reject();
            var session = root.GetProperty("session_id").GetString();
            if (session is null || session.Length != 32 || !session.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f')
                || !root.GetProperty("heartbeat_sequence").TryGetInt64(out var sequence) || sequence <= 0)
                return Reject();
            var reasons = root.GetProperty("compatibility_reasons").EnumerateArray()
                .Select(reason => reason.GetString()).ToArray();
            if (reasons.Any(string.IsNullOrWhiteSpace) || reasons.Distinct().Count() != reasons.Length
                || reasons.Length > 0 || root.GetProperty("state").GetString() != "RUNNING"
                || root.GetProperty("game_running").ValueKind is not (JsonValueKind.True or JsonValueKind.Null)
                || !root.GetProperty("worker_running").GetBoolean())
                return Reject();

            if (_session != session || sequence < _sequence)
            {
                _session = session;
                _sequence = sequence;
                _lastProgress = now;
                _confirmed = false;
            }
            else if (sequence > _sequence)
            {
                _sequence = sequence;
                _lastProgress = now;
                _confirmed = true;
            }
            if (now - _lastProgress >= TimeoutSeconds)
                return AdviceAvailability.Unavailable;
            return _confirmed ? AdviceAvailability.Ready : AdviceAvailability.Waiting;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException
            or JsonException or InvalidOperationException or KeyNotFoundException or FormatException or OverflowException)
        {
            return Reject();
        }
    }

    private AdviceAvailability Reject()
    {
        Reset();
        return AdviceAvailability.Unavailable;
    }
}
