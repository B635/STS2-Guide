using System.Text.Json;
using STS2Guide.ReadOnlyExporter;

internal static class BackendLivenessTests
{
    internal static void Run()
    {
        var directory = Path.Combine(Path.GetTempPath(), "guide-live-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "runtime-status.json");
        var fingerprint = new string('a', 64);
        var session = new string('b', 32);
        void Publish(long sequence, string state = "RUNNING", string? release = null) =>
            File.WriteAllText(path, JsonSerializer.Serialize(new
            {
                schema_version = 2, session_id = session, heartbeat_sequence = sequence,
                release_fingerprint = release ?? fingerprint, state,
                game_running = true, worker_running = state == "RUNNING",
                compatibility_reasons = Array.Empty<string>(), activation_count = 0,
            }));
        var reader = new BackendLiveness();
        void Expect(double time, AdviceAvailability expected)
        {
            var actual = reader.Read(path, fingerprint, time);
            if (actual != expected) throw new Exception($"Liveness at {time}: expected {expected}, got {actual}");
        }
        try
        {
            Expect(0, AdviceAvailability.Unavailable);
            Publish(1);
            Expect(0, AdviceAvailability.Waiting);
            Expect(5, AdviceAvailability.Unavailable); // residual file never proves life
            Publish(2);
            Expect(5.1, AdviceAvailability.Ready);
            Expect(10, AdviceAvailability.Ready); // advice age itself is irrelevant
            Expect(10.1, AdviceAvailability.Unavailable);
            Publish(3);
            Expect(10.2, AdviceAvailability.Ready);
            session = new string('c', 32);
            Publish(1);
            Expect(10.3, AdviceAvailability.Waiting);
            Publish(2);
            Expect(10.4, AdviceAvailability.Ready);
            Publish(3, "WORKER_FAILED");
            Expect(10.5, AdviceAvailability.Unavailable);
            Publish(4);
            Expect(10.6, AdviceAvailability.Waiting);
            Publish(5);
            Expect(10.7, AdviceAvailability.Ready);
            Publish(6, release: new string('d', 64));
            Expect(10.8, AdviceAvailability.Unavailable);
            File.WriteAllText(path, "{broken");
            Expect(10.9, AdviceAvailability.Unavailable);
            File.Delete(path);
            Expect(11, AdviceAvailability.Unavailable);
        }
        finally { Directory.Delete(directory, true); }
    }
}
