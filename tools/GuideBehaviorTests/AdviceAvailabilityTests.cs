using System.Text.Json;
using STS2Guide.ReadOnlyExporter;

internal static class AdviceAvailabilityTests
{
    internal static void Run()
    {
        Check("""{"candidates":[{"score":50}],"data_gaps":["optional_missing"]}""", AdviceAvailability.Ready);
        Check("""{"candidates":[{"score":50,"eligible":true},{"score":null,"eligible":false}],"data_gaps":[]}""", AdviceAvailability.Ready);
        Check("""{"candidates":[{"score":50,"eligible":true},{"score":null,"eligible":true,"data_gaps":["unknown_card:X"]}],"data_gaps":[]}""", AdviceAvailability.DataMissing);
        Check("""{"candidates":[{"score":null,"eligible":false}],"data_gaps":[]}""", AdviceAvailability.Unavailable);
        Check("""{"candidates":[{"score":null}],"data_gaps":["risk_profile_missing:route_risk_profile_v1"]}""", AdviceAvailability.DataMissing);
        Check("""{"candidates":[{"score":null}],"data_gaps":["route_search_limit_exceeded"]}""", AdviceAvailability.Unavailable);
        Check("""{"candidates":[{"score":null,"data_gaps":["unknown_cause"]}],"data_gaps":[]}""", AdviceAvailability.Unavailable);
        Check("""{"candidates":[{"score":null}],"data_gaps":[]}""", AdviceAvailability.Unavailable);
        Check("""{"candidates":[{"score":null,"data_gaps":["unknown_card:NEW_CARD"]}],"data_gaps":[]}""", AdviceAvailability.DataMissing);
        Check("""{"candidates":[{"score":52,"data_gaps":["unknown_card:DECK_CARD"]}],"data_gaps":[]}""", AdviceAvailability.Ready);
        if (AdviceAvailabilityText.Message(AdviceAvailability.Waiting) != "等待当前建议")
            throw new Exception("Waiting must not claim backend is offline.");
        if (AdviceAvailabilityText.Message(AdviceAvailability.Unavailable) != "建议暂不可用")
            throw new Exception("Unknown cause must stay generic.");
        var directory = Path.Combine(Path.GetTempPath(), "guide-advice-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var path = Path.Combine(directory, "advice.json");
            string? Match(JsonElement root) => root.GetProperty("event_id").GetString() == "current" ? "current" : null;
            void Expect(AdviceAvailability status, bool hasValue)
            {
                var result = AdviceFileReader.Read(path, Match);
                if (result.Availability != status || (result.Value is not null) != hasValue)
                    throw new Exception($"Read path: expected {status}/{hasValue}, got {result}");
            }
            Expect(AdviceAvailability.Waiting, false);
            File.WriteAllText(path, "{bad");
            Expect(AdviceAvailability.Unavailable, false);
            File.WriteAllText(path, """{"event_id":"old"}""");
            Expect(AdviceAvailability.Unavailable, false);
            File.WriteAllText(path, """{"event_id":"current","recommendation":{"candidates":[{"score":53}],"data_gaps":[]}}""");
            Expect(AdviceAvailability.Ready, true);
            var unchangedTime = File.GetLastWriteTimeUtc(path);
            using (var locked = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
                Expect(AdviceAvailability.Unavailable, false);
            if (File.GetLastWriteTimeUtc(path) != unchangedTime)
                throw new Exception("Read-lock test must not change advice timestamp.");
            Expect(AdviceAvailability.Ready, true);
            File.WriteAllText(path, """{"event_id":"old"}""");
            Expect(AdviceAvailability.Unavailable, false);
            File.Delete(path);
            Expect(AdviceAvailability.Waiting, false);
            File.WriteAllText(path, """{"event_id":"current","recommendation":{"candidates":[{"score":null}],"data_gaps":["hp_missing"]}}""");
            Expect(AdviceAvailability.DataMissing, true);
            File.WriteAllText(path, """{"event_id":"current","recommendation":{"candidates":[{"score":71}],"data_gaps":[]}}""");
            Expect(AdviceAvailability.Ready, true);
        }
        finally { Directory.Delete(directory, true); }
    }

    private static void Check(string json, AdviceAvailability expected)
    {
        using var document = JsonDocument.Parse(json);
        var actual = AdviceAvailabilityText.FromValidatedRecommendation(document.RootElement);
        if (actual != expected)
            throw new Exception($"Advice availability: expected {expected}, got {actual}");
    }
}
