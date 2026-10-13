using System.Text.Json;
using STS2Guide.ReadOnlyExporter;

internal static class AdviceReasonTests
{
    internal static void Run()
    {
        string[] Read(string json)
        {
            using var document = JsonDocument.Parse(json);
            return AdviceReasonReader.Read(document.RootElement).ToArray();
        }
        foreach (var json in new[] { "null", "{}", "{\"score\":null,\"factors\":[]}",
            "{\"score\":101,\"factors\":[]}", "{\"score\":50,\"factors\":null}" })
            if (Read(json).Length != 0) throw new Exception("Invalid candidate produced reasons.");
        var result = Read("""
            {"score":53,"factors":[
            {"message":"zero","delta":0},{"message":" bad ","delta":"5"},
            {"message":" first ","delta":-8},{"message":"second","delta":8},
            {"message":"first","delta":7},{"message":"third","delta":1},
            {"message":"fourth","delta":0.1},{"message":" ","delta":99},
            {"message":null,"delta":99},{"message":"overflow","delta":1e400},null]}
            """);
        if (!result.SequenceEqual(new[] { "first", "second", "third" }))
            throw new Exception("Reason sorting/dedup/invalid filtering changed.");
        if (Read("""{"score":null,"factors":[{"message":"old","delta":5}]}""").Length != 0)
            throw new Exception("Unscored candidate retained stale reasons.");
        var large = Read(JsonSerializer.Serialize(new { score = 50, factors = new[] {
            new { message = new string('a', 1000), delta = 5 } } }));
        if (large.Single().Length != 241) throw new Exception("Reason output must be bounded.");
        var limitation = Read("""
            {"score":47,"factors":[
            {"message":"prior","delta":-5},{"message":"synergy","delta":2},
            {"message":"damage","delta":-1.65},{"message":"ignored zero","delta":0},
            {"code":"unmodeled_mechanic","message":"unmodeled strike scaling","delta":0},
            {"code":"unmodeled_mechanic","message":"unmodeled strike scaling","delta":0}]}
            """);
        if (!limitation.SequenceEqual(new[] { "unmodeled strike scaling", "prior", "synergy" }))
            throw new Exception("An unmodeled mechanic must be disclosed before scored reasons without adding weight.");
        if (Read("""{"score":null,"factors":[{"code":"unmodeled_mechanic","message":"stale limitation","delta":0}]}""").Length != 0)
            throw new Exception("Unscored candidate retained stale limitation.");
        var estimate = Read("""
            {"score":50,"factors":[
            {"code":"damage_efficiency","message":"neutral damage","delta":0},
            {"message":"prior","delta":-8},{"message":"synergy","delta":4},
            {"message":"coverage","delta":3},
            {"code":"estimate_scope","message":"permanent deck baseline only","delta":0},
            {"code":"estimate_scope","message":"permanent deck baseline only","delta":0}]}
            """);
        if (!estimate.SequenceEqual(new[] { "permanent deck baseline only", "prior", "synergy" }))
            throw new Exception("Estimate boundaries must remain visible even when damage efficiency is neutral.");
        // Deck fitness (v2): an unmodeled-effect note first, then the note naming the owned cards that
        // make the candidate stronger (explanation only, delta 0), then the largest capability change.
        var fitness = Read("""
            {"score":57.7,"factors":[
            {"code":"unmodeled_mechanic","message":"part of FLATTEN is not scored","delta":0},
            {"code":"partially_modeled_deck_cards","message":"one deck card is partly scored","delta":0},
            {"code":"deck_fitness:cycle","message":"cycle falls","delta":-1.35},
            {"code":"deck_fitness:defense","message":"defense falls","delta":-0.89},
            {"code":"deck_fitness:synergy","message":"Bodyguard makes FLATTEN stronger","delta":0},
            {"code":"deck_fitness:lookahead","message":"later rewards","delta":0.05}]}
            """);
        if (!fitness.SequenceEqual(new[] { "part of FLATTEN is not scored", "Bodyguard makes FLATTEN stronger", "cycle falls" }))
            throw new Exception("Deck-fitness reasons must show limits and synergy before capability changes.");
        if (Read("""{"score":null,"factors":[{"code":"estimate_scope","message":"stale estimate","delta":0}]}""").Length != 0)
            throw new Exception("Unscored candidate retained stale estimate boundary.");
    }
}
