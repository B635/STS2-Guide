using System.Text.Json.Serialization;

namespace STS2Guide.ReadOnlyExporter;

internal sealed class StateEvent
{
    [JsonPropertyName("schema_version")]
    public int SchemaVersion { get; init; } = 4;

    [JsonPropertyName("event_id")]
    public required string EventId { get; init; }

    [JsonPropertyName("event_type")]
    public required string EventType { get; init; }

    [JsonPropertyName("emitted_at")]
    public required string EmittedAt { get; init; }

    [JsonPropertyName("source")]
    public string Source { get; init; } = "sts2-guide-readonly-mod";

    [JsonPropertyName("game_version")]
    public string? GameVersion { get; init; }

    [JsonPropertyName("run_id")]
    public required string RunId { get; init; }

    [JsonPropertyName("sequence")]
    public required long Sequence { get; init; }

    [JsonPropertyName("state")]
    public required RunStateSnapshot State { get; init; }

    [JsonPropertyName("options")]
    public required List<DecisionOption> Options { get; init; }

    [JsonPropertyName("decision")]
    public DecisionContext? Decision { get; init; }

    [JsonPropertyName("parent_event_id")]
    public string? ParentEventId { get; init; }

    [JsonPropertyName("outcome")]
    public DecisionOutcome? Outcome { get; init; }

    [JsonPropertyName("map_context")]
    public MapChoiceContext? MapContext { get; init; }

    [JsonPropertyName("run_result")]
    public RunResult? RunResult { get; init; }
}

internal sealed class RunStateSnapshot
{
    [JsonPropertyName("character")]
    public required string Character { get; init; }

    [JsonPropertyName("ascension")]
    public required int Ascension { get; init; }

    [JsonPropertyName("act")]
    public required int Act { get; init; }

    [JsonPropertyName("floor")]
    public required int Floor { get; init; }

    [JsonPropertyName("hp")]
    public int? Hp { get; init; }

    [JsonPropertyName("max_hp")]
    public int? MaxHp { get; init; }

    [JsonPropertyName("gold")]
    public int? Gold { get; init; }

    [JsonPropertyName("energy")]
    public required int Energy { get; init; }

    [JsonPropertyName("deck")]
    public required List<DeckCardState> Deck { get; init; }

    [JsonPropertyName("relics")]
    public required List<string> Relics { get; init; }

    [JsonPropertyName("relic_states")]
    public required List<RelicState> RelicStates { get; init; }

    [JsonPropertyName("potions")]
    public required List<PotionState> Potions { get; init; }

    [JsonPropertyName("max_potion_slots")]
    public int? MaxPotionSlots { get; init; }

    [JsonPropertyName("modifiers")]
    public required List<string> Modifiers { get; init; }

    [JsonPropertyName("capture_warnings")]
    public required List<string> CaptureWarnings { get; init; }
}

internal sealed class DeckCardState
{
    [JsonPropertyName("card")]
    public required string Card { get; init; }

    [JsonPropertyName("count")]
    public required int Count { get; init; }

    [JsonPropertyName("upgrades")]
    public required int Upgrades { get; init; }

    [JsonPropertyName("enchantment")]
    public string? Enchantment { get; init; }

    [JsonPropertyName("enchantment_amount")]
    public int? EnchantmentAmount { get; init; }

    [JsonPropertyName("affliction")]
    public string? Affliction { get; init; }

    [JsonPropertyName("affliction_amount")]
    public int? AfflictionAmount { get; init; }
}

internal sealed class DecisionOption
{
    [JsonPropertyName("card")]
    public required string Card { get; init; }

    [JsonPropertyName("upgrades")]
    public required int Upgrades { get; init; }

    [JsonPropertyName("enchantment")]
    public string? Enchantment { get; init; }

    [JsonPropertyName("enchantment_amount")]
    public int? EnchantmentAmount { get; init; }

    [JsonPropertyName("affliction")]
    public string? Affliction { get; init; }

    [JsonPropertyName("affliction_amount")]
    public int? AfflictionAmount { get; init; }
}

internal sealed class RelicState
{
    [JsonPropertyName("relic")]
    public required string Relic { get; init; }

    [JsonPropertyName("display_amount")]
    public int? DisplayAmount { get; init; }

    [JsonPropertyName("stack_count")]
    public required int StackCount { get; init; }

    [JsonPropertyName("status")]
    public string? Status { get; init; }
}

internal sealed class PotionState
{
    [JsonPropertyName("potion")]
    public required string Potion { get; init; }

    [JsonPropertyName("slot")]
    public required int Slot { get; init; }
}

internal sealed class DecisionContext
{
    [JsonPropertyName("can_skip")]
    public required bool CanSkip { get; init; }

    [JsonPropertyName("can_reroll")]
    public required bool CanReroll { get; init; }

    [JsonPropertyName("reward_source")]
    public string? RewardSource { get; init; }
}

internal sealed class DecisionOutcome
{
    [JsonPropertyName("kind")]
    public required string Kind { get; init; }

    [JsonPropertyName("selected_card")]
    public string? SelectedCard { get; init; }

    [JsonPropertyName("selected_option_index")]
    public int? SelectedOptionIndex { get; init; }
}

internal sealed class RunResult
{
    [JsonPropertyName("outcome")]
    public required string Outcome { get; init; }

    [JsonPropertyName("final_score")]
    public int? FinalScore { get; init; }

    [JsonPropertyName("started_at")]
    public string? StartedAt { get; init; }

    [JsonPropertyName("ended_at")]
    public required string EndedAt { get; init; }
}

internal sealed class MapNodeState
{
    [JsonPropertyName("node_id")]
    public required string NodeId { get; init; }

    [JsonPropertyName("kind")]
    public required string Kind { get; init; }

    [JsonPropertyName("row")]
    public int Row { get; init; }

    [JsonPropertyName("col")]
    public int Col { get; init; }

    [JsonPropertyName("edges")]
    public required List<string> Edges { get; init; }

    [JsonPropertyName("label")]
    public string? Label { get; init; }
}

internal sealed class MapChoiceContext
{
    [JsonPropertyName("map_name")]
    public string? MapName { get; init; }

    [JsonPropertyName("player_row")]
    public int? PlayerRow { get; init; }

    [JsonPropertyName("nodes")]
    public required List<MapNodeState> Nodes { get; init; }

    [JsonPropertyName("node_count")]
    public int NodeCount { get; init; }

    [JsonPropertyName("current_node_id")]
    public string? CurrentNodeId { get; init; }

    [JsonPropertyName("available_next_node_ids")]
    public required List<string> AvailableNextNodeIds { get; init; }

    [JsonPropertyName("boss_node_ids")]
    public required List<string> BossNodeIds { get; init; }

    [JsonPropertyName("boss_encounter_ids")]
    public required List<string> BossEncounterIds { get; init; }
}
