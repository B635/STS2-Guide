using System.Text.Json.Serialization;

namespace STS2Guide.ReadOnlyExporter;

internal sealed class StateEvent
{
    internal const int CurrentSchemaVersion = 9;
    internal const string ProducerId = "STS2GuideReadOnlyExporter";
    internal const string ProducerVersion = "0.3.0";
    internal const string ProducerSource = "sts2-guide-readonly-mod";
    internal const string ReleaseFingerprint =
        "56f928dcd6650946d2929910f2efa9c5c0b7f962667177421dd6c2ecf2f15afd";

    [JsonPropertyName("schema_version")]
    public int SchemaVersion { get; init; } = CurrentSchemaVersion;

    [JsonPropertyName("event_id")]
    public required string EventId { get; init; }

    [JsonPropertyName("event_type")]
    public required string EventType { get; init; }

    [JsonPropertyName("emitted_at")]
    public required string EmittedAt { get; init; }

    [JsonPropertyName("source")]
    public string Source { get; init; } = ProducerSource;

    [JsonPropertyName("game_version")]
    public string? GameVersion { get; init; }

    [JsonPropertyName("snapshot_kind")]
    public string SnapshotKind { get; init; } = "complete";

    [JsonPropertyName("state_revision")]
    public required long StateRevision { get; init; }

    [JsonPropertyName("producer_id")]
    public string ProducerIdValue { get; init; } = ProducerId;

    [JsonPropertyName("producer_version")]
    public string ProducerVersionValue { get; init; } = ProducerVersion;

    [JsonPropertyName("game_assembly_sha256")]
    public string? GameAssemblySha256 { get; init; }

    [JsonPropertyName("release_fingerprint")]
    public string ReleaseFingerprintValue { get; init; } =
        ReleaseFingerprint;

    [JsonPropertyName("guide_preferences")]
    public required GuidePreferences GuidePreferences { get; init; }

    [JsonPropertyName("run_id")]
    public required string RunId { get; init; }

    [JsonPropertyName("sequence")]
    public required long Sequence { get; init; }

    [JsonPropertyName("decision_id")]
    public string? DecisionId { get; init; }

    [JsonPropertyName("state")]
    public required RunStateSnapshot State { get; init; }

    [JsonPropertyName("options")]
    public required List<DecisionOption> Options { get; init; }

    [JsonPropertyName("candidates")]
    public required List<DecisionCandidateEnvelope> Candidates { get; init; }

    [JsonPropertyName("decision")]
    public DecisionContext? Decision { get; init; }

    [JsonPropertyName("decision_parent")]
    public DecisionParentContext? DecisionParent { get; init; }

    [JsonPropertyName("parent_event_id")]
    public string? ParentEventId { get; init; }

    [JsonPropertyName("outcome")]
    public DecisionOutcome? Outcome { get; init; }

    [JsonPropertyName("map_context")]
    public MapChoiceContext? MapContext { get; init; }

    [JsonPropertyName("run_result")]
    public RunResult? RunResult { get; init; }
}

internal static class GuideRouteModes
{
    internal const string Balanced = "balanced";
    internal const string Survival = "survival";
    internal const string Growth = "growth";

    internal static bool IsValid(string? value)
        => value is Balanced or Survival or Growth;
}

internal sealed class GuidePreferences
{
    [JsonPropertyName("route_mode")]
    public required string RouteMode { get; init; }
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
    [JsonPropertyName("candidate_id")]
    public string? CandidateId { get; set; }

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

internal sealed class DecisionCandidateEnvelope
{
    [JsonPropertyName("candidate_id")]
    public required string CandidateId { get; init; }

    [JsonPropertyName("kind")]
    public required string Kind { get; init; }

    [JsonPropertyName("entity_id")]
    public string? EntityId { get; init; }

    [JsonPropertyName("label")]
    public required string Label { get; init; }

    [JsonPropertyName("eligible")]
    public required bool Eligible { get; init; }

    [JsonPropertyName("unavailable_reason")]
    public string? UnavailableReason { get; init; }

    [JsonPropertyName("costs")]
    public required List<DecisionCost> Costs { get; init; }

    [JsonPropertyName("payload")]
    public required CandidatePayload Payload { get; init; }
}

internal sealed class DecisionCost
{
    [JsonPropertyName("kind")]
    public required string Kind { get; init; }

    [JsonPropertyName("amount")]
    public required int Amount { get; init; }

    [JsonPropertyName("resource_id")]
    public string? ResourceId { get; init; }
}

internal sealed class CandidatePayload
{
    [JsonPropertyName("card")]
    public string? Card { get; init; }

    [JsonPropertyName("upgrades")]
    public int? Upgrades { get; init; }

    [JsonPropertyName("enchantment")]
    public string? Enchantment { get; init; }

    [JsonPropertyName("enchantment_amount")]
    public int? EnchantmentAmount { get; init; }

    [JsonPropertyName("affliction")]
    public string? Affliction { get; init; }

    [JsonPropertyName("affliction_amount")]
    public int? AfflictionAmount { get; init; }

    [JsonPropertyName("node_id")]
    public string? NodeId { get; init; }

    [JsonPropertyName("slot_id")]
    public string? SlotId { get; init; }

    [JsonPropertyName("offer_kind")]
    public string? OfferKind { get; init; }

    [JsonPropertyName("is_stocked")]
    public bool? IsStocked { get; init; }

    [JsonPropertyName("replacement_supported")]
    public bool? ReplacementSupported { get; init; }

    [JsonPropertyName("action_id")]
    public string? ActionId { get; init; }

    [JsonPropertyName("operation")]
    public string? Operation { get; init; }

    [JsonPropertyName("blessing_id")]
    public string? BlessingId { get; init; }

    [JsonPropertyName("stage_id")]
    public string? StageId { get; init; }

    [JsonPropertyName("event_id")]
    public string? EventId { get; init; }

    [JsonPropertyName("page_id")]
    public string? PageId { get; init; }

    [JsonPropertyName("option_id")]
    public string? OptionId { get; init; }

    [JsonPropertyName("target_candidate_ids")]
    public List<string> TargetCandidateIds { get; init; } = [];

    [JsonPropertyName("effects")]
    public List<ChoiceEffect> Effects { get; init; } = [];
}

internal sealed class ChoiceEffect
{
    [JsonPropertyName("kind")]
    public required string Kind { get; init; }

    [JsonPropertyName("amount")]
    public double? Amount { get; init; }

    [JsonPropertyName("min_amount")]
    public double? MinAmount { get; init; }

    [JsonPropertyName("max_amount")]
    public double? MaxAmount { get; init; }

    [JsonPropertyName("entity_type")]
    public string? EntityType { get; init; }

    [JsonPropertyName("entity_id")]
    public string? EntityId { get; init; }

    [JsonPropertyName("target_mode")]
    public required string TargetMode { get; init; }

    [JsonPropertyName("certainty")]
    public required string Certainty { get; init; }

    [JsonPropertyName("source_code")]
    public required string SourceCode { get; init; }

    [JsonPropertyName("child_decision_type")]
    public string? ChildDecisionType { get; init; }
}

internal sealed class DecisionParentContext
{
    [JsonPropertyName("decision_id")]
    public required string DecisionId { get; init; }

    [JsonPropertyName("candidate_id")]
    public string? CandidateId { get; init; }

    [JsonPropertyName("source_type")]
    public required string SourceType { get; init; }

    [JsonPropertyName("source_id")]
    public required string SourceId { get; init; }
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

    [JsonPropertyName("selected_candidate_id")]
    public string? SelectedCandidateId { get; init; }

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

    [JsonPropertyName("origin_node_id")]
    public string? OriginNodeId { get; init; }

    [JsonPropertyName("available_next_node_ids")]
    public required List<string> AvailableNextNodeIds { get; init; }

    [JsonPropertyName("boss_node_ids")]
    public required List<string> BossNodeIds { get; init; }

    [JsonPropertyName("boss_encounter_ids")]
    public required List<string> BossEncounterIds { get; init; }
}
