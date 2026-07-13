using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace STS2Guide.ReadOnlyExporter;

internal static class StateEventWriter
{
    private static readonly object WriteGate = new();
    private static string _runId = $"temporary-{Guid.NewGuid():N}";
    private static string _runStartedAt =
        DateTimeOffset.UtcNow.ToString("O");
    private static bool _runIdentityStable;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
    };
    private static long _sequence;
    private static RunStateSnapshot? _lastState;
    private static PendingDecision? _pendingDecision;
    private static bool _ended;
    private static string? _previousRunId;
    private static bool _previousRunEnded;

    internal static PendingDecisionView? GetPendingDecision()
    {
        lock (WriteGate)
        {
            return _pendingDecision is null
                ? null
                : new PendingDecisionView(
                    _runId,
                    _pendingDecision.ParentEventId,
                    _pendingDecision.Options
                        .Select(option => option.Card)
                        .ToList()
                );
        }
    }

    internal static void BeginRun()
    {
        lock (WriteGate)
        {
            var identity = RunIdentityReader.Read();

            // If the previous run was still in progress (never ended),
            // only emit a synthetic run_ended when the NEW identity differs.
            // Same identity → save+continue resume, not a real run end.
            if (_lastState is not null
                && !_ended
                && identity.RunId != _runId)
            {
                Write(
                    "run_ended",
                    _lastState,
                    new List<DecisionOption>(),
                    runResult: CreateRunResult("abandon")
                );
                _previousRunEnded = true;
            }

            // Guard: if the new identity matches a previous stable Run ID
            // and the previous run ended, the game's History.Seed is stale.
            // Force a new temporary ID to prevent cross-run contamination.
            if (identity.IsStable
                && _previousRunEnded
                && identity.RunId == _previousRunId)
            {
                Log.Error(
                    $"[STS2-Guide] Stale History.Seed produced same Run ID "
                    + $"{identity.RunId} as previous ended run. "
                    + "Generating new temporary identity."
                );
                identity = new RunIdentity(
                    $"temporary-{Guid.NewGuid():N}",
                    DateTimeOffset.UtcNow.ToString("O"),
                    false
                );
            }

            _runId = identity.RunId;
            _runStartedAt = identity.StartedAt;
            _runIdentityStable = identity.IsStable;
            _sequence = RecoverSequence(_runId);

            if (!_previousRunEnded)
            {
                _previousRunId = _runId;
            }
            _lastState = null;
            _pendingDecision = null;
            _ended = false;
            _previousRunEnded = false;
            RunStateReader.Clear();
            Log.Info(
                $"[STS2-Guide] Run identity {_runId}; "
                + $"resuming sequence {_sequence}."
            );
        }
    }

    internal static void EmitRunEnded(string outcome)
    {
        lock (WriteGate)
        {
            if (_ended)
            {
                return;
            }
            _ended = true;
            RunStateSnapshot? state = null;
            if (!RunStateReader.TryCapture(out state) || state is null)
            {
                state = _lastState;
            }
            if (state is not null)
            {
                EnsureStableRunIdentity(state);
                Write(
                    "run_ended",
                    state,
                    new List<DecisionOption>(),
                    runResult: CreateRunResult(outcome)
                );
            }
            // Record that this run ended so BeginRun() can guard against
            // stale History.Seed reuse on the next run.
            _previousRunId = _runId;
            _previousRunEnded = true;
            _lastState = null;
            _pendingDecision = null;
            RunStateReader.Clear();
        }
    }

    internal static void EmitCardReward(
        List<DecisionOption> options,
        DecisionContext decision
    )
    {
        if (!RunStateReader.TryCapture(out var state) || state is null)
        {
            return;
        }
        lock (WriteGate)
        {
            EnsureStableRunIdentity(state);
            _lastState = state;
            var eventId = Write(
                "card_reward",
                state,
                options,
                decision: decision
            );
            if (eventId is not null)
            {
                _pendingDecision = new PendingDecision(eventId, options);
            }
        }
    }

    internal static void EmitCardSelected(CardModel card)
    {
        var selected = RunStateReader.ReadDecisionOption(card);
        lock (WriteGate)
        {
            var pending = _pendingDecision;
            var state = _lastState;
            if (pending is null || state is null)
            {
                return;
            }
            var optionIndex = pending.Options.FindIndex(option =>
                SameOption(option, selected)
            );
            if (optionIndex < 0)
            {
                optionIndex = pending.Options.FindIndex(option =>
                    option.Card == selected.Card
                );
            }
            Write(
                "decision_closed",
                state,
                new List<DecisionOption>(),
                parentEventId: pending.ParentEventId,
                outcome: new DecisionOutcome
                {
                    Kind = "selected",
                    SelectedCard = selected.Card,
                    SelectedOptionIndex = optionIndex >= 0
                        ? optionIndex
                        : null,
                }
            );
            _pendingDecision = null;
        }
    }

    internal static void EmitCardSkipped()
    {
        lock (WriteGate)
        {
            ClosePendingDecision(new DecisionOutcome
            {
                Kind = "skipped",
            });
        }
    }

    internal static void EmitDecisionClosed()
    {
        lock (WriteGate)
        {
            ClosePendingDecision(new DecisionOutcome
            {
                Kind = "closed_unknown",
            });
        }
    }

    internal static void EmitMapChoice(
        RunStateSnapshot state,
        MapSnapshot snapshot
    )
    {
        lock (WriteGate)
        {
            EnsureStableRunIdentity(state);
            _lastState = state;
            Write(
                "map_choice",
                state,
                new List<DecisionOption>(),
                mapContext: new MapChoiceContext
                {
                    Nodes = snapshot.Nodes,
                    PlayerRow = snapshot.PlayerRow,
                    NodeCount = snapshot.Nodes.Count,
                    MapName = $"Act_{state.Act}",
                    CurrentNodeId = snapshot.CurrentNodeId,
                    AvailableNextNodeIds =
                        snapshot.AvailableNextNodeIds,
                    BossNodeIds = snapshot.BossNodeIds,
                    BossEncounterIds = snapshot.BossEncounterIds,
                }
            );
        }
    }

    private static void ClosePendingDecision(DecisionOutcome outcome)
    {
        var pending = _pendingDecision;
        var state = _lastState;
        if (pending is null || state is null)
        {
            return;
        }
        Write(
            "decision_closed",
            state,
            new List<DecisionOption>(),
            parentEventId: pending.ParentEventId,
            outcome: outcome
        );
        _pendingDecision = null;
    }

    private static string? Write(
        string eventType,
        RunStateSnapshot state,
        List<DecisionOption> options,
        DecisionContext? decision = null,
        string? parentEventId = null,
        DecisionOutcome? outcome = null,
        MapChoiceContext? mapContext = null,
        RunResult? runResult = null
    )
    {
        var sequence = Interlocked.Increment(ref _sequence);
        var eventId = $"{_runId}:{sequence}";
        var stateEvent = new StateEvent
        {
            EventId = eventId,
            EventType = eventType,
            EmittedAt = DateTimeOffset.UtcNow.ToString("O"),
            GameVersion = GameVersionReader.Read(),
            RunId = _runId,
            Sequence = sequence,
            State = state,
            Options = options,
            Decision = decision,
            ParentEventId = parentEventId,
            Outcome = outcome,
            MapContext = mapContext,
            RunResult = runResult,
        };
        var json = JsonSerializer.Serialize(stateEvent, JsonOptions);
        var outputPath = ProjectSettings.GlobalizePath(
            "user://STS2Guide/state-event.json"
        );
        var directory = Path.GetDirectoryName(outputPath);
        if (string.IsNullOrWhiteSpace(directory))
        {
            Log.Error("[STS2-Guide] Invalid state-event output path.");
            return null;
        }

        try
        {
            Directory.CreateDirectory(directory);
            var eventDirectory = Path.Combine(directory, "events");
            Directory.CreateDirectory(eventDirectory);
            var eventPath = Path.Combine(
                eventDirectory,
                $"{_runId}-{sequence:D12}-{eventType}.json"
            );
            WriteAtomically(eventPath, json);

            try
            {
                WriteAtomically(outputPath, json);
            }
            catch (Exception latestException)
            {
                Log.Error(
                    "[STS2-Guide] Latest-event mirror write failed; "
                    + $"queued event is intact: {latestException.Message}"
                );
            }
            Log.Info(
                $"[STS2-Guide] Emitted {eventType} event "
                + $"{stateEvent.EventId}."
            );
            return eventId;
        }
        catch (Exception exception)
        {
            Log.Error(
                $"[STS2-Guide] State-event write failed: "
                + exception.Message
            );
            return null;
        }
    }

    private static void WriteAtomically(string path, string json)
    {
        var temporaryPath = $"{path}.tmp";
        File.WriteAllText(temporaryPath, json);
        File.Move(temporaryPath, path, true);
    }

    private static void EnsureStableRunIdentity(RunStateSnapshot state)
    {
        if (_runIdentityStable)
        {
            return;
        }
        if (_sequence > 0)
        {
            Log.Error(
                "[STS2-Guide] Stable run identity became available after "
                + "events were emitted; keeping temporary identity."
            );
            return;
        }
        // Read identity preferring the current player's RunState seed.
        // The player's seed is fresh for this run; RunManager.History.Seed
        // may be stale from a previous ended run.
        var identity = ReadIdentityFromCurrentRun();
        if (!identity.IsStable)
        {
            if (!state.CaptureWarnings.Contains("run_identity_fallback"))
            {
                state.CaptureWarnings.Add("run_identity_fallback");
            }
            return;
        }
        // Guard: if this stable ID matches a previously ended run, reject it.
        if (_previousRunEnded && identity.RunId == _previousRunId)
        {
            Log.Error(
                $"[STS2-Guide] Stable identity {identity.RunId} matches "
                + "previously ended run; keeping temporary ID to prevent "
                + "cross-run contamination."
            );
            if (!state.CaptureWarnings.Contains("run_identity_fallback"))
            {
                state.CaptureWarnings.Add("run_identity_fallback");
            }
            return;
        }
        _runId = identity.RunId;
        _runStartedAt = identity.StartedAt;
        _runIdentityStable = true;
        _sequence = RecoverSequence(_runId);
    }

    /// <summary>
    /// Read a stable run identity preferring the current player's seed.
    /// RunManager.History.Seed may be stale after a previous run ended.
    /// </summary>
    private static RunIdentity ReadIdentityFromCurrentRun()
    {
        // Prefer the player's RunState seed (fresh per-run).
        var playerSeed = RunStateReader
            .GetObservedPlayer()?
            .RunState
            .Rng
            .StringSeed;
        if (!string.IsNullOrWhiteSpace(playerSeed))
        {
            var startTime = RunManager.Instance.History?.StartTime ?? 0;
            if (startTime <= 0)
            {
                var field = typeof(RunManager).GetField(
                    "_startTime",
                    BindingFlags.NonPublic | BindingFlags.Instance
                );
                if (field?.GetValue(RunManager.Instance) is long st)
                    startTime = st;
            }
            if (startTime > 0)
            {
                var bytes = SHA256.HashData(
                    Encoding.UTF8.GetBytes($"{playerSeed}|{startTime}")
                );
                var digest = Convert.ToHexString(bytes).ToLowerInvariant();
                return new RunIdentity(
                    $"sts2-{digest[..24]}",
                    DateTimeOffset.FromUnixTimeSeconds(startTime).ToString("O"),
                    true
                );
            }
        }
        // Fall back to the standard reader.
        return RunIdentityReader.Read();
    }

    private static RunResult CreateRunResult(string outcome)
    {
        int? finalScore = null;
        try
        {
            finalScore = Math.Max(
                0,
                SaveManager.Instance.GetCurrentScore()
            );
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Final score unavailable: "
                + exception.Message
            );
        }
        return new RunResult
        {
            Outcome = outcome,
            FinalScore = finalScore,
            StartedAt = _runStartedAt,
            EndedAt = DateTimeOffset.UtcNow.ToString("O"),
        };
    }

    private static long RecoverSequence(string runId)
    {
        var directory = GetExchangeDirectory();
        if (directory is null)
        {
            return 0;
        }
        var maximum = 0L;
        maximum = Math.Max(
            maximum,
            ReadSequence(
                Path.Combine(directory, "active-run.json"),
                runId,
                "last_sequence"
            )
        );
        maximum = Math.Max(
            maximum,
            ReadSequence(
                Path.Combine(directory, "state-event.json"),
                runId,
                "sequence"
            )
        );
        var eventDirectory = Path.Combine(directory, "events");
        if (Directory.Exists(eventDirectory))
        {
            foreach (var path in Directory.EnumerateFiles(
                eventDirectory,
                "*.json"
            ))
            {
                maximum = Math.Max(
                    maximum,
                    ReadSequence(path, runId, "sequence")
                );
            }
        }
        return maximum;
    }

    private static long ReadSequence(
        string path,
        string runId,
        string sequenceProperty
    )
    {
        if (!File.Exists(path))
        {
            return 0;
        }
        try
        {
            using var document = JsonDocument.Parse(
                File.ReadAllText(path)
            );
            var root = document.RootElement;
            if (!root.TryGetProperty("run_id", out var storedRunId)
                || storedRunId.GetString() != runId
                || !root.TryGetProperty(
                    sequenceProperty,
                    out var storedSequence
                ))
            {
                return 0;
            }
            return storedSequence.TryGetInt64(out var sequence)
                ? Math.Max(0, sequence)
                : 0;
        }
        catch
        {
            return 0;
        }
    }

    private static string? GetExchangeDirectory()
    {
        var outputPath = ProjectSettings.GlobalizePath(
            "user://STS2Guide/state-event.json"
        );
        return Path.GetDirectoryName(outputPath);
    }

    private static bool SameOption(
        DecisionOption first,
        DecisionOption second
    )
    {
        return first.Card == second.Card
            && first.Upgrades == second.Upgrades
            && first.Enchantment == second.Enchantment
            && first.EnchantmentAmount == second.EnchantmentAmount
            && first.Affliction == second.Affliction
            && first.AfflictionAmount == second.AfflictionAmount;
    }

    private sealed record PendingDecision(
        string ParentEventId,
        List<DecisionOption> Options
    );
}

internal sealed record PendingDecisionView(
    string RunId,
    string EventId,
    IReadOnlyList<string> CardIds
);
