using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Runs;

namespace STS2Guide.RouteLiveProbe;

internal static class ProbeRecorder
{
    private const int ProbeSchemaVersion = 1;
    private const double SampleIntervalSeconds = 0.5;
    private const string ActStartOrigin = "__act_start__";

    private static readonly object Gate = new();
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = false,
    };
    private static readonly PropertyInfo? IsTravelableProperty =
        typeof(NMapPoint).GetProperty(
            "IsTravelable",
            BindingFlags.Instance
                | BindingFlags.Public
                | BindingFlags.NonPublic
                | BindingFlags.DeclaredOnly
        );
    private static readonly string SessionId = Guid.NewGuid().ToString("N");
    private static readonly string AssemblyVersion =
        typeof(NMapScreen).Assembly.GetName().Version?.ToString() ?? "unknown";
    private static readonly string AssemblyMvid =
        typeof(NMapScreen).Assembly.ManifestModule.ModuleVersionId.ToString("D");

    private static string? _logPath;
    private static long _sequence;
    private static long _lastSampleTimestamp;
    private static string? _lastSampleFingerprint;
    private static string? _lastCurrentNodeId;
    private static bool _writeFailureReported;

    internal static void Initialize()
    {
        lock (Gate)
        {
            try
            {
                var root = Path.Combine(
                    System.Environment.GetFolderPath(
                        System.Environment.SpecialFolder.LocalApplicationData
                    ),
                    "STS2Guide",
                    "dev",
                    "route-probe"
                );
                Directory.CreateDirectory(root);
                _logPath = Path.Combine(
                    root,
                    "route-probe-"
                        + DateTimeOffset.UtcNow.ToString("yyyyMMdd-HHmmss")
                        + "-"
                        + SessionId[..8]
                        + ".jsonl"
                );
            }
            catch (Exception exception)
            {
                ReportWriteFailure(exception);
            }
        }

        WriteEnvelope(
            "probe_session_started",
            null,
            new Dictionary<string, object?>
            {
                ["process_id"] = System.Environment.ProcessId,
                ["is_travelable_accessor"] =
                    IsTravelableProperty?.GetMethod is not null
                        ? "exact_property_available"
                        : "unavailable",
                ["log_contains_seed_or_player_name"] = false,
            }
        );
    }

    internal static void Capture(
        string eventName,
        NMapScreen screen,
        IReadOnlyDictionary<string, object?>? details = null
    )
    {
        try
        {
            var snapshot = BuildSnapshot(screen, null);
            UpdateLastObservation(snapshot);
            WriteEnvelope(eventName, snapshot, details);
        }
        catch (Exception exception)
        {
            WriteEnvelope(
                "probe_capture_failed",
                null,
                new Dictionary<string, object?>
                {
                    ["source_event"] = eventName,
                    ["error_type"] = exception.GetType().FullName,
                }
            );
        }
    }

    internal static void CaptureSelection(
        NMapScreen screen,
        NMapPoint point
    )
    {
        try
        {
            var priorSampleCurrentNodeId = _lastCurrentNodeId;
            var snapshot = BuildSnapshot(screen, null);
            var details = new Dictionary<string, object?>
            {
                ["selected_node_id"] = SafeNodeId(point.Point),
                ["selected_visual_state"] = point.State.ToString(),
                ["selected_is_travelable_exact"] = ReadExactIsTravelable(point),
                ["prior_sample_current_node_id"] = priorSampleCurrentNodeId,
                ["prior_sample_origin_node_id"] =
                    priorSampleCurrentNodeId ?? ActStartOrigin,
                ["postfix_current_map_point_id"] =
                    snapshot.Run?.CurrentMapPointId,
                ["postfix_origin_node_id"] = snapshot.Run?.OriginNodeId,
            };
            UpdateLastObservation(snapshot);
            WriteEnvelope("map_point_selected_postfix", snapshot, details);
        }
        catch (Exception exception)
        {
            WriteEnvelope(
                "probe_capture_failed",
                null,
                new Dictionary<string, object?>
                {
                    ["source_event"] = "map_point_selected_postfix",
                    ["error_type"] = exception.GetType().FullName,
                }
            );
        }
    }

    internal static void CaptureRunLifecycle(
        string eventName,
        RunState? runState,
        IReadOnlyDictionary<string, object?>? details = null
    )
    {
        try
        {
            var run = BuildRunSnapshot(runState, []);
            var snapshot = new ProbeSnapshot(null, run, []);
            UpdateLastObservation(snapshot);
            WriteEnvelope(eventName, snapshot, details);
        }
        catch (Exception exception)
        {
            WriteEnvelope(
                "probe_capture_failed",
                null,
                new Dictionary<string, object?>
                {
                    ["source_event"] = eventName,
                    ["error_type"] = exception.GetType().FullName,
                }
            );
        }
    }

    internal static void SampleIfChanged(NMapScreen screen)
    {
        var now = Stopwatch.GetTimestamp();
        var previous = Interlocked.Read(ref _lastSampleTimestamp);
        var elapsed = (now - previous) / (double)Stopwatch.Frequency;
        if (previous != 0 && elapsed < SampleIntervalSeconds)
        {
            return;
        }
        Interlocked.Exchange(ref _lastSampleTimestamp, now);

        try
        {
            var snapshot = BuildSnapshot(screen, null);
            var serialized = JsonSerializer.Serialize(snapshot, JsonOptions);
            var fingerprint = Convert.ToHexString(
                SHA256.HashData(Encoding.UTF8.GetBytes(serialized))
            );
            if (string.Equals(
                    fingerprint,
                    _lastSampleFingerprint,
                    StringComparison.Ordinal
                ))
            {
                return;
            }
            _lastSampleFingerprint = fingerprint;
            UpdateLastObservation(snapshot);
            WriteEnvelope("map_state_changed_sample", snapshot, null);
        }
        catch (Exception exception)
        {
            WriteEnvelope(
                "probe_capture_failed",
                null,
                new Dictionary<string, object?>
                {
                    ["source_event"] = "map_state_changed_sample",
                    ["error_type"] = exception.GetType().FullName,
                }
            );
        }
    }

    internal static string RuntimeObjectId(object instance)
    {
        return RuntimeHelpers.GetHashCode(instance).ToString("X8");
    }

    private static ProbeSnapshot BuildSnapshot(
        NMapScreen screen,
        RunState? explicitRunState
    )
    {
        var visualPoints = EnumerateDescendants(screen)
            .OfType<NMapPoint>()
            .Where(point => GodotObject.IsInstanceValid(point))
            .Select(point => BuildVisualPoint(screen, point))
            .OrderBy(point => point.NodeId, StringComparer.Ordinal)
            .ToList();
        ProbeVector2? viewportSize = screen.GetViewport() is { } viewport
            ? Vector(viewport.GetVisibleRect().Size)
            : null;
        var screenSnapshot = new ProbeScreenSnapshot(
            screen.GetInstanceId().ToString(),
            screen.IsInsideTree(),
            screen.Visible,
            screen.IsOpen,
            screen.IsTravelEnabled,
            screen.IsTraveling,
            screen.IsDebugTravelEnabled,
            Vector(screen.Position),
            Vector(screen.GlobalPosition),
            Vector(screen.Size),
            Vector(screen.Scale),
            Transform(screen.GetGlobalTransform()),
            viewportSize
        );
        var run = BuildRunSnapshot(explicitRunState, visualPoints);
        return new ProbeSnapshot(screenSnapshot, run, visualPoints);
    }

    private static ProbeMapPointSnapshot BuildVisualPoint(
        NMapScreen screen,
        NMapPoint point
    )
    {
        var center = point.GetGlobalRect().GetCenter();
        var netPosition = screen.GetNetPositionFromScreenPosition(center);
        var roundTrip = screen.GetScreenPositionFromNetPosition(netPosition);
        return new ProbeMapPointSnapshot(
            SafeNodeId(point.Point),
            point.GetInstanceId().ToString(),
            point.GetParent() is { } parent
                ? parent.GetInstanceId().ToString()
                : null,
            point.State.ToString(),
            ReadExactIsTravelable(point),
            point.State == MapPointState.Travelable,
            point.IsInsideTree(),
            point.Visible,
            Vector(point.Position),
            Vector(point.GlobalPosition),
            Vector(point.Size),
            Vector(point.Scale),
            Transform(point.GetGlobalTransform()),
            Vector(center),
            Vector(netPosition),
            Vector(roundTrip)
        );
    }

    private static ProbeRunSnapshot? BuildRunSnapshot(
        RunState? explicitRunState,
        IReadOnlyList<ProbeMapPointSnapshot> visualPoints
    )
    {
        RunManager? manager;
        try
        {
            manager = RunManager.Instance;
        }
        catch
        {
            manager = null;
        }

        RunState? runState = explicitRunState;
        if (runState is null && manager is not null)
        {
            try
            {
                runState = manager.DebugOnlyGetState();
            }
            catch
            {
                runState = null;
            }
        }
        if (runState is null)
        {
            return null;
        }

        MapPoint? currentPoint;
        try
        {
            currentPoint = runState.CurrentMapPoint;
        }
        catch
        {
            currentPoint = null;
        }
        var map = SafeRead<ActMap?>(() => runState.Map, null);
        var modelNext = currentPoint is not null
            ? SafeRead<IEnumerable<MapPoint>>(
                    () => currentPoint.Children,
                    Enumerable.Empty<MapPoint>()
                ).Select(SafeNodeId)
            : map?.startMapPoints.Select(SafeNodeId) ?? [];
        var modelNextIds = modelNext
            .Distinct(StringComparer.Ordinal)
            .OrderBy(id => id, StringComparer.Ordinal)
            .ToList();
        var visualTravelableIds = visualPoints
            .Where(point =>
                point.IsTravelableExact is true
                || point.StateEqualsTravelable
            )
            .Select(point => point.NodeId)
            .Distinct(StringComparer.Ordinal)
            .OrderBy(id => id, StringComparer.Ordinal)
            .ToList();
        var currentNodeId = currentPoint is null
            ? null
            : SafeNodeId(currentPoint);
        var currentCoord = SafeRead<MapCoord?>(
                () => runState.CurrentMapCoord,
                null
            ) is { } coord
            ? NodeId(coord)
            : null;
        var visited = SafeRead<IReadOnlyList<MapCoord>>(
                () => runState.VisitedMapCoords,
                []
            )
            .Select(NodeId)
            .ToList();

        var isInProgress = manager is not null
            && SafeRead(() => manager.IsInProgress, false);
        var isSingleplayer = manager is not null
            && SafeRead(
                () => manager.IsSingleplayerOrFakeMultiplayer,
                false
            );

        return new ProbeRunSnapshot(
            RuntimeObjectId(runState),
            isInProgress,
            isSingleplayer,
            SafeRead(() => runState.CurrentActIndex, -1),
            SafeRead(() => runState.ActFloor, -1),
            SafeRead(() => runState.TotalFloor, -1),
            SafeRead(() => runState.MapLocation.ToString(), "__unavailable__"),
            SafeRead(() => runState.RunLocation.ToString(), "__unavailable__"),
            currentNodeId ?? ActStartOrigin,
            currentNodeId,
            currentCoord,
            visited,
            modelNextIds,
            visualTravelableIds,
            BuildMapFingerprint(map)
        );
    }

    private static string? BuildMapFingerprint(ActMap? map)
    {
        if (map is null)
        {
            return null;
        }
        var points = map.GetAllMapPoints()
            .Cast<MapPoint?>()
            .Concat(new MapPoint?[]
            {
                map.BossMapPoint,
                map.SecondBossMapPoint,
            })
            .OfType<MapPoint>()
            .Distinct()
            .OrderBy(point => point.coord.row)
            .ThenBy(point => point.coord.col)
            .ToList();
        var canonical = string.Join(
            "|",
            points.Select(point =>
                SafeNodeId(point)
                + ">"
                + string.Join(
                    ",",
                    point.Children
                        .Select(SafeNodeId)
                        .OrderBy(id => id, StringComparer.Ordinal)
                )
            )
        );
        return Convert.ToHexString(
            SHA256.HashData(Encoding.UTF8.GetBytes(canonical))
        );
    }

    private static IEnumerable<Node> EnumerateDescendants(Node root)
    {
        foreach (var child in root.GetChildren(includeInternal: true))
        {
            yield return child;
            foreach (var descendant in EnumerateDescendants(child))
            {
                yield return descendant;
            }
        }
    }

    private static bool? ReadExactIsTravelable(NMapPoint point)
    {
        try
        {
            return IsTravelableProperty?.GetValue(point) as bool?;
        }
        catch
        {
            return null;
        }
    }

    private static T SafeRead<T>(Func<T> read, T fallback)
    {
        try
        {
            return read();
        }
        catch
        {
            return fallback;
        }
    }

    private static void UpdateLastObservation(ProbeSnapshot snapshot)
    {
        _lastCurrentNodeId = snapshot.Run?.CurrentMapPointId;
    }

    private static void WriteEnvelope(
        string eventName,
        ProbeSnapshot? snapshot,
        IReadOnlyDictionary<string, object?>? details
    )
    {
        lock (Gate)
        {
            if (_logPath is null)
            {
                return;
            }
            try
            {
                var envelope = new ProbeEnvelope(
                    ProbeSchemaVersion,
                    SessionId,
                    ++_sequence,
                    DateTimeOffset.UtcNow.ToString("O"),
                    eventName,
                    AssemblyVersion,
                    AssemblyMvid,
                    snapshot,
                    details ?? new Dictionary<string, object?>()
                );
                var line = JsonSerializer.Serialize(envelope, JsonOptions);
                using var stream = new FileStream(
                    _logPath,
                    FileMode.Append,
                    System.IO.FileAccess.Write,
                    FileShare.Read
                );
                using var writer = new StreamWriter(
                    stream,
                    new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)
                );
                writer.WriteLine(line);
                writer.Flush();
                stream.Flush(flushToDisk: true);
            }
            catch (Exception exception)
            {
                ReportWriteFailure(exception);
            }
        }
    }

    private static void ReportWriteFailure(Exception exception)
    {
        _logPath = null;
        if (_writeFailureReported)
        {
            return;
        }
        _writeFailureReported = true;
        Log.Error(
            "[STS2-Guide Route Probe] JSONL write disabled after failure: "
            + exception.GetType().Name
        );
    }

    private static string SafeNodeId(MapPoint? point)
    {
        return point is null ? "__missing__" : NodeId(point.coord);
    }

    private static string NodeId(MapCoord coord)
    {
        return $"{coord.row}:{coord.col}";
    }

    private static ProbeVector2 Vector(Vector2 value)
    {
        return new ProbeVector2(Round(value.X), Round(value.Y));
    }

    private static ProbeTransform2D Transform(Transform2D value)
    {
        return new ProbeTransform2D(
            Vector(value.X),
            Vector(value.Y),
            Vector(value.Origin)
        );
    }

    private static double Round(float value)
    {
        return Math.Round(value, 3, MidpointRounding.AwayFromZero);
    }
}
