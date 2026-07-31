using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace STS2Guide.ReadOnlyExporter;

internal sealed record MapSnapshot(
    List<MapNodeState> Nodes,
    string? CurrentNodeId,
    string? OriginNodeId,
    List<string> AvailableNextNodeIds,
    List<string> BossNodeIds,
    List<string> BossEncounterIds,
    int? PlayerRow,
    string? MapFingerprint
);

/// <summary>
/// Reads only the currently verified map API. Missing data is reported as an
/// empty snapshot; no guessed edges are ever produced.
/// </summary>
internal static class MapNodeReader
{
    internal static MapSnapshot Read(Player player)
    {
        try
        {
            var runState = player.RunState;
            var map = runState.Map;
            if (map is null)
            {
                return Empty();
            }

            // ActMap.GetAllMapPoints() only enumerates the regular Grid.  Boss
            // points are special nodes stored separately on ActMap, so they
            // must be added before we build the node lookup.  Otherwise the
            // final-row edges to the boss are silently discarded as well.
            var specialBossPoints = new MapPoint?[]
            {
                map.BossMapPoint,
                map.SecondBossMapPoint,
            };
            var points = map
                .GetAllMapPoints()
                .Cast<MapPoint?>()
                .Concat(specialBossPoints)
                .OfType<MapPoint>()
                .Distinct()
                .OrderBy(point => point.coord.row)
                .ThenBy(point => point.coord.col)
                .ToList();
            var pointIds = points.ToDictionary(
                point => point,
                NodeId
            );
            var nodes = points
                .Select(point => new MapNodeState
                {
                    NodeId = pointIds[point],
                    Kind = NormaliseKind(point.PointType),
                    Row = point.coord.row,
                    Col = point.coord.col,
                    Edges = point.Children
                        .Where(pointIds.ContainsKey)
                        .Select(child => pointIds[child])
                        .Distinct(StringComparer.Ordinal)
                        .OrderBy(id => id, StringComparer.Ordinal)
                        .ToList(),
                    Label = point.PointType.ToString(),
                })
                .ToList();

            MapPoint? currentPoint = null;
            try
            {
                var candidate = runState.CurrentMapPoint;
                if (candidate is not null && pointIds.ContainsKey(candidate))
                {
                    currentPoint = candidate;
                }
            }
            catch
            {
                // CurrentMapPoint may be unavailable before the first choice.
            }

            // CurrentMapCoord is a verified public RunState member.  It lets
            // us resolve the real Act-start origin (for example 0:3) without
            // treating an arbitrary startMapPoints entry as an origin.
            if (currentPoint is null)
            {
                try
                {
                    if (runState.CurrentMapCoord is { } currentCoord)
                    {
                        var coordinateId = $"{currentCoord.row}:{currentCoord.col}";
                        currentPoint = points.FirstOrDefault(point =>
                            pointIds[point] == coordinateId
                        );
                    }
                }
                catch
                {
                    // A transient saved-run setup has no usable route origin.
                }
            }

            var currentNodeId = currentPoint is null
                ? null
                : pointIds[currentPoint];
            // A route identity is allowed to use only a real current point.
            // If the public state is still transient, RouteChoiceObserver
            // fails closed rather than fabricating a start/sentinel origin.
            var originNodeId = currentNodeId;
            var availableNextNodeIds = currentPoint is not null
                ? currentPoint.Children
                    .Where(pointIds.ContainsKey)
                    .Select(child => pointIds[child])
                    .Distinct(StringComparer.Ordinal)
                    .OrderBy(id => id, StringComparer.Ordinal)
                    .ToList()
                : [];
            var bossNodeIds = specialBossPoints
                .OfType<MapPoint>()
                .Where(pointIds.ContainsKey)
                .Select(point => pointIds[point])
                .Distinct(StringComparer.Ordinal)
                .OrderBy(id => id, StringComparer.Ordinal)
                .ToList();
            var bossEncounterIds = new[]
                {
                    runState.Act.BossEncounter,
                    runState.Act.SecondBossEncounter,
                }
                .Select(encounter => encounter?.Id.Entry)
                .OfType<string>()
                .Where(id => !string.IsNullOrWhiteSpace(id))
                .Distinct(StringComparer.Ordinal)
                .OrderBy(id => id, StringComparer.Ordinal)
                .ToList();

            // Defensive fallback for a future map implementation that exposes
            // a boss inside its regular Grid instead of through BossMapPoint.
            if (bossNodeIds.Count == 0)
            {
                var bossPoints = points
                    .Where(p => p.PointType == MapPointType.Boss)
                    .ToList();
                if (bossPoints.Count > 0)
                {
                    bossNodeIds = bossPoints
                        .Select(p => pointIds[p])
                        .Distinct(StringComparer.Ordinal)
                        .OrderBy(id => id, StringComparer.Ordinal)
                        .ToList();
                    Log.Info(
                        $"[STS2-Guide] No special BossMapPoint was available; found "
                        + $"{bossPoints.Count} Boss node(s) via "
                        + "GetAllMapPoints() scan: "
                        + string.Join(", ", bossNodeIds)
                    );
                }
                else
                {
                    Log.Info(
                        "[STS2-Guide] No special BossMapPoint was available "
                        + "and the map contained no "
                        + "MapPointType.Boss nodes. Boss data is unavailable "
                        + "in this game version; boss_node_ids will be empty."
                    );
                }
            }

            return new MapSnapshot(
                nodes,
                currentNodeId,
                originNodeId,
                availableNextNodeIds,
                bossNodeIds,
                bossEncounterIds,
                currentPoint?.coord.row,
                CreateMapFingerprint(nodes)
            );
        }
        catch (Exception exception)
        {
            Log.Error(
                "[STS2-Guide] Verified map API read failed; "
                + "no map event will be emitted: "
                + exception.Message
            );
            return Empty();
        }
    }

    private static MapSnapshot Empty()
    {
        return new MapSnapshot(
            [],
            null,
            null,
            [],
            [],
            [],
            null,
            null
        );
    }

    /// <summary>
    /// Stable map identity shared by observation and checkpoint recovery.
    /// It intentionally uses only the verified logical graph, never Godot
    /// object IDs or screen coordinates.
    /// </summary>
    internal static string? CreateMapFingerprint(
        IReadOnlyList<MapNodeState> nodes
    )
    {
        if (nodes.Count == 0 || nodes.Select(node => node.NodeId)
            .Distinct(StringComparer.Ordinal).Count() != nodes.Count)
        {
            return null;
        }
        var canonical = string.Join(
            "|",
            nodes.OrderBy(node => node.Row)
                .ThenBy(node => node.Col)
                .ThenBy(node => node.NodeId, StringComparer.Ordinal)
                // Route scoring consumes node kind and geometry as well as
                // edges.  Leaving either out would let an initialization
                // UNKNOWN -> ELITE/CAMPFIRE transition reuse an old decision
                // identity and old recommendation.
                .Select(node => string.Join(
                    ":",
                    node.NodeId,
                    node.Kind,
                    node.Row,
                    node.Col
                ) + ">" + string.Join(
                    ",",
                    node.Edges.OrderBy(edge => edge, StringComparer.Ordinal)
                ))
        );
        return Convert.ToHexString(
            SHA256.HashData(Encoding.UTF8.GetBytes(canonical))
        );
    }

    internal static string? CreateMapFingerprint(JsonElement context)
    {
        if (!context.TryGetProperty("nodes", out var nodes)
            || nodes.ValueKind != JsonValueKind.Array)
        {
            return null;
        }
        var values = new List<MapNodeState>();
        foreach (var node in nodes.EnumerateArray())
        {
            if (node.ValueKind != JsonValueKind.Object
                || !node.TryGetProperty("node_id", out var id)
                || id.ValueKind != JsonValueKind.String
                || !node.TryGetProperty("kind", out var kind)
                || kind.ValueKind != JsonValueKind.String
                || string.IsNullOrWhiteSpace(kind.GetString())
                || !node.TryGetProperty("row", out var row)
                || !row.TryGetInt32(out var rowValue)
                || !node.TryGetProperty("col", out var col)
                || !col.TryGetInt32(out var colValue)
                || !node.TryGetProperty("edges", out var edges)
                || edges.ValueKind != JsonValueKind.Array)
            {
                return null;
            }
            var edgeIds = new List<string>();
            foreach (var edge in edges.EnumerateArray())
            {
                if (edge.ValueKind != JsonValueKind.String
                    || string.IsNullOrWhiteSpace(edge.GetString()))
                {
                    return null;
                }
                edgeIds.Add(edge.GetString()!);
            }
            values.Add(new MapNodeState
            {
                NodeId = id.GetString()!,
                Kind = kind.GetString()!,
                Row = rowValue,
                Col = colValue,
                Edges = edgeIds,
            });
        }
        return CreateMapFingerprint(values);
    }

    private static string NodeId(MapPoint point)
    {
        return $"{point.coord.row}:{point.coord.col}";
    }

    private static string NormaliseKind(MapPointType pointType)
    {
        return pointType switch
        {
            MapPointType.Monster => "MONSTER",
            MapPointType.Elite => "ELITE",
            MapPointType.RestSite => "CAMPFIRE",
            MapPointType.Shop => "SHOP",
            MapPointType.Unknown or MapPointType.Ancient => "EVENT",
            MapPointType.Boss => "BOSS",
            MapPointType.Treasure => "TREASURE",
            _ => "UNKNOWN",
        };
    }
}
