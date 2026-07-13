using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;

namespace STS2Guide.ReadOnlyExporter;

internal sealed record MapSnapshot(
    List<MapNodeState> Nodes,
    string? CurrentNodeId,
    List<string> AvailableNextNodeIds,
    List<string> BossNodeIds,
    List<string> BossEncounterIds,
    int? PlayerRow
);

/// <summary>
/// Reads only the verified v0.107.1 map API.  Missing data is reported as an
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

            var currentNodeId = currentPoint is null
                ? null
                : pointIds[currentPoint];
            var availableNextNodeIds = currentPoint is not null
                ? currentPoint.Children
                    .Where(pointIds.ContainsKey)
                    .Select(child => pointIds[child])
                    .Distinct(StringComparer.Ordinal)
                    .OrderBy(id => id, StringComparer.Ordinal)
                    .ToList()
                : map.startMapPoints
                    .Where(pointIds.ContainsKey)
                    .Select(point => pointIds[point])
                    .Distinct(StringComparer.Ordinal)
                    .OrderBy(id => id, StringComparer.Ordinal)
                    .ToList();
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
                availableNextNodeIds,
                bossNodeIds,
                bossEncounterIds,
                currentPoint?.coord.row
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
            [],
            [],
            [],
            null
        );
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
