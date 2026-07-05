using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;

namespace STS2Guide.ReadOnlyExporter;

internal sealed record MapSnapshot(
    List<MapNodeState> Nodes,
    string? CurrentNodeId,
    List<string> AvailableNextNodeIds,
    List<string> BossNodeIds,
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

            var points = map
                .GetAllMapPoints()
                .Where(point => point is not null)
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
            var bossNodeIds = new[] {
                    map.BossMapPoint,
                    map.SecondBossMapPoint,
                }
                .OfType<MapPoint>()
                .Where(pointIds.ContainsKey)
                .Select(point => pointIds[point])
                .Distinct(StringComparer.Ordinal)
                .OrderBy(id => id, StringComparer.Ordinal)
                .ToList();

            return new MapSnapshot(
                nodes,
                currentNodeId,
                availableNextNodeIds,
                bossNodeIds,
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
