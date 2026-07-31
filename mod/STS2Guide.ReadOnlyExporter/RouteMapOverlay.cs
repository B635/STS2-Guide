using Godot;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>
/// Read-only route lines.  The controller validates advice identity; this
/// layer repeats topology and visual-mapping checks so a stale/invalid draw
/// can never leave a plausible line on the native map.
/// </summary>
internal static class RouteMapOverlay
{
    private static readonly object Gate = new();
    private static NMapScreen? _owner;
    private static string? _decisionId;
    private static RouteOverlayControl? _control;

    internal static void Render(
        NMapScreen owner,
        string decisionId,
        RoutePresentation presentation,
        MapChoiceContext context)
    {
        lock (Gate)
        {
            if (!ValidateTopology(presentation, context))
            {
                Hide(owner, decisionId);
                return;
            }
            if (presentation.PrimaryPathNodeIds.Count == 0)
            {
                Hide(owner, decisionId);
                return;
            }
            if (!ReferenceEquals(_owner, owner) || _decisionId != decisionId)
            {
                HideInternal();
                var nativePaths = owner.GetNodeOrNull<Control>(
                    "TheMap/Paths"
                );
                if (nativePaths is null)
                {
                    return;
                }
                _owner = owner;
                _decisionId = decisionId;
                _control = new RouteOverlayControl(
                    owner,
                    presentation,
                    context.OriginNodeId!
                );
                // Share the native map transform and remain below the Points
                // sibling and global top bar.  The line is still an
                // independent read-only child and never touches native path
                // textures or player drawings.
                nativePaths.AddChild(_control);
            }
            if (_control is not null && GodotObject.IsInstanceValid(_control))
            {
                _control.SetPresentation(
                    presentation,
                    context.OriginNodeId!
                );
                _control.RefreshLine();
            }
        }
    }

    internal static void Hide(
        NMapScreen? owner = null,
        string? decisionId = null)
    {
        lock (Gate)
        {
            if ((owner is not null && !ReferenceEquals(owner, _owner))
                || (decisionId is not null && decisionId != _decisionId))
            {
                return;
            }
            HideInternal();
        }
    }

    private static void HideInternal()
    {
        if (_control is not null && GodotObject.IsInstanceValid(_control))
        {
            _control.QueueFree();
        }
        _control = null;
        _owner = null;
        _decisionId = null;
    }

    private static bool ValidateTopology(
        RoutePresentation presentation,
        MapChoiceContext context)
    {
        try
        {
            var nodes = context.Nodes.ToDictionary(node => node.NodeId, StringComparer.Ordinal);
            if (nodes.Count != context.Nodes.Count
                || string.IsNullOrWhiteSpace(context.OriginNodeId)
                || !nodes.ContainsKey(context.OriginNodeId)
                || presentation.RecommendedCandidateId is null)
            {
                return presentation.RecommendedCandidateId is null
                    && presentation.PrimaryPathNodeIds.Count == 0;
            }
            if (presentation.PrimaryPathNodeIds.Count == 0
                || presentation.PrimaryPathNodeIds[0] != presentation.RecommendedCandidateId
                || !context.AvailableNextNodeIds.Contains(
                    presentation.RecommendedCandidateId,
                    StringComparer.Ordinal)
                || !nodes[context.OriginNodeId].Edges.Contains(
                    presentation.RecommendedCandidateId,
                    StringComparer.Ordinal)
                || !ValidatePath(presentation.PrimaryPathNodeIds, nodes, context.BossNodeIds))
            {
                return false;
            }
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool ValidatePath(
        IReadOnlyList<string> path,
        IReadOnlyDictionary<string, MapNodeState> nodes,
        IReadOnlyList<string> bossIds)
    {
        if (path.Count == 0)
        {
            return false;
        }
        for (var index = 0; index < path.Count; index++)
        {
            if (!nodes.TryGetValue(path[index], out var node)
                || (index + 1 < path.Count
                    && !node.Edges.Contains(path[index + 1], StringComparer.Ordinal)))
            {
                return false;
            }
        }
        var terminal = path[^1];
        return bossIds.Contains(terminal, StringComparer.Ordinal)
            && nodes[terminal].Kind == "BOSS";
    }

    private sealed class RouteOverlayControl : Control
    {
        private readonly NMapScreen _owner;
        private readonly Line2D _line;
        private RoutePresentation _presentation;
        private string _originNodeId;

        internal RouteOverlayControl(
            NMapScreen owner,
            RoutePresentation presentation,
            string originNodeId)
        {
            _owner = owner;
            _presentation = presentation;
            _originNodeId = originNodeId;
            Name = "STS2GuideRouteMapOverlay";
            MouseFilter = MouseFilterEnum.Ignore;
            ZIndex = 0;
            SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
            _line = new Line2D
            {
                Name = "STS2GuidePrimaryRouteLine",
                Width = 6F,
                DefaultColor = new Color(1F, 0.78F, 0.22F, 0.95F),
                Antialiased = true
            };
            AddChild(_line);
            SetProcess(true);
        }

        internal void SetPresentation(
            RoutePresentation presentation,
            string originNodeId)
        {
            _presentation = presentation;
            _originNodeId = originNodeId;
        }

        public override void _Process(double delta)
        {
            // Map scroll, viewport/content scaling and window changes can all
            // move GetGlobalRect() without a new advice file.  Rebuild every
            // frame from the current owner; never cache screen coordinates.
            RefreshLine();
        }

        internal void RefreshLine()
        {
            try
            {
                if (!GodotObject.IsInstanceValid(_owner))
                {
                    _line.ClearPoints();
                    return;
                }
                var visuals = ResolveVisuals();
                var completePath = new[] { _originNodeId }
                    .Concat(_presentation.PrimaryPathNodeIds)
                    .ToList();
                if (visuals is null
                    || !TryResolvePoints(
                        completePath,
                        visuals,
                        out var primary
                    ))
                {
                    _line.ClearPoints();
                    return;
                }
                _line.Points = primary.Count >= 2 ? primary.ToArray() : [];
            }
            catch
            {
                // Rendering is strictly best-effort.  A current-frame visual
                // failure hides all lines by drawing nothing, not by throwing
                // from Godot's draw loop.
                _line.ClearPoints();
            }
        }

        private Dictionary<string, NMapPoint>? ResolveVisuals()
        {
            var result = new Dictionary<string, NMapPoint>(StringComparer.Ordinal);
            foreach (var point in EnumerateDescendants(_owner).OfType<NMapPoint>())
            {
                if (!GodotObject.IsInstanceValid(point))
                {
                    return null;
                }
                var nodeId = $"{point.Point.coord.row}:{point.Point.coord.col}";
                if (!result.TryAdd(nodeId, point))
                {
                    // A duplicate mapping makes the visual identity
                    // ambiguous; draw no route.
                    return null;
                }
            }
            return result;
        }

        private bool TryResolvePoints(
            IReadOnlyList<string> nodeIds,
            IReadOnlyDictionary<string, NMapPoint> visuals,
            out List<Vector2> points)
        {
            points = [];
            foreach (var nodeId in nodeIds)
            {
                if (!visuals.TryGetValue(nodeId, out var visual)
                    || !GodotObject.IsInstanceValid(visual))
                {
                    return false;
                }
                points.Add(_line.ToLocal(visual.GetGlobalRect().GetCenter()));
            }
            return true;
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
    }
}
