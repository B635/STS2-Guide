namespace STS2Guide.RouteLiveProbe;

internal sealed record ProbeEnvelope(
    int SchemaVersion,
    string SessionId,
    long Sequence,
    string ObservedAtUtc,
    string EventName,
    string AssemblyVersion,
    string AssemblyMvid,
    ProbeSnapshot? Snapshot,
    IReadOnlyDictionary<string, object?> Details
);

internal sealed record ProbeSnapshot(
    ProbeScreenSnapshot? Screen,
    ProbeRunSnapshot? Run,
    IReadOnlyList<ProbeMapPointSnapshot> VisualPoints
);

internal sealed record ProbeScreenSnapshot(
    string OwnerInstanceId,
    bool IsInsideTree,
    bool IsVisible,
    bool IsOpen,
    bool IsTravelEnabled,
    bool IsTraveling,
    bool IsDebugTravelEnabled,
    ProbeVector2 Position,
    ProbeVector2 GlobalPosition,
    ProbeVector2 Size,
    ProbeVector2 Scale,
    ProbeTransform2D GlobalTransform,
    ProbeVector2? ViewportSize
);

internal sealed record ProbeRunSnapshot(
    string RuntimeObjectId,
    bool IsInProgress,
    bool IsSingleplayerOrFakeMultiplayer,
    int CurrentActIndex,
    int ActFloor,
    int TotalFloor,
    string MapLocation,
    string RunLocation,
    string OriginNodeId,
    string? CurrentMapPointId,
    string? CurrentMapCoord,
    IReadOnlyList<string> VisitedMapCoordIds,
    IReadOnlyList<string> ModelNextNodeIds,
    IReadOnlyList<string> VisualTravelableNodeIds,
    string? MapFingerprint
);

internal sealed record ProbeMapPointSnapshot(
    string NodeId,
    string SceneInstanceId,
    string? ParentSceneInstanceId,
    string State,
    bool? IsTravelableExact,
    bool StateEqualsTravelable,
    bool IsInsideTree,
    bool IsVisible,
    ProbeVector2 Position,
    ProbeVector2 GlobalPosition,
    ProbeVector2 Size,
    ProbeVector2 Scale,
    ProbeTransform2D GlobalTransform,
    ProbeVector2 GlobalRectCenter,
    ProbeVector2 NetPositionFromCenter,
    ProbeVector2 ScreenPositionRoundTrip
);

internal readonly record struct ProbeVector2(double X, double Y);

internal sealed record ProbeTransform2D(
    ProbeVector2 X,
    ProbeVector2 Y,
    ProbeVector2 Origin
);
