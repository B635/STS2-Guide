using Godot;

namespace STS2Guide.ReadOnlyExporter;

internal readonly record struct ContextDrawerRow(
    string Caption,
    double? Score,
    bool Recommended,
    IReadOnlyList<string>? Reasons = null
);

internal readonly record struct ContextDrawerHandle(long Value)
{
    internal bool IsValid => Value > 0;
}

/// <summary>
/// Shared read-only presentation shell for every STS2 Guide decision.
/// Decision-specific observers own validation and parsing; this class only
/// owns layout, rows, viewport placement, and the collapse interaction.
/// </summary>
internal static class ContextDrawer
{
    private const float MinWidth = 160F;
    private const float MaxWidth = 280F;
    private const float ViewportRatio = 0.18F;
    private const float MaxViewportCoverage = 0.3F;
    private const float RowHeight = 32F;
    private const float HeaderHeight = 42F;
    private const float RouteModeHeight = 38F;
    private const float VerticalPadding = 24F;
    private const float EdgeMargin = 10F;
    private const float ToggleWidth = 30F;
    private const float ToggleHeight = 48F;
    private const float ToggleOverlap = 2F;
    private const double SlideDuration = 0.22;
    private static readonly object Gate = new();

    private static Control? _owner;
    private static Control? _anchor;
    private static PanelContainer? _panel;
    private static Button? _toggle;
    private static Tween? _slideTween;
    private static List<Label> _rows = [];
    private static Dictionary<string, Button> _routeModeButtons = new(
        StringComparer.Ordinal
    );
    private static Func<string, bool>? _routeModeChanged;
    private static string? _routeMode;
    private static bool _collapsed;
    private static bool _sliding;
    private static long _nextHandle;
    private static ContextDrawerHandle _activeHandle;
    private static float _requestedHeight;

    internal static bool IsVisible(ContextDrawerHandle handle)
    {
        lock (Gate)
        {
            return handle.IsValid
                && handle == _activeHandle
                && _owner is not null
                && _panel is not null
                && GodotObject.IsInstanceValid(_owner)
                && GodotObject.IsInstanceValid(_panel);
        }
    }

    internal static ContextDrawerHandle Show(
        Control owner,
        Control? anchor,
        string title,
        IReadOnlyList<string> captions,
        string? routeMode = null,
        Func<string, bool>? routeModeChanged = null)
    {
        lock (Gate)
        {
            HideInternal();
            if (captions.Count == 0)
            {
                return default;
            }
            if ((routeMode is null) != (routeModeChanged is null)
                || (routeMode is not null
                    && !GuideRouteModes.IsValid(routeMode)))
            {
                return default;
            }
            try
            {
                _activeHandle = new ContextDrawerHandle(
                    Interlocked.Increment(ref _nextHandle)
                );
                _owner = owner;
                _anchor = anchor;
                _routeMode = routeMode;
                _routeModeChanged = routeModeChanged;
                _requestedHeight = HeaderHeight
                    + (routeMode is null ? 0F : RouteModeHeight)
                    + captions.Count * RowHeight
                    + VerticalPadding;
                var viewport = owner.GetViewport().GetVisibleRect();
                var initialHeight = Math.Min(
                    _requestedHeight,
                    AvailableHeight(viewport)
                );
                _panel = BuildPanel(
                    title,
                    captions,
                    initialHeight,
                    routeMode
                );
                owner.AddChild(_panel);
                _toggle = BuildToggleButton();
                owner.AddChild(_toggle);
                _collapsed = false;
                PlaceInternal();
                return _activeHandle;
            }
            catch
            {
                // Show is transactional: callers either receive a live handle
                // or no drawer state/nodes survive the failed construction.
                HideInternal();
                throw;
            }
        }
    }

    internal static void Render(
        ContextDrawerHandle handle,
        IReadOnlyList<ContextDrawerRow> rows)
    {
        lock (Gate)
        {
            if (handle != _activeHandle || _rows.Count != rows.Count)
            {
                return;
            }
            for (var index = 0; index < rows.Count; index++)
            {
                var row = rows[index];
                _rows[index].Text = row.Score.HasValue
                    ? $"{row.Caption}   {Math.Clamp(row.Score.Value, 0, 100):0}"
                    : $"{row.Caption}   --";
                _rows[index].AddThemeColorOverride(
                    "font_color",
                    row.Recommended
                        ? new Color(1F, 0.78F, 0.24F)
                        : new Color(0.92F, 0.94F, 0.98F)
                );
                _rows[index].TooltipText = row.Reasons is { Count: > 0 }
                    ? row.Caption
                        + "\n"
                        + string.Join(
                            "\n",
                            row.Reasons
                                .Take(3)
                                .Select(reason => "• " + reason)
                        )
                    : row.Caption;
            }
        }
    }

    internal static void Reposition(ContextDrawerHandle handle)
    {
        lock (Gate)
        {
            if (handle == _activeHandle)
            {
                PlaceInternal();
            }
        }
    }

    internal static void SetRouteMode(
        ContextDrawerHandle handle,
        string routeMode)
    {
        lock (Gate)
        {
            if (handle != _activeHandle
                || !GuideRouteModes.IsValid(routeMode)
                || _routeModeChanged is null)
            {
                return;
            }
            _routeMode = routeMode;
            RenderRouteModeButtons();
        }
    }

    internal static void Hide(ContextDrawerHandle handle)
    {
        lock (Gate)
        {
            if (handle == _activeHandle)
            {
                HideInternal();
            }
        }
    }

    private static PanelContainer BuildPanel(
        string title,
        IReadOnlyList<string> captions,
        float height,
        string? routeMode)
    {
        var panel = new PanelContainer
        {
            Name = "STS2GuideContextDrawer",
            MouseFilter = Control.MouseFilterEnum.Ignore,
            CustomMinimumSize = new Vector2(MinWidth, height),
            Size = new Vector2(MinWidth, height),
            ZIndex = 100,
        };
        panel.AddThemeStyleboxOverride("panel", new StyleBoxFlat
        {
            BgColor = new Color(0.025F, 0.035F, 0.055F, 0.84F),
            BorderColor = new Color(0.35F, 0.48F, 0.62F, 0.88F),
            CornerRadiusTopLeft = 12,
            CornerRadiusTopRight = 12,
            CornerRadiusBottomLeft = 12,
            CornerRadiusBottomRight = 12,
            BorderWidthLeft = 0,
            BorderWidthTop = 2,
            BorderWidthRight = 2,
            BorderWidthBottom = 2,
            ContentMarginLeft = 12,
            ContentMarginTop = 10,
            ContentMarginRight = 12,
            ContentMarginBottom = 10,
        });

        var column = new VBoxContainer
        {
            MouseFilter = Control.MouseFilterEnum.Ignore,
            SizeFlagsVertical = Control.SizeFlags.ExpandFill,
        };
        column.AddThemeConstantOverride("separation", 5);
        panel.AddChild(column);

        var heading = new Label
        {
            Text = title,
            HorizontalAlignment = HorizontalAlignment.Center,
            MouseFilter = Control.MouseFilterEnum.Ignore,
            CustomMinimumSize = new Vector2(0F, HeaderHeight - 10F),
        };
        heading.AddThemeFontSizeOverride("font_size", 20);
        heading.AddThemeColorOverride(
            "font_color",
            new Color(0.78F, 0.88F, 1F)
        );
        column.AddChild(heading);

        if (routeMode is not null)
        {
            var modeRow = new HBoxContainer
            {
                Name = "STS2GuideRouteModes",
                MouseFilter = Control.MouseFilterEnum.Ignore,
                SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
                CustomMinimumSize = new Vector2(0F, RouteModeHeight),
            };
            modeRow.AddThemeConstantOverride("separation", 4);
            column.AddChild(modeRow);
            _routeModeButtons = new Dictionary<string, Button>(
                StringComparer.Ordinal
            );
            AddRouteModeButton(
                modeRow,
                GuideRouteModes.Balanced,
                "智能均衡"
            );
            AddRouteModeButton(
                modeRow,
                GuideRouteModes.Survival,
                "稳健生存"
            );
            AddRouteModeButton(
                modeRow,
                GuideRouteModes.Growth,
                "激进成长"
            );
            RenderRouteModeButtons();
        }

        var scroll = new ScrollContainer
        {
            Name = "STS2GuideContextDrawerScroll",
            MouseFilter = Control.MouseFilterEnum.Stop,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
            VerticalScrollMode = ScrollContainer.ScrollMode.Auto,
            SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
            SizeFlagsVertical = Control.SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0F, RowHeight),
        };
        column.AddChild(scroll);
        var rowColumn = new VBoxContainer
        {
            MouseFilter = Control.MouseFilterEnum.Ignore,
            SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
        };
        rowColumn.AddThemeConstantOverride("separation", 5);
        scroll.AddChild(rowColumn);

        _rows = [];
        foreach (var caption in captions)
        {
            var label = new Label
            {
                Text = $"{caption}   --",
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                // Rows are read-only, but must receive hover events so their
                // deterministic reason tooltip is actually reachable.
                MouseFilter = Control.MouseFilterEnum.Stop,
                CustomMinimumSize = new Vector2(0F, RowHeight),
            };
            label.AddThemeFontSizeOverride("font_size", 18);
            rowColumn.AddChild(label);
            _rows.Add(label);
        }
        return panel;
    }

    private static void AddRouteModeButton(
        HBoxContainer row,
        string routeMode,
        string caption)
    {
        var button = new Button
        {
            Name = "STS2GuideRouteMode_" + routeMode,
            Text = caption,
            TooltipText = caption,
            MouseFilter = Control.MouseFilterEnum.Stop,
            FocusMode = Control.FocusModeEnum.None,
            SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0F, RouteModeHeight - 4F),
        };
        button.AddThemeFontSizeOverride("font_size", 14);
        button.Pressed += () => RouteModePressed(routeMode);
        row.AddChild(button);
        _routeModeButtons[routeMode] = button;
    }

    private static void RouteModePressed(string requestedMode)
    {
        Func<string, bool>? callback;
        ContextDrawerHandle handle;
        lock (Gate)
        {
            if (!IsValid()
                || _routeModeChanged is null
                || !GuideRouteModes.IsValid(requestedMode)
                || _routeMode == requestedMode)
            {
                return;
            }
            callback = _routeModeChanged;
            handle = _activeHandle;
        }

        var committed = false;
        try
        {
            committed = callback!(requestedMode);
        }
        catch
        {
            committed = false;
        }
        lock (Gate)
        {
            if (committed && handle == _activeHandle)
            {
                _routeMode = requestedMode;
                RenderRouteModeButtons();
            }
        }
    }

    private static void RenderRouteModeButtons()
    {
        foreach (var pair in _routeModeButtons)
        {
            if (!GodotObject.IsInstanceValid(pair.Value))
            {
                continue;
            }
            var selected = pair.Key == _routeMode;
            pair.Value.Disabled = selected;
            pair.Value.AddThemeColorOverride(
                "font_color",
                selected
                    ? new Color(1F, 0.78F, 0.24F)
                    : new Color(0.82F, 0.88F, 0.96F)
            );
        }
    }

    private static Button BuildToggleButton()
    {
        var button = new Button
        {
            Name = "STS2GuideContextDrawerToggle",
            Text = "‹",
            TooltipText = "收起建议",
            MouseFilter = Control.MouseFilterEnum.Stop,
            FocusMode = Control.FocusModeEnum.None,
            CustomMinimumSize = new Vector2(ToggleWidth, ToggleHeight),
            Size = new Vector2(ToggleWidth, ToggleHeight),
            ZIndex = 101,
        };
        button.AddThemeFontSizeOverride("font_size", 26);
        button.AddThemeColorOverride(
            "font_color",
            new Color(0.82F, 0.9F, 1F)
        );
        button.AddThemeColorOverride(
            "font_hover_color",
            new Color(1F, 0.82F, 0.36F)
        );
        button.AddThemeStyleboxOverride(
            "normal",
            BuildToggleStyle(new Color(0.025F, 0.035F, 0.055F, 0.84F))
        );
        button.AddThemeStyleboxOverride(
            "hover",
            BuildToggleStyle(new Color(0.08F, 0.12F, 0.18F, 0.94F))
        );
        button.AddThemeStyleboxOverride(
            "pressed",
            BuildToggleStyle(new Color(0.12F, 0.16F, 0.23F, 0.98F))
        );
        button.Pressed += Toggle;
        return button;
    }

    private static StyleBoxFlat BuildToggleStyle(Color background)
    {
        return new StyleBoxFlat
        {
            BgColor = background,
            BorderColor = new Color(0.35F, 0.48F, 0.62F, 0.88F),
            BorderWidthLeft = 0,
            BorderWidthTop = 2,
            BorderWidthRight = 2,
            BorderWidthBottom = 2,
            CornerRadiusTopRight = 10,
            CornerRadiusBottomRight = 10,
        };
    }

    private static void Toggle()
    {
        lock (Gate)
        {
            if (!IsValid())
            {
                return;
            }
            _collapsed = !_collapsed;
            _toggle!.Text = _collapsed ? "›" : "‹";
            _toggle.TooltipText = _collapsed ? "展开建议" : "收起建议";
            PlaceInternal(animate: true);
        }
    }

    private static void PlaceInternal(bool animate = false)
    {
        if (!IsValid() || (_sliding && !animate))
        {
            return;
        }
        var viewport = _owner!.GetViewport().GetVisibleRect();
        var maximumWidth = Math.Max(
            MinWidth,
            Math.Min(MaxWidth, viewport.Size.X * MaxViewportCoverage)
        );
        var contentWidth = _panel!.GetCombinedMinimumSize().X;
        var width = Math.Clamp(
            Math.Max(viewport.Size.X * ViewportRatio, contentWidth),
            MinWidth,
            maximumWidth
        );
        var height = Math.Min(
            Math.Max(1F, _requestedHeight),
            AvailableHeight(viewport)
        );
        _panel.CustomMinimumSize = new Vector2(MinWidth, height);
        _panel.Size = new Vector2(width, height);
        var y = viewport.Position.Y + (viewport.Size.Y - height) / 2F;
        if (_anchor is not null && GodotObject.IsInstanceValid(_anchor))
        {
            y = _anchor.GetGlobalRect().GetCenter().Y - height / 2F;
        }
        y = Math.Clamp(
            y,
            viewport.Position.Y + EdgeMargin,
            Math.Max(
                viewport.Position.Y + EdgeMargin,
                viewport.End.Y - height - EdgeMargin
            )
        );
        var expandedX = viewport.Position.X;
        var x = _collapsed ? expandedX - width : expandedX;
        var panelTarget = new Vector2(x, y);
        var toggleTarget = new Vector2(
            x + width - ToggleOverlap,
            y + (height - ToggleHeight) / 2F
        );
        if (!animate)
        {
            _panel.GlobalPosition = panelTarget;
            _toggle!.GlobalPosition = toggleTarget;
            return;
        }
        StopTween();
        _sliding = true;
        _slideTween = _owner.CreateTween();
        _slideTween.SetParallel();
        _slideTween.SetTrans(Tween.TransitionType.Cubic);
        _slideTween.SetEase(Tween.EaseType.Out);
        _slideTween.TweenProperty(
            _panel,
            "global_position",
            panelTarget,
            SlideDuration
        );
        _slideTween.TweenProperty(
            _toggle,
            "global_position",
            toggleTarget,
            SlideDuration
        );
        _slideTween.Finished += FinishTween;
    }

    private static bool IsValid()
    {
        return _owner is not null
            && _panel is not null
            && _toggle is not null
            && GodotObject.IsInstanceValid(_owner)
            && GodotObject.IsInstanceValid(_panel)
            && GodotObject.IsInstanceValid(_toggle);
    }

    private static float AvailableHeight(Rect2 viewport)
    {
        return Math.Max(1F, viewport.Size.Y - EdgeMargin * 2F);
    }

    private static void StopTween()
    {
        if (_slideTween is not null && GodotObject.IsInstanceValid(_slideTween))
        {
            _slideTween.Finished -= FinishTween;
            _slideTween.Kill();
        }
        _slideTween = null;
        _sliding = false;
    }

    private static void FinishTween()
    {
        lock (Gate)
        {
            if (_slideTween is not null
                && GodotObject.IsInstanceValid(_slideTween))
            {
                _slideTween.Finished -= FinishTween;
            }
            _slideTween = null;
            _sliding = false;
            PlaceInternal();
        }
    }

    private static void HideInternal()
    {
        StopTween();
        if (_toggle is not null && GodotObject.IsInstanceValid(_toggle))
        {
            _toggle.Pressed -= Toggle;
            _toggle.QueueFree();
        }
        if (_panel is not null && GodotObject.IsInstanceValid(_panel))
        {
            _panel.QueueFree();
        }
        _owner = null;
        _anchor = null;
        _panel = null;
        _toggle = null;
        _rows = [];
        _routeModeButtons = new Dictionary<string, Button>(
            StringComparer.Ordinal
        );
        _routeModeChanged = null;
        _routeMode = null;
        _collapsed = false;
        _sliding = false;
        _activeHandle = default;
        _requestedHeight = 0F;
    }
}
