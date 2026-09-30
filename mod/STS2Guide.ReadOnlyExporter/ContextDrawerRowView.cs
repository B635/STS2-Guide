using Godot;

namespace STS2Guide.ReadOnlyExporter;

/// <summary>One bounded, read-only candidate row with an optional local explanation.</summary>
internal sealed class ContextDrawerRowView
{
    internal PanelContainer Node { get; }
    internal bool HasReasons { get; private set; }
    private readonly Label _caption;
    private readonly Label _score;
    private readonly Label _marker;
    private readonly Label _reasons;
    private readonly Button _details;
    private readonly StyleBoxFlat _style;
    private static readonly Color Text = new(0.90F, 0.93F, 0.97F);
    private static readonly Color Accent = new(1F, 0.78F, 0.24F);

    internal ContextDrawerRowView(int index, string caption, Action onDetails)
    {
        Node = new PanelContainer
        {
            Name = "CandidateRow" + index,
            MouseFilter = Control.MouseFilterEnum.Ignore,
            SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
        };
        _style = new StyleBoxFlat
        {
            BgColor = new Color(0.065F, 0.085F, 0.12F, 0.70F),
            BorderWidthLeft = 3,
            BorderColor = new Color(0.20F, 0.28F, 0.36F, 0.6F),
            CornerRadiusTopLeft = 5, CornerRadiusTopRight = 5,
            CornerRadiusBottomLeft = 5, CornerRadiusBottomRight = 5,
            ContentMarginLeft = 6, ContentMarginRight = 6,
            ContentMarginTop = 4, ContentMarginBottom = 4,
        };
        Node.AddThemeStyleboxOverride("panel", _style);
        var column = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        column.AddThemeConstantOverride("separation", 6);
        Node.AddChild(column);
        var header = new HBoxContainer
        {
            MouseFilter = Control.MouseFilterEnum.Ignore,
            CustomMinimumSize = new Vector2(0F, 32F),
        };
        header.AddThemeConstantOverride("separation", 4);
        column.AddChild(header);
        _marker = new Label
        {
            Name = "CandidateMarker" + index,
            CustomMinimumSize = new Vector2(8F, 0F),
            VerticalAlignment = VerticalAlignment.Center,
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        _marker.AddThemeFontSizeOverride("font_size", 9);
        _marker.AddThemeColorOverride("font_color", Accent);
        header.AddChild(_marker);
        _caption = new Label
        {
            Name = "CandidateCaption" + index,
            SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
            VerticalAlignment = VerticalAlignment.Center,
            ClipText = true,
            TextOverrunBehavior = TextServer.OverrunBehavior.TrimEllipsis,
            MouseFilter = Control.MouseFilterEnum.Stop,
        };
        _caption.AddThemeFontSizeOverride("font_size", 18);
        header.AddChild(_caption);
        _score = new Label
        {
            Name = "CandidateScore" + index,
            CustomMinimumSize = new Vector2(32F, 0F),
            HorizontalAlignment = HorizontalAlignment.Right,
            VerticalAlignment = VerticalAlignment.Center,
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        _score.AddThemeFontSizeOverride("font_size", 20);
        header.AddChild(_score);
        _details = new Button
        {
            Name = "CandidateDetails" + index,
            Text = "+", Flat = true,
            CustomMinimumSize = new Vector2(22F, 0F),
            MouseFilter = Control.MouseFilterEnum.Stop,
            FocusMode = Control.FocusModeEnum.None,
        };
        _details.AddThemeFontSizeOverride("font_size", 16);
        _details.Pressed += onDetails;
        header.AddChild(_details);
        _reasons = new Label
        {
            Name = "CandidateReasons" + index,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            MouseFilter = Control.MouseFilterEnum.Stop,
            Visible = false,
        };
        _reasons.AddThemeFontSizeOverride("font_size", 14);
        _reasons.AddThemeColorOverride("font_color", new Color(0.75F, 0.81F, 0.89F));
        column.AddChild(_reasons);
        Render(new ContextDrawerRow(caption, null, false), false);
    }

    internal void Render(ContextDrawerRow row, bool expanded)
    {
        var valid = row.Score.HasValue && double.IsFinite(row.Score.Value)
            && row.Score.Value is >= 0 and <= 100;
        var recommended = valid && row.Recommended;
        _caption.Text = row.Caption;
        _caption.TooltipText = row.Caption;
        _caption.AddThemeColorOverride("font_color", recommended ? Accent : Text);
        _score.Text = valid ? $"{row.Score!.Value:0}" : "--";
        _score.AddThemeColorOverride("font_color", recommended ? Accent : Text);
        _marker.Text = recommended ? "●" : "";
        _style.BorderColor = recommended ? Accent : new Color(0.20F, 0.28F, 0.36F, 0.6F);
        _style.BgColor = recommended ? new Color(0.19F, 0.15F, 0.08F, 0.85F)
            : new Color(0.065F, 0.085F, 0.12F, 0.70F);
        var reasons = valid ? (row.Reasons ?? []).Where(reason => !string.IsNullOrWhiteSpace(reason))
            .Take(3).Select(reason => reason.Length > 240 ? reason[..240] + "…" : reason).ToArray() : [];
        HasReasons = reasons.Length > 0;
        _reasons.Text = string.Join("\n", reasons.Select(reason => "• " + reason));
        _details.Disabled = !HasReasons;
        SetExpanded(expanded);
    }

    internal void SetExpanded(bool expanded)
    {
        _reasons.Visible = expanded && HasReasons;
        _details.Text = _reasons.Visible ? "−" : "+";
        _details.TooltipText = !HasReasons ? "暂无可展示的评分依据"
            : _reasons.Visible ? "收起评分依据" : "展开评分依据（本地规则）";
    }
}
