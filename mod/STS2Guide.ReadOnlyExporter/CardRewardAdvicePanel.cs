using System.Reflection;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

namespace STS2Guide.ReadOnlyExporter;

internal static class CardRewardAdvicePanel
{
    private const float MinSidebarWidth = 160F;
    private const float MaxSidebarWidth = 280F;
    private const float SidebarViewportRatio = 0.18F;
    private const float MaxViewportCoverage = 0.3F;
    private const float RowHeight = 32F;
    private const float HeaderHeight = 42F;
    private const float VerticalPadding = 24F;
    private const float VerticalEdgeMargin = 10F;
    private const float ToggleWidth = 30F;
    private const float ToggleHeight = 48F;
    private const float ToggleOverlap = 2F;
    private const double SlideDuration = 0.22;
    private static readonly object Gate = new();
    private static readonly FieldInfo? CardRowField =
        typeof(NCardRewardSelectionScreen).GetField(
            "_cardRow",
            BindingFlags.NonPublic | BindingFlags.Instance
        );

    private static NCardRewardSelectionScreen? _screen;
    private static Control? _cardRow;
    private static PanelContainer? _panel;
    private static Button? _toggleButton;
    private static Tween? _slideTween;
    private static Godot.Timer? _timer;
    private static List<Label> _scoreLabels = [];
    private static List<string> _cardNames = [];
    private static PendingDecisionView? _pending;
    private static long _lastWriteTicks;
    private static int _recommendedIndex = -1;
    private static bool _isCollapsed;
    private static bool _isSliding;

    internal static void Show(NCardRewardSelectionScreen screen)
    {
        lock (Gate)
        {
            HideInternal();
            _pending = StateEventWriter.GetPendingDecision();
            var cardCount = _pending?.CardIds.Count ?? 0;
            if (_pending is null || cardCount == 0)
            {
                Log.Info(
                    "[STS2-Guide] Advice panel skipped: no pending card options."
                );
                return;
            }

            _screen = screen;
            _cardRow = CardRowField?.GetValue(screen) as Control;
            _cardNames = Enumerable
                .Range(1, cardCount)
                .Select(index => $"卡牌 {index}")
                .ToList();
            _panel = BuildPanel();
            screen.AddChild(_panel);
            _toggleButton = BuildToggleButton();
            screen.AddChild(_toggleButton);
            _isCollapsed = false;
            PlacePanel();

            _timer = new Godot.Timer
            {
                WaitTime = 0.1,
                OneShot = false,
                Autostart = true,
            };
            _timer.Timeout += PollAdvice;
            _panel.AddChild(_timer);
            _lastWriteTicks = 0;
            _recommendedIndex = -1;
            var placeholder = new double?[cardCount + 1];
            Render(placeholder);
        }
    }

    internal static void Hide()
    {
        lock (Gate)
        {
            HideInternal();
        }
    }

    private static void HideInternal()
    {
        StopSlideTween();
        if (_timer is not null)
        {
            _timer.Stop();
            _timer.Timeout -= PollAdvice;
        }
        if (_toggleButton is not null)
        {
            if (GodotObject.IsInstanceValid(_toggleButton))
            {
                _toggleButton.Pressed -= ToggleSidebar;
                _toggleButton.QueueFree();
            }
        }
        if (_panel is not null && GodotObject.IsInstanceValid(_panel))
        {
            _panel.QueueFree();
        }
        _screen = null;
        _cardRow = null;
        _panel = null;
        _toggleButton = null;
        _slideTween = null;
        _timer = null;
        _scoreLabels = [];
        _cardNames = [];
        _pending = null;
        _lastWriteTicks = 0;
        _recommendedIndex = -1;
        _isCollapsed = false;
        _isSliding = false;
    }

    private static PanelContainer BuildPanel()
    {
        var cardCount = _pending?.CardIds.Count ?? 0;
        var rowCount = cardCount + 1;
        var panelHeight = HeaderHeight
            + rowCount * RowHeight
            + VerticalPadding;
        var panel = new PanelContainer
        {
            Name = "STS2GuideCardRewardAdvice",
            MouseFilter = Control.MouseFilterEnum.Ignore,
            CustomMinimumSize = new Vector2(
                MinSidebarWidth,
                panelHeight
            ),
            Size = new Vector2(MinSidebarWidth, panelHeight),
            ZIndex = 100,
        };
        var style = new StyleBoxFlat
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
        };
        panel.AddThemeStyleboxOverride("panel", style);

        var column = new VBoxContainer
        {
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        column.AddThemeConstantOverride("separation", 5);
        panel.AddChild(column);

        var title = new Label
        {
            Text = "选牌建议",
            HorizontalAlignment = HorizontalAlignment.Center,
            MouseFilter = Control.MouseFilterEnum.Ignore,
            CustomMinimumSize = new Vector2(0F, HeaderHeight - 10F),
        };
        title.AddThemeFontSizeOverride("font_size", 20);
        title.AddThemeColorOverride(
            "font_color",
            new Color(0.78F, 0.88F, 1F)
        );
        column.AddChild(title);

        _scoreLabels = [];
        for (var idx = 0; idx < cardCount; idx++)
        {
            var caption = CardCaption(idx);
            var label = new Label
            {
                Text = $"{caption}   --",
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                MouseFilter = Control.MouseFilterEnum.Ignore,
                CustomMinimumSize = new Vector2(0F, RowHeight),
            };
            label.AddThemeFontSizeOverride("font_size", 18);
            column.AddChild(label);
            _scoreLabels.Add(label);
        }
        // Skip row (always last)
        {
            var label = new Label
            {
                Text = "跳过   --",
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
                MouseFilter = Control.MouseFilterEnum.Ignore,
                CustomMinimumSize = new Vector2(0F, RowHeight),
            };
            label.AddThemeFontSizeOverride("font_size", 18);
            column.AddChild(label);
            _scoreLabels.Add(label);
        }
        return panel;
    }

    private static Button BuildToggleButton()
    {
        var button = new Button
        {
            Name = "STS2GuideAdviceToggle",
            Text = "‹",
            TooltipText = "收起选牌建议",
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
        button.Pressed += ToggleSidebar;
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

    private static void ToggleSidebar()
    {
        lock (Gate)
        {
            if (_screen is null
                || _panel is null
                || _toggleButton is null
                || !GodotObject.IsInstanceValid(_screen)
                || !GodotObject.IsInstanceValid(_panel)
                || !GodotObject.IsInstanceValid(_toggleButton))
            {
                return;
            }

            _isCollapsed = !_isCollapsed;
            _toggleButton.Text = _isCollapsed ? "›" : "‹";
            _toggleButton.TooltipText = _isCollapsed
                ? "展开选牌建议"
                : "收起选牌建议";
            PlacePanel(animate: true);
        }
    }

    private static void PollAdvice()
    {
        lock (Gate)
        {
            if (_screen is null
                || _panel is null
                || _pending is null
                || !GodotObject.IsInstanceValid(_screen)
                || !GodotObject.IsInstanceValid(_panel))
            {
                HideInternal();
                return;
            }
            PlacePanel();

            var advicePath = ProjectSettings.GlobalizePath(
                "user://STS2Guide/advice-event.json"
            );
            if (!File.Exists(advicePath))
            {
                return;
            }
            long writeTicks;
            try
            {
                writeTicks = File.GetLastWriteTimeUtc(advicePath).Ticks;
            }
            catch
            {
                return;
            }
            if (writeTicks == _lastWriteTicks)
            {
                return;
            }
            _lastWriteTicks = writeTicks;

            try
            {
                var scores = ReadMatchingScores(
                    advicePath,
                    _pending
                );
                if (scores is not null)
                {
                    Render(scores);
                }
            }
            catch (Exception exception)
            {
                Log.Error(
                    "[STS2-Guide] Advice panel read failed: "
                    + exception.Message
                );
                var empty = new double?[_scoreLabels.Count];
                Render(empty);
            }
        }
    }

    private static double?[]? ReadMatchingScores(
        string path,
        PendingDecisionView pending
    )
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;
        if (!TryReadString(root, "run_id", out var runId)
            || !TryReadString(root, "event_id", out var eventId)
            || runId != pending.RunId
            || eventId != pending.EventId
            || !TryReadString(root, "event_type", out var eventType)
            || eventType != "card_reward"
            || !TryReadString(root, "status", out var status)
            || status != "processed"
            || !root.TryGetProperty("advice", out var advice)
            || advice.ValueKind != JsonValueKind.Object
            || !advice.TryGetProperty(
                "recommendations",
                out var recommendations
            )
            || recommendations.ValueKind != JsonValueKind.Array)
        {
            return null;
        }

        var cardCount = pending.CardIds.Count;
        var scores = new double?[cardCount + 1];
        var cardNames = new string[cardCount];
        var matched = new bool[cardCount];
        var knownCount = 0;
        var bestCardIndex = -1;
        var bestCardRank = int.MaxValue;
        foreach (var recommendation in recommendations.EnumerateArray())
        {
            if (!recommendation.TryGetProperty(
                    "option_index",
                    out var indexElement
                )
                || !indexElement.TryGetInt32(out var optionIndex)
                || optionIndex < 0
                || optionIndex >= cardCount
                || !TryReadString(
                    recommendation,
                    "card_id",
                    out var cardId
                )
                || !string.Equals(
                    cardId,
                    pending.CardIds[optionIndex],
                    StringComparison.OrdinalIgnoreCase
                ))
            {
                return null;
            }
            matched[optionIndex] = true;
            if (recommendation.TryGetProperty(
                    "rank",
                    out var rankElement
                )
                && rankElement.TryGetInt32(out var rank)
                && rank < bestCardRank)
            {
                bestCardRank = rank;
                bestCardIndex = optionIndex;
            }
            cardNames[optionIndex] = TryReadString(
                recommendation,
                "card",
                out var cardName
            )
                ? cardName
                : pending.CardIds[optionIndex];
            var known = recommendation.TryGetProperty(
                "known",
                out var knownElement
            ) && knownElement.ValueKind == JsonValueKind.True;
            if (known
                && recommendation.TryGetProperty(
                    "score",
                    out var scoreElement
                )
                && scoreElement.TryGetDouble(out var score))
            {
                scores[optionIndex] = score;
                knownCount++;
            }
        }
        if (matched.Any(value => !value))
        {
            return null;
        }
        _cardNames = cardNames.ToList();

        if (advice.TryGetProperty(
                "recommended_option_index",
                out var recommendedIndexElement
            )
            && recommendedIndexElement.TryGetInt32(
                out var recommendedIndex
            )
            && recommendedIndex >= 0
            && recommendedIndex <= cardCount)
        {
            _recommendedIndex = recommendedIndex;
        }
        else
        {
            var skipRecommended = advice.TryGetProperty(
                "skip_recommended",
                out var skipRecommendedElement
            ) && skipRecommendedElement.ValueKind == JsonValueKind.True;
            _recommendedIndex = skipRecommended
                ? cardCount
                : bestCardIndex;
        }

        if (knownCount > 0
            && advice.TryGetProperty(
                "skip_candidate",
                out var skipCandidate
            )
            && skipCandidate.TryGetProperty(
                "score",
                out var skipScore
            )
            && skipScore.TryGetDouble(out var skip))
        {
            scores[cardCount] = skip;
        }
        return scores;
    }

    private static bool TryReadString(
        JsonElement element,
        string property,
        out string value
    )
    {
        value = "";
        if (!element.TryGetProperty(property, out var child)
            || child.ValueKind != JsonValueKind.String)
        {
            return false;
        }
        value = child.GetString() ?? "";
        return value.Length > 0;
    }

    private static void Render(IReadOnlyList<double?> scores)
    {
        var rowCount = scores.Count;
        if (_scoreLabels.Count != rowCount || rowCount < 2)
        {
            return;
        }

        var cardCount = rowCount - 1; // last is always skip
        var bestIndex = _recommendedIndex >= 0
            && _recommendedIndex < scores.Count
            && scores[_recommendedIndex].HasValue
                ? _recommendedIndex
                : -1;

        for (var index = 0; index < rowCount; index++)
        {
            var isSkip = index == cardCount;
            string caption;
            if (isSkip)
            {
                caption = "跳过";
            }
            else
            {
                caption = CardCaption(index);
            }

            var score = scores[index];
            _scoreLabels[index].Text = score.HasValue
                ? $"{caption}   {Math.Clamp(score.Value, 0, 100):0}"
                : $"{caption}   --";
            _scoreLabels[index].AddThemeColorOverride(
                "font_color",
                index == bestIndex
                    ? new Color(1F, 0.78F, 0.24F)
                    : new Color(0.92F, 0.94F, 0.98F)
            );
        }
    }

    private static void PlacePanel(bool animate = false)
    {
        if (_screen is null
            || _panel is null
            || _toggleButton is null
            || !GodotObject.IsInstanceValid(_screen)
            || !GodotObject.IsInstanceValid(_panel)
            || !GodotObject.IsInstanceValid(_toggleButton))
        {
            return;
        }
        if (_isSliding && !animate)
        {
            return;
        }

        var viewportRect = _screen.GetViewport().GetVisibleRect();
        var maximumWidth = Math.Max(
            MinSidebarWidth,
            Math.Min(
                MaxSidebarWidth,
                viewportRect.Size.X * MaxViewportCoverage
            )
        );
        var contentWidth = _panel.GetCombinedMinimumSize().X;
        var panelWidth = Math.Clamp(
            Math.Max(
                viewportRect.Size.X * SidebarViewportRatio,
                contentWidth
            ),
            MinSidebarWidth,
            maximumWidth
        );
        var panelHeight = Math.Max(
            _panel.Size.Y,
            _panel.CustomMinimumSize.Y
        );
        _panel.Size = new Vector2(panelWidth, panelHeight);
        var y = viewportRect.Position.Y
            + (viewportRect.Size.Y - panelHeight) / 2F;
        if (_cardRow is not null && GodotObject.IsInstanceValid(_cardRow))
        {
            var cardRect = _cardRow.GetGlobalRect();
            y = cardRect.GetCenter().Y - panelHeight / 2F;
        }
        var panelY = Math.Clamp(
            y,
            viewportRect.Position.Y + VerticalEdgeMargin,
            Math.Max(
                viewportRect.Position.Y + VerticalEdgeMargin,
                viewportRect.End.Y
                    - panelHeight
                    - VerticalEdgeMargin
            )
        );
        var expandedX = viewportRect.Position.X;
        var panelX = _isCollapsed
            ? expandedX - panelWidth
            : expandedX;
        var panelTarget = new Vector2(panelX, panelY);
        var buttonTarget = new Vector2(
            panelX + panelWidth - ToggleOverlap,
            panelY + (panelHeight - ToggleHeight) / 2F
        );

        if (!animate)
        {
            _panel.GlobalPosition = panelTarget;
            _toggleButton.GlobalPosition = buttonTarget;
            return;
        }

        StopSlideTween();
        _isSliding = true;
        _slideTween = _screen.CreateTween();
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
            _toggleButton,
            "global_position",
            buttonTarget,
            SlideDuration
        );
        _slideTween.Finished += FinishSlide;
    }

    private static void StopSlideTween()
    {
        if (_slideTween is not null
            && GodotObject.IsInstanceValid(_slideTween))
        {
            _slideTween.Finished -= FinishSlide;
            _slideTween.Kill();
        }
        _slideTween = null;
        _isSliding = false;
    }

    private static void FinishSlide()
    {
        lock (Gate)
        {
            if (_slideTween is not null
                && GodotObject.IsInstanceValid(_slideTween))
            {
                _slideTween.Finished -= FinishSlide;
            }
            _slideTween = null;
            _isSliding = false;
            PlacePanel();
        }
    }

    private static string CardCaption(int index)
    {
        return index >= 0 && index < _cardNames.Count
            ? _cardNames[index]
            : $"卡牌 {index + 1}";
    }
}
