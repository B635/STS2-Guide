using Godot;
using STS2Guide.ReadOnlyExporter;

public partial class Main : Control
{
    public override async void _Ready()
    {
        try
        {
            var preview = OS.GetCmdlineUserArgs().FirstOrDefault(arg => arg.StartsWith("--preview="));
            if (preview is not null)
            {
                await SavePreview(preview[10..]);
                GetTree().Quit();
                return;
            }
            await CheckLayout(4, false, 1080);
            await CheckLayout(6, false, 720);
            await CheckLayout(3, true, 1080);
            await CheckLayout(4, false, 1080, largeFont: true);
            await CheckLayout(12, false, 300);
            await CheckReasonsAndNames();
            await CheckExpandedScrollingAndNarrowRoute();
            GD.Print("DRAWER_LAYOUT_PASS");
            GetTree().Quit();
        }
        catch (Exception exception)
        {
            GD.PrintErr(exception);
            GetTree().Quit(1);
        }
    }

    private async Task CheckLayout(int count, bool route, int viewportHeight, bool largeFont = false)
    {
        GetWindow().Size = new Vector2I(1920, viewportHeight);
        var captions = Enumerable.Range(0, count)
            .Select(i => i == count - 1 ? "跳过" : "剑柄打击").ToArray();
        var handle = ContextDrawer.Show(this, null, "选牌建议", captions,
            route ? GuideRouteModes.Balanced : null, route ? _ => true : null);
        ContextDrawer.Render(handle, captions.Select(c => new ContextDrawerRow(c, 32, false)).ToArray());
        if (largeFont)
        {
            var productionPanel = GetNode<PanelContainer>("STS2GuideContextDrawer");
            var productionScroll = (ScrollContainer)productionPanel.FindChild(
                "STS2GuideContextDrawerScroll", true, false);
            foreach (var label in productionScroll.FindChildren("*", "Label", true, false).OfType<Label>())
                label.AddThemeFontSizeOverride("font_size", 30);
            ContextDrawer.Render(handle, captions.Select(c => new ContextDrawerRow(c, null, false)).ToArray(),
                AdviceAvailability.DataMissing);
        }
        for (var frame = 0; frame < 8; frame++)
        {
            await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
            ContextDrawer.Reposition(handle);
        }
        var panel = GetNode<PanelContainer>("STS2GuideContextDrawer");
        var scroll = panel.FindChild("STS2GuideContextDrawerScroll", true, false) as ScrollContainer
            ?? throw new Exception("Missing production scroll container");
        var rows = scroll.GetChild<VBoxContainer>(0);
        var last = rows.GetChild<Control>(count - 1);
        var bottom = last.GetGlobalRect().End.Y;
        GD.Print($"rows={count}, route={route}, viewport={viewportHeight}, panel={panel.Size.Y}, "
            + $"scroll={scroll.Size.Y}, content={rows.Size.Y}, clipped={bottom - scroll.GetGlobalRect().End.Y}");
        if (viewportHeight >= 720)
        {
            Require(bottom <= scroll.GetGlobalRect().End.Y + 1, "Last row clipped with ample viewport space");
            Require(!scroll.GetVScrollBar().Visible, "Unnecessary scrollbar");
        }
        else
        {
            Require(panel.GetGlobalRect().Position.Y >= 9, "Panel escapes viewport top");
            Require(panel.GetGlobalRect().End.Y <= viewportHeight - 9, "Panel escapes viewport bottom");
            Require(scroll.GetVScrollBar().Visible, "Small viewport needs scrolling");
            scroll.ScrollVertical = 10000;
            await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
            Require(last.GetGlobalRect().End.Y <= scroll.GetGlobalRect().End.Y + 1,
                "Cannot scroll to final row");
            GetWindow().Size = new Vector2I(1920, 1080);
            for (var frame = 0; frame < 8; frame++)
            {
                await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
                ContextDrawer.Reposition(handle);
            }
            Require(!scroll.GetVScrollBar().Visible, "Scrollbar remains after enlarging viewport");
            Require(last.GetGlobalRect().End.Y <= scroll.GetGlobalRect().End.Y + 1,
                "Resized panel still clips final row");
        }
        ContextDrawer.Hide(handle);
        await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
    }

    private async Task Settle(ContextDrawerHandle handle)
    {
        for (var frame = 0; frame < 10; frame++)
        {
            await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
            ContextDrawer.Reposition(handle);
        }
    }

    private async Task CheckReasonsAndNames()
    {
        GetWindow().Size = new Vector2I(800, 600);
        var names = new[] { "这是一张特别特别长的中文卡牌名称", "另一张牌", "跳过" };
        var handle = ContextDrawer.Show(this, null, "选牌建议", names);
        var values = names.Select((name, i) => new ContextDrawerRow(name, i == 0 ? 100 : 61.4 - i, i == 0,
            new[] { "与当前牌组协同。", "这是一条较长的真实评分依据，用来验证原因会换行而不是撑破侧栏边界。" })).ToArray();
        ContextDrawer.Render(handle, values);
        await Settle(handle);
        var panel = GetNode<PanelContainer>("STS2GuideContextDrawer");
        T Node<T>(string name) where T : Node => (T)panel.FindChild(name, true, false);
        var caption = Node<Label>("CandidateCaption0");
        var score = Node<Label>("CandidateScore0");
        var first = Node<Button>("CandidateDetails0");
        var second = Node<Button>("CandidateDetails1");
        var reason0 = Node<Label>("CandidateReasons0");
        var reason1 = Node<Label>("CandidateReasons1");
        Require(panel.Size.X <= 240.5F, "Long name exceeded viewport width budget");
        Require(caption.GetGlobalRect().End.X <= score.GetGlobalRect().Position.X, "Name overlaps score");
        Require(score.Text == "100.0" && caption.TooltipText == names[0], "Score or full name lost");
        Require(caption.MouseFilter == MouseFilterEnum.Stop && first.MouseFilter == MouseFilterEnum.Stop,
            "Caption tooltip or explanation button cannot receive pointer events");
        Require(!reason0.Visible && !reason1.Visible, "Reasons must start collapsed");
        first.EmitSignal(BaseButton.SignalName.Pressed);
        await Settle(handle);
        Require(reason0.Visible && reason0.Text.Contains("当前牌组"), "Explanation did not open");
        Require(reason0.GetGlobalRect().End.X <= panel.GetGlobalRect().End.X, "Explanation overflows width");
        second.EmitSignal(BaseButton.SignalName.Pressed);
        await Settle(handle);
        Require(!reason0.Visible && reason1.Visible, "Only one explanation may be open");
        ContextDrawer.Render(handle, names.Select(name => new ContextDrawerRow(name, null, true,
            new[] { "stale reason" })).ToArray(), AdviceAvailability.Unavailable);
        await Settle(handle);
        Require(!reason1.Visible && reason1.Text == "" && second.Disabled, "Stale explanation remains");
        Require(Node<Label>("CandidateMarker0").Text == "", "Missing score retains recommended marker");
        ContextDrawer.Render(handle, values);
        await Settle(handle);
        Require(!reason0.Visible && !reason1.Visible && !first.Disabled, "Recovery must be collapsed and usable");
        var partial = new[] { values[0], new ContextDrawerRow(names[1], null, false), values[2] };
        ContextDrawer.Render(handle, partial, AdviceAvailability.Unavailable);
        Require(score.Text == "100.0" && !first.Disabled && second.Disabled,
            "One unavailable candidate erased another candidate's validated score/reasons");
        ContextDrawer.Hide(handle);
        var replacement = ContextDrawer.Show(this, null, "新决策", names);
        ContextDrawer.Render(replacement, values);
        first.EmitSignal(BaseButton.SignalName.Pressed); // Old node is queued, not freed yet.
        ContextDrawer.Render(handle, values);
        await Settle(replacement);
        panel = GetChildren().OfType<PanelContainer>().Single();
        Require(!Node<Label>("CandidateReasons0").Visible, "Old callback changed new decision");
        ContextDrawer.Render(replacement, names.Select(name => new ContextDrawerRow(name, null, false)).ToArray());
        ContextDrawer.Render(handle, values);
        Require(Node<Label>("CandidateScore0").Text == "--", "Old handle changed new decision");
        ContextDrawer.Hide(replacement);
        await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
    }

    private async Task CheckExpandedScrollingAndNarrowRoute()
    {
        GetWindow().Size = new Vector2I(640, 300);
        var names = Enumerable.Range(0, 12).Select(i => "路线候选" + i).ToArray();
        var handle = ContextDrawer.Show(this, null, "路线建议", names, GuideRouteModes.Balanced, _ => true);
        ContextDrawer.Render(handle, names.Select(name => new ContextDrawerRow(name, 58, false,
            new[] { new string('长', 150), "当前生命偏低，休整机会更有价值。" })).ToArray());
        await Settle(handle);
        var panel = GetNode<PanelContainer>("STS2GuideContextDrawer");
        Require(panel.Size.X <= 192.5F, "Route controls exceed narrow viewport budget");
        var button = (Button)panel.FindChild("CandidateDetails0", true, false);
        button.EmitSignal(BaseButton.SignalName.Pressed);
        await Settle(handle);
        var scroll = (ScrollContainer)panel.FindChild("STS2GuideContextDrawerScroll", true, false);
        Require(panel.GetGlobalRect().End.Y <= 291, "Expanded panel escapes viewport");
        scroll.ScrollVertical = 10000;
        await Settle(handle);
        var last = (Control)panel.FindChild("CandidateRow11", true, false);
        Require(last.GetGlobalRect().End.Y <= scroll.GetGlobalRect().End.Y + 1, "Expanded reasons hide final candidate");
        button.EmitSignal(BaseButton.SignalName.Pressed);
        GetWindow().Size = new Vector2I(1920, 1080);
        await Settle(handle);
        Require(!scroll.GetVScrollBar().Visible, "Collapsed/resized list remains artificially tall");
        var oldMode = (Button)panel.FindChild("STS2GuideRouteMode_growth", true, false);
        ContextDrawer.Hide(handle);
        var modeChanges = 0;
        var next = ContextDrawer.Show(this, null, "新路线", new[] { "路线" }, GuideRouteModes.Balanced,
            _ => { modeChanges++; return true; });
        oldMode.EmitSignal(BaseButton.SignalName.Pressed);
        Require(modeChanges == 0, "Old route button modified new decision preferences");
        await Settle(next);
        var nextPanel = GetChildren().OfType<PanelContainer>().Single();
        ((Button)nextPanel.FindChild("STS2GuideRouteMode_growth", true, false)).EmitSignal(BaseButton.SignalName.Pressed);
        Require(modeChanges == 1, "Current route mode control stopped working");
        ContextDrawer.Hide(next);
        await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }

    private async Task SavePreview(string path)
    {
        GetWindow().Size = new Vector2I(1280, 720);
        var names = new[] { "剑柄打击", "上勾拳", "愤怒", "跳过" };
        var handle = ContextDrawer.Show(this, null, "选牌建议", names);
        ContextDrawer.Render(handle, new[] {
            new ContextDrawerRow(names[0], 68, true, new[] { "当前牌组需要额外过牌。", "低费攻击有助于补足当前输出。" }),
            new ContextDrawerRow(names[1], 56, false, new[] { "提供虚弱与易伤效果。" }),
            new ContextDrawerRow(names[2], 47, false), new ContextDrawerRow(names[3], 42, false),
        });
        await Settle(handle);
        var panel = GetNode<PanelContainer>("STS2GuideContextDrawer");
        ((Button)panel.FindChild("CandidateDetails0", true, false)).EmitSignal(BaseButton.SignalName.Pressed);
        await Settle(handle);
        await ToSignal(RenderingServer.Singleton, RenderingServer.SignalName.FramePostDraw);
        var image = GetViewport().GetTexture().GetImage();
        Require(image is not null && !image.IsEmpty(), "Preview renderer returned no image");
        var area = new Rect2I(0, Math.Max(0, (int)panel.Position.Y - 24),
            (int)panel.Size.X + 70, (int)panel.Size.Y + 48);
        Require(image!.GetRegion(area).SavePng(path) == Error.Ok, "Cannot save preview");
        GD.Print("DRAWER_PREVIEW_PASS (fixture, not a game screenshot)");
        ContextDrawer.Hide(handle);
    }
}
