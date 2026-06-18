using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using KaraokeStudio.WPF.Models;

namespace KaraokeStudio.WPF.Views.Controls;

public partial class TimelineControl : UserControl
{
    // ── Dependency Properties ─────────────────────────────────────────────────

    public static readonly DependencyProperty LinesProperty =
        DependencyProperty.Register(nameof(Lines), typeof(ObservableCollection<SubtitleLine>),
            typeof(TimelineControl), new PropertyMetadata(null, OnLinesChanged));

    public static readonly DependencyProperty PlayheadPositionProperty =
        DependencyProperty.Register(nameof(PlayheadPosition), typeof(TimeSpan),
            typeof(TimelineControl), new PropertyMetadata(TimeSpan.Zero, OnPlayheadChanged));

    public static readonly DependencyProperty TotalDurationProperty =
        DependencyProperty.Register(nameof(TotalDuration), typeof(TimeSpan),
            typeof(TimelineControl), new PropertyMetadata(TimeSpan.FromMinutes(5), OnLayoutChanged));

    public static readonly DependencyProperty WaveformSamplesProperty =
        DependencyProperty.Register(nameof(WaveformSamples), typeof(float[]),
            typeof(TimelineControl), new PropertyMetadata(Array.Empty<float>(), OnWaveformChanged));

    public static readonly DependencyProperty SelectedLineProperty =
        DependencyProperty.Register(nameof(SelectedLine), typeof(SubtitleLine),
            typeof(TimelineControl), new PropertyMetadata(null, OnSelectedLineChanged));

    // ── Events ────────────────────────────────────────────────────────────────

    public event Action<TimeSpan>? SeekRequested;
    public event Action<SubtitleLine>? LineSelected;
    public event Action<SubtitleLine>? LineDoubleClicked;

    // ── Properties ────────────────────────────────────────────────────────────

    public ObservableCollection<SubtitleLine>? Lines
    {
        get => (ObservableCollection<SubtitleLine>?)GetValue(LinesProperty);
        set => SetValue(LinesProperty, value);
    }

    public TimeSpan PlayheadPosition
    {
        get => (TimeSpan)GetValue(PlayheadPositionProperty);
        set => SetValue(PlayheadPositionProperty, value);
    }

    public TimeSpan TotalDuration
    {
        get => (TimeSpan)GetValue(TotalDurationProperty);
        set => SetValue(TotalDurationProperty, value);
    }

    public float[] WaveformSamples
    {
        get => (float[])GetValue(WaveformSamplesProperty);
        set => SetValue(WaveformSamplesProperty, value);
    }

    public SubtitleLine? SelectedLine
    {
        get => (SubtitleLine?)GetValue(SelectedLineProperty);
        set => SetValue(SelectedLineProperty, value);
    }

    // ── Private state ─────────────────────────────────────────────────────────

    private double _pixelsPerSecond = 60.0;
    private double _scrollOffset    = 0.0;

    // Drag state
    private enum DragMode { None, MoveBar, ResizeLeft, ResizeRight, Seek }
    private DragMode      _dragMode;
    private SubtitleLine? _dragLine;
    private Point         _dragStart;
    private TimeSpan      _dragOrigStart;
    private TimeSpan      _dragOrigEnd;

    // Visual map: Canvas rect → SubtitleLine
    private readonly Dictionary<Rectangle, SubtitleLine> _rectMap = [];

    // Inline text editor
    private TextBox? _editBox;

    // Colours
    private static readonly SolidColorBrush BrushNormal   = new(Color.FromArgb(180, 33,  150, 243));
    private static readonly SolidColorBrush BrushActive   = new(Color.FromArgb(220, 255, 152,   0));
    private static readonly SolidColorBrush BrushSelected = new(Color.FromArgb(220, 76,  175,  80));
    private static readonly SolidColorBrush BrushText     = Brushes.White;
    private static readonly SolidColorBrush BrushPlayhead = new(Color.FromArgb(220, 244, 67, 54));
    private static readonly SolidColorBrush BrushWave     = new(Color.FromArgb(160, 0, 188, 212));

    public TimelineControl()
    {
        InitializeComponent();
        SizeChanged += (_, _) => Rebuild();
    }

    // ── DP callbacks ─────────────────────────────────────────────────────────

    private static void OnLinesChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        var tc = (TimelineControl)d;
        if (e.OldValue is ObservableCollection<SubtitleLine> old)
            old.CollectionChanged -= tc.Lines_CollectionChanged;
        if (e.NewValue is ObservableCollection<SubtitleLine> neo)
            neo.CollectionChanged += tc.Lines_CollectionChanged;
        tc.Rebuild();
    }

    private static void OnPlayheadChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        => ((TimelineControl)d).UpdatePlayhead();

    private static void OnLayoutChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        => ((TimelineControl)d).Rebuild();

    private static void OnWaveformChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        => ((TimelineControl)d).DrawWaveform();

    private static void OnSelectedLineChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        => ((TimelineControl)d).RefreshBarColours();

    private void Lines_CollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
        => Rebuild();

    // ── Layout ────────────────────────────────────────────────────────────────

    private double TimeToX(TimeSpan t) =>
        (t.TotalSeconds * _pixelsPerSecond) - _scrollOffset;

    private TimeSpan XToTime(double x) =>
        TimeSpan.FromSeconds((x + _scrollOffset) / _pixelsPerSecond);

    private void Rebuild()
    {
        TimelineCanvas.Children.Clear();
        PlayheadCanvas.Children.Clear();
        _rectMap.Clear();
        _editBox = null;

        if (Lines == null) return;

        foreach (var line in Lines)
        {
            double x = TimeToX(line.Start);
            double w = (line.End - line.Start).TotalSeconds * _pixelsPerSecond;
            if (w < 2) w = 2;

            var rect = new Rectangle
            {
                Width        = w,
                Height       = TimelineCanvas.ActualHeight > 0 ? TimelineCanvas.ActualHeight - 6 : 46,
                RadiusX      = 4,
                RadiusY      = 4,
                Fill         = line.IsActive ? BrushActive : (line.IsSelected ? BrushSelected : BrushNormal),
                Stroke       = Brushes.White,
                StrokeThickness = 1,
                ToolTip      = $"{line.Text}\n{FormatTime(line.Start)} → {FormatTime(line.End)}"
            };
            Canvas.SetLeft(rect, x);
            Canvas.SetTop(rect, 3);

            var tb = new TextBlock
            {
                Text             = line.Text.Length > 20 ? line.Text[..20] + "…" : line.Text,
                Foreground       = BrushText,
                FontSize         = 11,
                IsHitTestVisible = false,
                Clip             = new RectangleGeometry(new Rect(0, 0, Math.Max(w - 4, 0), 40))
            };
            Canvas.SetLeft(tb, x + 3);
            Canvas.SetTop(tb, 10);

            _rectMap[rect] = line;
            TimelineCanvas.Children.Add(rect);
            TimelineCanvas.Children.Add(tb);
        }

        DrawRuler();
        UpdatePlayhead();
        DrawWaveform();
    }

    private void RefreshBarColours()
    {
        foreach (var (rect, line) in _rectMap)
        {
            rect.Fill = line.IsActive ? BrushActive
                      : (line == SelectedLine || line.IsSelected) ? BrushSelected
                      : BrushNormal;
        }
    }

    private void DrawRuler()
    {
        RulerCanvas.Children.Clear();
        double step = _pixelsPerSecond >= 30 ? 5 : 10;
        double totalSec = TotalDuration.TotalSeconds;
        for (double s = 0; s <= totalSec; s += step)
        {
            double x = TimeToX(TimeSpan.FromSeconds(s));
            if (x < -10 || x > ActualWidth + 10) continue;

            var tick = new Line
            {
                X1 = x, Y1 = 0, X2 = x, Y2 = 8,
                Stroke = Brushes.Gray, StrokeThickness = 1
            };
            var label = new TextBlock
            {
                Text       = FormatTime(TimeSpan.FromSeconds(s)),
                Foreground = Brushes.DarkGray,
                FontSize   = 10
            };
            Canvas.SetLeft(label, x + 2);
            Canvas.SetTop(label, 4);
            RulerCanvas.Children.Add(tick);
            RulerCanvas.Children.Add(label);
        }
    }

    private void UpdatePlayhead()
    {
        PlayheadCanvas.Children.Clear();
        double x = TimeToX(PlayheadPosition);
        var line = new Line
        {
            X1 = x, Y1 = 0,
            X2 = x, Y2 = PlayheadCanvas.ActualHeight,
            Stroke          = BrushPlayhead,
            StrokeThickness = 2
        };
        PlayheadCanvas.Children.Add(line);
        RefreshBarColours();
    }

    private void DrawWaveform()
    {
        WaveformCanvas.Children.Clear();
        var samples = WaveformSamples;
        if (samples == null || samples.Length == 0) return;

        double w = WaveformCanvas.ActualWidth;
        double h = WaveformCanvas.ActualHeight;
        if (w <= 0 || h <= 0) return;

        int samplesPerPixel = Math.Max(1, samples.Length / (int)w);

        var geo = new StreamGeometry();
        using (var ctx = geo.Open())
        {
            ctx.BeginFigure(new Point(0, h / 2), false, false);
            for (int px = 0; px < (int)w; px++)
            {
                int si = px * samplesPerPixel;
                if (si >= samples.Length) break;
                float maxAmp = 0;
                for (int j = si; j < Math.Min(si + samplesPerPixel, samples.Length); j++)
                    maxAmp = Math.Max(maxAmp, Math.Abs(samples[j]));
                double y = h / 2 - (maxAmp * h / 2 * 0.9);
                ctx.LineTo(new Point(px, y), true, false);
            }
            for (int px = (int)w - 1; px >= 0; px--)
            {
                int si = px * samplesPerPixel;
                if (si >= samples.Length) break;
                float maxAmp = 0;
                for (int j = si; j < Math.Min(si + samplesPerPixel, samples.Length); j++)
                    maxAmp = Math.Max(maxAmp, Math.Abs(samples[j]));
                double y = h / 2 + (maxAmp * h / 2 * 0.9);
                ctx.LineTo(new Point(px, y), true, false);
            }
        }
        geo.Freeze();

        WaveformCanvas.Children.Add(new System.Windows.Shapes.Path
        {
            Data            = geo,
            Fill            = BrushWave,
            Stroke          = BrushWave,
            StrokeThickness = 1
        });
    }

    // ── Mouse interaction ─────────────────────────────────────────────────────

    private SubtitleLine? HitTest(Point p, out DragMode mode)
    {
        mode = DragMode.None;
        foreach (var (rect, line) in _rectMap)
        {
            double left  = Canvas.GetLeft(rect);
            double right = left + rect.Width;
            double top   = Canvas.GetTop(rect);
            double bot   = top + rect.Height;

            if (p.Y < top || p.Y > bot) continue;
            if (p.X < left - 4 || p.X > right + 4) continue;

            if (Math.Abs(p.X - left)  < 8) { mode = DragMode.ResizeLeft;  return line; }
            if (Math.Abs(p.X - right) < 8) { mode = DragMode.ResizeRight; return line; }
            if (p.X >= left && p.X <= right) { mode = DragMode.MoveBar; return line; }
        }
        return null;
    }

    private void TimelineCanvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        CommitEdit();
        var pos = e.GetPosition(TimelineCanvas);
        var hit = HitTest(pos, out var mode);

        if (hit != null)
        {
            _dragLine      = hit;
            _dragMode      = mode;
            _dragStart     = pos;
            _dragOrigStart = hit.Start;
            _dragOrigEnd   = hit.End;
            SelectedLine   = hit;
            LineSelected?.Invoke(hit);
            TimelineCanvas.CaptureMouse();

            if (e.ClickCount == 2)
            {
                _dragMode = DragMode.None;
                TimelineCanvas.ReleaseMouseCapture();
                OpenInlineEditor(hit);
                LineDoubleClicked?.Invoke(hit);
            }
        }
        else
        {
            _dragMode = DragMode.Seek;
            var t = XToTime(pos.X);
            SeekRequested?.Invoke(t);
            TimelineCanvas.CaptureMouse();
        }
    }

    private void TimelineCanvas_MouseMove(object sender, MouseEventArgs e)
    {
        var pos = e.GetPosition(TimelineCanvas);

        if (_dragMode == DragMode.Seek && e.LeftButton == MouseButtonState.Pressed)
        {
            SeekRequested?.Invoke(XToTime(pos.X));
            return;
        }

        if (_dragLine == null || _dragMode == DragMode.None || e.LeftButton != MouseButtonState.Pressed)
        {
            // Update cursor
            var hit = HitTest(pos, out var m);
            TimelineCanvas.Cursor = m switch
            {
                DragMode.ResizeLeft  or DragMode.ResizeRight => Cursors.SizeWE,
                DragMode.MoveBar     => Cursors.SizeAll,
                _                    => Cursors.Arrow
            };
            return;
        }

        double dx   = pos.X - _dragStart.X;
        double dSec = dx / _pixelsPerSecond;

        switch (_dragMode)
        {
            case DragMode.MoveBar:
                var newStart = _dragOrigStart + TimeSpan.FromSeconds(dSec);
                var newEnd   = _dragOrigEnd   + TimeSpan.FromSeconds(dSec);
                if (newStart >= TimeSpan.Zero)
                {
                    _dragLine.Start = newStart;
                    _dragLine.End   = newEnd;
                }
                break;

            case DragMode.ResizeLeft:
                var ns = _dragOrigStart + TimeSpan.FromSeconds(dSec);
                if (ns >= TimeSpan.Zero && ns < _dragLine.End - TimeSpan.FromMilliseconds(200))
                    _dragLine.Start = ns;
                break;

            case DragMode.ResizeRight:
                var ne = _dragOrigEnd + TimeSpan.FromSeconds(dSec);
                if (ne > _dragLine.Start + TimeSpan.FromMilliseconds(200))
                    _dragLine.End = ne;
                break;
        }

        Rebuild();
    }

    private void TimelineCanvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        _dragMode = DragMode.None;
        _dragLine = null;
        TimelineCanvas.ReleaseMouseCapture();
    }

    private void TimelineCanvas_MouseWheel(object sender, MouseWheelEventArgs e)
    {
        if (Keyboard.Modifiers == ModifierKeys.Control)
        {
            // Zoom
            _pixelsPerSecond = Math.Clamp(_pixelsPerSecond + (e.Delta > 0 ? 10 : -10), 10, 300);
        }
        else
        {
            // Scroll
            _scrollOffset = Math.Max(0, _scrollOffset - e.Delta / 3.0);
        }
        Rebuild();
    }

    private void TimelineCanvas_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
    {
        // Right-click = context menu (future: split, delete…)
        CommitEdit();
    }

    // ── Inline editing ────────────────────────────────────────────────────────

    private void OpenInlineEditor(SubtitleLine line)
    {
        CommitEdit();

        double x = TimeToX(line.Start);
        double w = Math.Max((line.End - line.Start).TotalSeconds * _pixelsPerSecond, 80);

        _editBox = new TextBox
        {
            Text       = line.Text,
            Width      = w,
            Height     = 28,
            FontSize   = 12,
            Background = new SolidColorBrush(Color.FromArgb(230, 30, 30, 30)),
            Foreground = Brushes.White,
            BorderBrush = BrushSelected,
            Tag        = line
        };
        _editBox.KeyDown += EditBox_KeyDown;
        Canvas.SetLeft(_editBox, x);
        Canvas.SetTop(_editBox, 16);

        TimelineCanvas.Children.Add(_editBox);
        _editBox.Focus();
        _editBox.SelectAll();
    }

    private void EditBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key is Key.Enter or Key.Escape) CommitEdit();
    }

    private void CommitEdit()
    {
        if (_editBox == null) return;
        if (_editBox.Tag is SubtitleLine line)
            line.Text = _editBox.Text;
        TimelineCanvas.Children.Remove(_editBox);
        _editBox = null;
        Rebuild();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static string FormatTime(TimeSpan t) =>
        $"{(int)t.TotalMinutes}:{t.Seconds:D2}";
}
