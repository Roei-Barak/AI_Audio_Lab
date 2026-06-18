using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using KaraokeStudio.WPF.Models;
using LibVLCSharp.Shared;

namespace KaraokeStudio.WPF.Views.Controls;

public partial class VideoPlayerControl : UserControl
{
    private static readonly LibVLC _libVlc = new("--no-video-title-show", "--quiet");
    private MediaPlayer?           _player;
    private readonly DispatcherTimer _timer = new() { Interval = TimeSpan.FromMilliseconds(250) };
    private bool _seeking;
    private ObservableCollection<SubtitleLine>? _lines;

    // ── Events ────────────────────────────────────────────────────────────────

    public event Action<TimeSpan>? PositionChanged;
    public event Action<TimeSpan>? DurationReady;

    // ── Properties ────────────────────────────────────────────────────────────

    public bool IsPlaying => _player?.IsPlaying ?? false;

    public TimeSpan Duration => _player != null
        ? TimeSpan.FromMilliseconds(_player.Length)
        : TimeSpan.Zero;

    public void SetSubtitleLines(ObservableCollection<SubtitleLine>? lines) => _lines = lines;

    public VideoPlayerControl()
    {
        InitializeComponent();
        _timer.Tick += Timer_Tick;
        Unloaded += (_, _) => Dispose();
    }

    // ── Public API ────────────────────────────────────────────────────────────

    public void Open(string filePath)
    {
        Dispose();
        _player = new MediaPlayer(_libVlc);
        VideoView.MediaPlayer = _player;

        using var media = new Media(_libVlc, filePath, FromType.FromPath);
        _player.Play(media);

        _player.LengthChanged += (_, _) =>
        {
            var dur = TimeSpan.FromMilliseconds(_player.Length);
            Dispatcher.Invoke(() =>
            {
                SeekBar.Maximum = dur.TotalSeconds;
                TimeLabel.Text  = $"0:00 / {FormatTime(dur)}";
                Placeholder.Visibility = Visibility.Collapsed;
                DurationReady?.Invoke(dur);
            });
        };

        VolumeSlider_ValueChanged(VolumeSlider, new RoutedPropertyChangedEventArgs<double>(80, VolumeSlider.Value));
        BtnPlayPause.Content = "⏸";
        _timer.Start();
    }

    public void PlayPause()
    {
        if (_player == null) return;
        if (_player.IsPlaying) { _player.Pause(); BtnPlayPause.Content = "▶"; }
        else                   { _player.Play();  BtnPlayPause.Content = "⏸"; }
    }

    public void Seek(TimeSpan position)
    {
        if (_player == null) return;
        _player.Time = (long)position.TotalMilliseconds;
    }

    public void Stop()
    {
        _player?.Stop();
        BtnPlayPause.Content = "▶";
        _timer.Stop();
    }

    public void Dispose()
    {
        _timer.Stop();
        _player?.Stop();
        _player?.Dispose();
        _player = null;
    }

    // ── Timer ─────────────────────────────────────────────────────────────────

    private void Timer_Tick(object? sender, EventArgs e)
    {
        if (_player == null || _seeking) return;

        var pos = TimeSpan.FromMilliseconds(_player.Time);
        var dur = TimeSpan.FromMilliseconds(Math.Max(1, _player.Length));

        _seeking = true;
        SeekBar.Value = pos.TotalSeconds;
        _seeking = false;

        TimeLabel.Text = $"{FormatTime(pos)} / {FormatTime(dur)}";

        // Update subtitle overlay
        if (_lines != null)
        {
            var active = _lines.FirstOrDefault(l => pos >= l.Start && pos < l.End);
            SubtitleOverlay.Text = active?.Text ?? string.Empty;
        }

        PositionChanged?.Invoke(pos);
    }

    // ── Event handlers ────────────────────────────────────────────────────────

    private void BtnPlayPause_Click(object sender, RoutedEventArgs e) => PlayPause();

    private void SeekBar_PreviewMouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        => _seeking = true;

    private void SeekBar_PreviewMouseUp(object sender, System.Windows.Input.MouseButtonEventArgs e)
    {
        if (_player != null)
            _player.Time = (long)(SeekBar.Value * 1000);
        _seeking = false;
    }

    private void SeekBar_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (!_seeking) return;
    }

    private void VolumeSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (_player != null)
            _player.Volume = (int)VolumeSlider.Value;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static string FormatTime(TimeSpan t) =>
        $"{(int)t.TotalMinutes}:{t.Seconds:D2}";
}
