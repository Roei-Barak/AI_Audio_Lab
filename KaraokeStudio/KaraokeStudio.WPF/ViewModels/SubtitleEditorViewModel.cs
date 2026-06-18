using System.Collections.ObjectModel;
using System.IO;
using System.Text;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KaraokeStudio.WPF.Models;
using KaraokeStudio.WPF.Services;
using Microsoft.Win32;

namespace KaraokeStudio.WPF.ViewModels;

public partial class SubtitleEditorViewModel : ObservableObject
{
    private readonly ApiService _api;

    // ── State ─────────────────────────────────────────────────────────────────
    [ObservableProperty] private string?   _assFilePath;
    [ObservableProperty] private string?   _videoFilePath;
    [ObservableProperty] private string?   _audioFilePath;
    [ObservableProperty] private TimeSpan  _playheadPosition;
    [ObservableProperty] private TimeSpan  _videoDuration;
    [ObservableProperty] private bool      _isPlaying;
    [ObservableProperty] private SubtitleLine? _selectedLine;
    [ObservableProperty] private string    _statusText = "טען קובץ ASS להתחלה";
    [ObservableProperty] private bool      _isRendering;
    [ObservableProperty] private float[]   _waveformSamples = [];

    // Search
    [ObservableProperty] private string _searchQuery = string.Empty;
    [ObservableProperty] private string _replaceWith = string.Empty;

    // Style
    [ObservableProperty] private int    _fontSize  = 80;
    [ObservableProperty] private string _colorHex  = "#FFFFFF";
    [ObservableProperty] private string _position  = "bottom";

    public ObservableCollection<SubtitleLine> Lines { get; } = [];

    // Undo / Redo
    private readonly Stack<List<SubtitleLine>> _undoStack = new();
    private readonly Stack<List<SubtitleLine>> _redoStack = new();

    public SubtitleEditorViewModel(ApiService api) => _api = api;

    // ── File I/O ──────────────────────────────────────────────────────────────

    [RelayCommand]
    private async Task LoadAssAsync(string? path = null)
    {
        if (path is null)
        {
            var dlg = new OpenFileDialog { Filter = "ASS Files|*.ass|All Files|*.*" };
            if (dlg.ShowDialog() != true) return;
            path = dlg.FileName;
        }

        AssFilePath = path;
        Lines.Clear();
        _undoStack.Clear();
        _redoStack.Clear();

        foreach (var line in File.ReadLines(path, Encoding.UTF8))
        {
            if (line.StartsWith("Dialogue:"))
                Lines.Add(SubtitleLine.FromAssLine(line));
        }

        StatusText = $"נטען: {Lines.Count} שורות";

        // Load waveform if audio is set
        if (AudioFilePath != null) await LoadWaveformAsync();
    }

    [RelayCommand]
    private async Task SaveAsync()
    {
        if (AssFilePath is null)
        {
            var dlg = new SaveFileDialog { Filter = "ASS Files|*.ass", DefaultExt = "ass" };
            if (dlg.ShowDialog() != true) return;
            AssFilePath = dlg.FileName;
        }

        await WriteAssAsync(AssFilePath, Lines, readHeaderFrom: AssFilePath);
        StatusText = "✅ נשמר";
    }

    [RelayCommand]
    private async Task ExportSrtAsync()
    {
        var dlg = new SaveFileDialog { Filter = "SRT Files|*.srt", DefaultExt = "srt" };
        if (dlg.ShowDialog() != true) return;

        var sb = new StringBuilder();
        for (int i = 0; i < Lines.Count; i++)
            sb.Append(Lines[i].ToSrtBlock(i + 1));
        await File.WriteAllTextAsync(dlg.FileName, sb.ToString(), Encoding.UTF8);
        StatusText = $"✅ SRT יוצא: {dlg.FileName}";
    }

    [RelayCommand]
    private void LoadVideo()
    {
        var dlg = new OpenFileDialog { Filter = "Video Files|*.mp4;*.mkv;*.avi;*.mov|All Files|*.*" };
        if (dlg.ShowDialog() == true) VideoFilePath = dlg.FileName;
    }

    [RelayCommand]
    private async Task LoadAudioAsync()
    {
        var dlg = new OpenFileDialog { Filter = "Audio Files|*.wav;*.mp3;*.flac|All Files|*.*" };
        if (dlg.ShowDialog() != true) return;
        AudioFilePath = dlg.FileName;
        await LoadWaveformAsync();
    }

    private async Task LoadWaveformAsync()
    {
        if (AudioFilePath is null) return;
        try
        {
            // Ask the API for waveform data (NAudio fallback for local file)
            var r = await _api.GetWaveformAsync(Path.GetFileName(AudioFilePath));
            if (r != null) WaveformSamples = r.Samples.ToArray();
        }
        catch { /* waveform is decorative — silently skip */ }
    }

    // ── Editing ───────────────────────────────────────────────────────────────

    private void PushUndo()
    {
        _undoStack.Push(Lines.Select(l => (SubtitleLine)l.Clone()).ToList());
        _redoStack.Clear();
    }

    [RelayCommand]
    private void Undo()
    {
        if (_undoStack.Count == 0) return;
        _redoStack.Push(Lines.Select(l => (SubtitleLine)l.Clone()).ToList());
        RestoreLines(_undoStack.Pop());
    }

    [RelayCommand]
    private void Redo()
    {
        if (_redoStack.Count == 0) return;
        _undoStack.Push(Lines.Select(l => (SubtitleLine)l.Clone()).ToList());
        RestoreLines(_redoStack.Pop());
    }

    private void RestoreLines(List<SubtitleLine> snapshot)
    {
        Lines.Clear();
        foreach (var l in snapshot) Lines.Add(l);
    }

    /// <summary>Split the selected subtitle at the current playhead position.</summary>
    [RelayCommand]
    private void SplitAtPlayhead()
    {
        if (SelectedLine is null) return;
        if (PlayheadPosition <= SelectedLine.Start || PlayheadPosition >= SelectedLine.End) return;

        PushUndo();
        int idx = Lines.IndexOf(SelectedLine);
        var second = new SubtitleLine
        {
            Start = PlayheadPosition,
            End   = SelectedLine.End,
            Text  = SelectedLine.Text,
        };
        SelectedLine.End  = PlayheadPosition;
        SelectedLine.Text = SelectedLine.Text;
        Lines.Insert(idx + 1, second);
        StatusText = "פוצלה כתובית";
    }

    /// <summary>Merge all selected subtitles into one.</summary>
    [RelayCommand]
    private void MergeSelected()
    {
        var sel = Lines.Where(l => l.IsSelected).OrderBy(l => l.Start).ToList();
        if (sel.Count < 2) return;

        PushUndo();
        var first = sel[0];
        first.End  = sel[^1].End;
        first.Text = string.Join(" ", sel.Select(l => l.Text));
        foreach (var l in sel.Skip(1)) Lines.Remove(l);
        StatusText = "מוזגו כתוביות";
    }

    /// <summary>Shift all subtitles from the given index onward by delta.</summary>
    public void ShiftFrom(int fromIndex, TimeSpan delta)
    {
        PushUndo();
        for (int i = fromIndex; i < Lines.Count; i++)
        {
            Lines[i].Start += delta;
            Lines[i].End   += delta;
        }
        StatusText = $"הוזזו {Lines.Count - fromIndex} כתוביות ב-{delta.TotalSeconds:+0.##;-0.##}s";
    }

    [RelayCommand]
    private void FindReplace()
    {
        if (string.IsNullOrEmpty(SearchQuery)) return;
        PushUndo();
        int count = 0;
        foreach (var l in Lines)
        {
            if (l.Text.Contains(SearchQuery, StringComparison.OrdinalIgnoreCase))
            {
                l.Text = l.Text.Replace(SearchQuery, ReplaceWith, StringComparison.OrdinalIgnoreCase);
                count++;
            }
        }
        StatusText = $"הוחלפו {count} מופעים";
    }

    // ── Render ────────────────────────────────────────────────────────────────

    [RelayCommand]
    private async Task RenderAsync()
    {
        if (VideoFilePath is null || AudioFilePath is null || AssFilePath is null)
        {
            StatusText = "❌ חסרים קבצים (וידאו, אודיו, ASS)";
            return;
        }

        IsRendering = true;
        StatusText  = "⏳ מרנדר...";

        // Save current edits to a temp file
        var tmpAss = Path.Combine(Path.GetTempPath(), $"ks_{Guid.NewGuid():N}.ass");
        await WriteAssAsync(tmpAss, Lines, readHeaderFrom: AssFilePath);

        try
        {
            var req = new RenderRequest(
                VideoFilePath, AudioFilePath, tmpAss,
                FontSize: FontSize, ColorHex: ColorHex,
                Position: Position, Force: true);

            var result = await _api.RenderAsync(req);
            StatusText = result != null ? $"✅ {result}" : "❌ רנדור נכשל";
        }
        catch (Exception ex) { StatusText = $"❌ {ex.Message}"; }
        finally
        {
            IsRendering = false;
            try { File.Delete(tmpAss); } catch { }
        }
    }

    // ── Playhead sync (called by VideoPlayerControl) ──────────────────────────

    public void OnPlayheadMoved(TimeSpan position)
    {
        PlayheadPosition = position;
        foreach (var l in Lines)
            l.IsActive = position >= l.Start && position < l.End;
        SelectedLine = Lines.FirstOrDefault(l => l.IsActive) ?? SelectedLine;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static async Task WriteAssAsync(
        string path, IEnumerable<SubtitleLine> lines, string? readHeaderFrom)
    {
        var sb = new StringBuilder();

        if (readHeaderFrom != null && File.Exists(readHeaderFrom))
        {
            foreach (var hline in File.ReadLines(readHeaderFrom, Encoding.UTF8))
            {
                if (hline.StartsWith("Dialogue:")) break;
                sb.AppendLine(hline);
            }
        }
        else
        {
            sb.AppendLine("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080");
            sb.AppendLine("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding");
            sb.AppendLine("Style: Karaoke,Arial,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1");
            sb.AppendLine("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text");
        }

        foreach (var l in lines)
            sb.AppendLine(l.ToAssLine());

        await File.WriteAllTextAsync(path, sb.ToString(), new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
    }
}
