using System.Collections.ObjectModel;
using System.IO;
using System.Text;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KaraokeStudio.WPF.Models;
using KaraokeStudio.WPF.Services;
using Microsoft.Win32;

namespace KaraokeStudio.WPF.ViewModels;

public partial class ParodyViewModel : ObservableObject
{
    private readonly ApiService _api;

    [ObservableProperty] private string? _assFilePath;
    [ObservableProperty] private string? _videoFilePath;
    [ObservableProperty] private string? _audioFilePath;
    [ObservableProperty] private bool    _useAlternative = true;
    [ObservableProperty] private bool    _useBidi;
    [ObservableProperty] private string  _statusText = "טען קובץ ASS להתחלה";
    [ObservableProperty] private bool    _isRendering;

    public ObservableCollection<SubtitleLine> Lines { get; } = [];

    public ParodyViewModel(ApiService api) => _api = api;

    [RelayCommand]
    private void LoadAss()
    {
        var dlg = new OpenFileDialog { Filter = "ASS Files|*.ass|All Files|*.*" };
        if (dlg.ShowDialog() != true) return;

        AssFilePath = dlg.FileName;
        Lines.Clear();
        foreach (var line in File.ReadLines(dlg.FileName, Encoding.UTF8))
        {
            if (line.StartsWith("Dialogue:"))
                Lines.Add(SubtitleLine.FromAssLine(line));
        }
        StatusText = $"נטען: {Lines.Count} שורות — ערוך את עמודת 'חלופי'";
    }

    [RelayCommand]
    private void LoadVideo()
    {
        var dlg = new OpenFileDialog { Filter = "Video|*.mp4;*.mkv;*.avi|All|*.*" };
        if (dlg.ShowDialog() == true) VideoFilePath = dlg.FileName;
    }

    [RelayCommand]
    private void LoadAudio()
    {
        var dlg = new OpenFileDialog { Filter = "Audio|*.wav;*.mp3;*.flac|All|*.*" };
        if (dlg.ShowDialog() == true) AudioFilePath = dlg.FileName;
    }

    [RelayCommand]
    private async Task ExportAsync()
    {
        var dlg = new SaveFileDialog { Filter = "ASS Files|*.ass", DefaultExt = "ass" };
        if (dlg.ShowDialog() != true) return;

        await WriteParodyAssAsync(dlg.FileName);
        StatusText = $"✅ יוצא: {dlg.FileName}";
    }

    [RelayCommand]
    private async Task RenderAsync()
    {
        if (VideoFilePath is null || AudioFilePath is null)
        {
            StatusText = "❌ בחר קבצי וידאו ואודיו";
            return;
        }

        IsRendering = true;
        StatusText  = "⏳ מרנדר פרודיה...";

        var tmpAss = Path.Combine(Path.GetTempPath(), $"parody_{Guid.NewGuid():N}.ass");
        await WriteParodyAssAsync(tmpAss);

        try
        {
            var req = new RenderRequest(
                VideoFilePath, AudioFilePath, tmpAss,
                UseBidi: UseBidi, Force: true);
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

    private async Task WriteParodyAssAsync(string path)
    {
        var sb = new StringBuilder();

        if (AssFilePath != null && File.Exists(AssFilePath))
        {
            foreach (var line in File.ReadLines(AssFilePath, Encoding.UTF8))
            {
                if (line.StartsWith("Dialogue:")) break;
                sb.AppendLine(line);
            }
        }

        foreach (var l in Lines)
        {
            string text = UseAlternative && !string.IsNullOrWhiteSpace(l.AltText)
                ? l.AltText
                : l.Text;

            sb.AppendLine($"Dialogue: 0,{SubtitleLine.FormatAssTime(l.Start)}," +
                          $"{SubtitleLine.FormatAssTime(l.End)},Karaoke,,0,0,0,,{text}");
        }

        await File.WriteAllTextAsync(path, sb.ToString(),
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
    }
}
