using CommunityToolkit.Mvvm.ComponentModel;

namespace KaraokeStudio.WPF.Models;

public enum JobStatus { Waiting, Running, Done, Failed }

/// <summary>Represents one item in the batch queue.</summary>
public partial class SongJob : ObservableObject
{
    public Guid Id { get; } = Guid.NewGuid();

    [ObservableProperty] private string    _query      = string.Empty;
    [ObservableProperty] private JobStatus _status     = JobStatus.Waiting;
    [ObservableProperty] private int       _progress;           // 0–100
    [ObservableProperty] private string    _statusText = "ממתין";
    [ObservableProperty] private string?   _outputPath;
    [ObservableProperty] private TimeSpan  _elapsed;

    public string StatusEmoji => Status switch
    {
        JobStatus.Waiting => "⏳",
        JobStatus.Running => "🔄",
        JobStatus.Done    => "✅",
        JobStatus.Failed  => "❌",
        _                 => "❓",
    };
}
