using CommunityToolkit.Mvvm.ComponentModel;

namespace KaraokeStudio.WPF.Models;

/// <summary>
/// A single subtitle entry.  Observable so the Timeline and DataGrid update live.
/// </summary>
public partial class SubtitleLine : ObservableObject, ICloneable
{
    [ObservableProperty] private TimeSpan _start;
    [ObservableProperty] private TimeSpan _end;
    [ObservableProperty] private string   _text    = string.Empty;
    [ObservableProperty] private string   _altText = string.Empty;   // parody mode
    [ObservableProperty] private bool     _isSelected;
    [ObservableProperty] private bool     _isActive;  // currently playing

    public double DurationSeconds => (End - Start).TotalSeconds;

    // ── ASS helpers ───────────────────────────────────────────────────────────

    public static SubtitleLine FromAssLine(string dialogue)
    {
        // Dialogue: 0,H:MM:SS.CC,H:MM:SS.CC,Style,,0,0,0,,Text
        var parts = dialogue.Split(',', 10);
        if (parts.Length < 10) return new SubtitleLine();
        return new SubtitleLine
        {
            Start = ParseAssTime(parts[1].Trim()),
            End   = ParseAssTime(parts[2].Trim()),
            Text  = parts[9].Trim(),
        };
    }

    public string ToAssLine()
        => $"Dialogue: 0,{FormatAssTime(Start)},{FormatAssTime(End)},Karaoke,,0,0,0,,{Text}";

    public string ToSrtBlock(int index)
        => $"{index}\n{FormatSrtTime(Start)} --> {FormatSrtTime(End)}\n{Text}\n";

    // ── Static parse / format ─────────────────────────────────────────────────

    public static TimeSpan ParseAssTime(string s)
    {
        // H:MM:SS.CC
        try
        {
            var parts = s.Split(':', '.');
            int h  = int.Parse(parts[0]);
            int m  = int.Parse(parts[1]);
            int sc = int.Parse(parts[2]);
            int cs = parts.Length > 3 ? int.Parse(parts[3]) : 0;
            return new TimeSpan(0, h, m, sc, cs * 10);
        }
        catch { return TimeSpan.Zero; }
    }

    public static string FormatAssTime(TimeSpan t)
    {
        int h  = (int)t.TotalHours;
        int m  = t.Minutes;
        int s  = t.Seconds;
        int cs = t.Milliseconds / 10;
        return $"{h}:{m:D2}:{s:D2}.{cs:D2}";
    }

    public static string FormatSrtTime(TimeSpan t)
        => $"{(int)t.TotalHours:D2}:{t.Minutes:D2}:{t.Seconds:D2},{t.Milliseconds:D3}";

    public object Clone() => new SubtitleLine
    {
        Start   = Start,
        End     = End,
        Text    = Text,
        AltText = AltText,
    };
}
