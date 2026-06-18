using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KaraokeStudio.WPF.Services;

namespace KaraokeStudio.WPF.ViewModels;

public partial class PipelineViewModel : ObservableObject
{
    private readonly ApiService _api;

    [ObservableProperty] private string  _url          = string.Empty;
    [ObservableProperty] private string  _lang         = "he";
    [ObservableProperty] private bool    _save4Stems;
    [ObservableProperty] private bool    _useBidi;
    [ObservableProperty] private bool    _forceReprocess;
    [ObservableProperty] private int     _progress;
    [ObservableProperty] private string  _progressText = string.Empty;
    [ObservableProperty] private bool    _isRunning;
    [ObservableProperty] private string? _outputVideoPath;

    public ObservableCollection<string> Logs { get; } = [];

    private CancellationTokenSource? _cts;

    public PipelineViewModel(ApiService api) => _api = api;

    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task StartAsync()
    {
        if (string.IsNullOrWhiteSpace(Url)) return;

        _cts        = new CancellationTokenSource();
        IsRunning   = true;
        Progress    = 0;
        OutputVideoPath = null;
        Logs.Clear();

        try
        {
            var req = new PipelineRequest(
                Url, Lang,
                new List<string> { "ass", "srt" },
                Save4Stems, UseBidi, ForceReprocess);

            await foreach (var ev in _api.PipelineStreamAsync(req, _cts.Token))
            {
                App.Dispatch(() =>
                {
                    switch (ev)
                    {
                        case SseProgressEvent p:
                            Progress     = p.Total > 0 ? (int)(100.0 * p.Idx / p.Total) : 0;
                            ProgressText = $"[{p.Idx}/{p.Total}] {p.Text}";
                            Logs.Add(ProgressText);
                            break;
                        case SseLogEvent l:
                            Logs.Add(l.Text);
                            break;
                        case SseDoneEvent d:
                            Progress = 100;
                            if (d.Success && d.Relative != null)
                                OutputVideoPath = _api.FileUrl(d.Relative);
                            Logs.Add(d.Success ? "✅ הסתיים!" : "❌ נכשל");
                            break;
                    }
                });
            }
        }
        catch (OperationCanceledException) { Logs.Add("⚠️ בוטל"); }
        catch (Exception ex)              { Logs.Add($"❌ {ex.Message}"); }
        finally
        {
            IsRunning = false;
            _cts?.Dispose();
        }
    }

    [RelayCommand]
    private void Cancel() => _cts?.Cancel();

    private bool CanStart() => !IsRunning;
}
