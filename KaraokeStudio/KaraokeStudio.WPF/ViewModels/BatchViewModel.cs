using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KaraokeStudio.WPF.Models;
using KaraokeStudio.WPF.Services;

namespace KaraokeStudio.WPF.ViewModels;

public partial class BatchViewModel : ObservableObject
{
    private readonly ApiService _api;

    [ObservableProperty] private string _lang         = "he";
    [ObservableProperty] private bool   _save4Stems;
    [ObservableProperty] private bool   _useBidi;
    [ObservableProperty] private bool   _forceReprocess;
    [ObservableProperty] private bool   _isRunning;
    [ObservableProperty] private string _newQuery = string.Empty;

    public ObservableCollection<SongJob> Jobs { get; } = [];

    private CancellationTokenSource? _cts;

    public BatchViewModel(ApiService api) => _api = api;

    [RelayCommand]
    private void AddJob()
    {
        if (string.IsNullOrWhiteSpace(NewQuery)) return;
        Jobs.Add(new SongJob { Query = NewQuery.Trim() });
        NewQuery = string.Empty;
    }

    [RelayCommand]
    private void RemoveJob(SongJob job) => Jobs.Remove(job);

    [RelayCommand]
    private void ClearAll() { if (!IsRunning) Jobs.Clear(); }

    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task StartAllAsync()
    {
        if (Jobs.Count == 0) return;
        _cts      = new CancellationTokenSource();
        IsRunning = true;

        foreach (var job in Jobs)
        {
            if (_cts.IsCancellationRequested) break;
            if (job.Status == JobStatus.Done) continue;

            job.Status     = JobStatus.Running;
            job.StatusText = "מעבד...";
            job.Progress   = 0;

            var started = DateTime.UtcNow;

            try
            {
                var req = new PipelineRequest(
                    job.Query, Lang,
                    new List<string> { "ass", "srt" },
                    Save4Stems, UseBidi, ForceReprocess);

                await foreach (var ev in _api.PipelineStreamAsync(req, _cts.Token))
                {
                    App.Dispatch(() =>
                    {
                        job.Elapsed = DateTime.UtcNow - started;
                        switch (ev)
                        {
                            case SseProgressEvent p:
                                job.Progress    = p.Total > 0 ? (int)(100.0 * p.Idx / p.Total) : 0;
                                job.StatusText  = p.Text.Length > 60 ? p.Text[..60] + "…" : p.Text;
                                break;
                            case SseDoneEvent d:
                                job.Progress    = 100;
                                job.Status      = d.Success ? JobStatus.Done : JobStatus.Failed;
                                job.StatusText  = d.Success ? "הסתיים ✅" : "נכשל ❌";
                                job.OutputPath  = d.Relative;
                                break;
                        }
                    });
                }
            }
            catch (OperationCanceledException)
            {
                job.Status     = JobStatus.Failed;
                job.StatusText = "בוטל";
            }
            catch (Exception ex)
            {
                job.Status     = JobStatus.Failed;
                job.StatusText = ex.Message;
            }
        }

        IsRunning = false;
        _cts?.Dispose();
    }

    [RelayCommand]
    private void Cancel() => _cts?.Cancel();

    private bool CanStart() => !IsRunning && Jobs.Count > 0;
}
