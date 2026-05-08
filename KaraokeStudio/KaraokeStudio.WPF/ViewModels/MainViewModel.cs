using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KaraokeStudio.WPF.Services;

namespace KaraokeStudio.WPF.ViewModels;

public partial class MainViewModel : ObservableObject
{
    public ApiService     Api     { get; }
    public BackendProcess Backend { get; }

    public PipelineViewModel       Pipeline { get; }
    public SubtitleEditorViewModel Editor   { get; }
    public ParodyViewModel         Parody   { get; }
    public BatchViewModel          Batch    { get; }

    [ObservableProperty] private string  _backendStatus = "⏳ מפעיל...";
    [ObservableProperty] private bool    _backendReady;
    [ObservableProperty] private int     _selectedTab;
    [ObservableProperty] private string  _backendLog = string.Empty;

    public MainViewModel()
    {
        Backend  = new BackendProcess();
        Api      = new ApiService();
        Pipeline = new PipelineViewModel(Api);
        Editor   = new SubtitleEditorViewModel(Api);
        Parody   = new ParodyViewModel(Api);
        Batch    = new BatchViewModel(Api);

        Backend.LogReceived += line =>
        {
            App.Dispatch(() => BackendLog += line + "\n");
        };
    }

    public async Task InitAsync(string projectRoot)
    {
        try
        {
            BackendStatus = "⏳ מפעיל Python backend...";
            await Backend.StartAsync(projectRoot);
            BackendStatus = "🟢 מוכן";
            BackendReady  = true;
        }
        catch (Exception ex)
        {
            BackendStatus = $"❌ {ex.Message}";
        }
    }

    [RelayCommand]
    private void NavigateTo(int tab) => SelectedTab = tab;
}
