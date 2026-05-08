using System.IO;
using System.Windows;
using KaraokeStudio.WPF.ViewModels;

namespace KaraokeStudio.WPF;

public partial class MainWindow : Window
{
    private readonly MainViewModel _vm;

    public MainWindow()
    {
        InitializeComponent();
        _vm = new MainViewModel();
        DataContext = _vm;

        Loaded += MainWindow_Loaded;
    }

    private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        // Project root is 2 levels up from the WPF executable (KaraokeStudio/KaraokeStudio.WPF/bin/Debug/net8.0-windows)
        var exeDir = AppContext.BaseDirectory;
        var projectRoot = FindProjectRoot(exeDir);

        await _vm.InitAsync(projectRoot);

        if (_vm.BackendReady)
            StatusDot.Color = System.Windows.Media.Color.FromRgb(0x55, 0xFF, 0x55);
    }

    private static string FindProjectRoot(string startDir)
    {
        // Walk up looking for api/server.py
        var dir = new DirectoryInfo(startDir);
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "api", "server.py")))
                return dir.FullName;
            dir = dir.Parent;
        }
        // Fallback: assume sibling of KaraokeStudio folder
        return Path.GetFullPath(Path.Combine(startDir, "..", "..", "..", "..", ".."));
    }

    private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
    {
        try { _vm.Backend.Stop(); } catch { /* ignore */ }
    }

    private void LogToggle_Click(object sender, RoutedEventArgs e)
        => LogExpander.IsExpanded = !LogExpander.IsExpanded;
}
