using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Navigation;
using KaraokeStudio.WPF.Services;
using KaraokeStudio.WPF.Views.Pages;

namespace KaraokeStudio.WPF;

public partial class App : Application
{
    public static void Dispatch(Action action) =>
        Current.Dispatcher.Invoke(action);

    private void Application_Startup(object sender, StartupEventArgs e)
    {
        var cfg = AppConfig.Instance;

        // Standalone mode or already authenticated → go straight to MainWindow
        if (cfg.AuthMode == "none" || cfg.AuthToken is not null)
        {
            new MainWindow().Show();
            return;
        }

        ShowLoginWindow();
    }

    /// <summary>
    /// Creates and shows the login host window.
    /// Single definition — called from startup and from logout.
    /// </summary>
    internal static void ShowLoginWindow()
    {
        var frame = new Frame { NavigationUIVisibility = NavigationUIVisibility.Hidden };
        frame.Navigate(new LoginPage());
        var win = new Window
        {
            Title  = "KaraokeStudio – כניסה",
            Content = frame,
            Width  = 440,
            Height = 520,
            ResizeMode = ResizeMode.NoResize,
            WindowStartupLocation = WindowStartupLocation.CenterScreen,
            Background = new SolidColorBrush(Color.FromRgb(0x1E, 0x1E, 0x1E)),
            FlowDirection = FlowDirection.RightToLeft,
        };
        win.Show();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        base.OnExit(e);
    }
}
