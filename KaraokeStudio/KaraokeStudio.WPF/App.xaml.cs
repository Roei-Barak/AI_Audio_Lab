using System.Windows;
using System.Windows.Threading;

namespace KaraokeStudio.WPF;

public partial class App : Application
{
    public static void Dispatch(Action action) =>
        Current.Dispatcher.Invoke(action);

    protected override void OnExit(ExitEventArgs e)
    {
        // MainViewModel.Backend.Stop() is called via MainWindow.Closing
        base.OnExit(e);
    }
}
