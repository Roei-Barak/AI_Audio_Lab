using System.Windows.Controls;
using KaraokeStudio.WPF.ViewModels;

namespace KaraokeStudio.WPF.Views.Pages;

public partial class SubtitleEditorPage : UserControl
{
    private SubtitleEditorViewModel? _vm;

    public SubtitleEditorPage()
    {
        InitializeComponent();
        DataContextChanged += (_, _) =>
        {
            _vm = DataContext as SubtitleEditorViewModel;

            // Wire timeline seek → seek video player
            Timeline.SeekRequested += t =>
            {
                Player.Seek(t);
                _vm?.OnPlayheadMoved(t);
            };

            // Wire video player position → timeline + VM
            Player.PositionChanged += t =>
            {
                Timeline.PlayheadPosition = t;
                _vm?.OnPlayheadMoved(t);
            };

            // Wire subtitle selection in grid → load audio (for waveform)
            if (_vm != null)
                _vm.PropertyChanged += (s, e) =>
                {
                    if (e.PropertyName == nameof(_vm.VideoFilePath) && _vm.VideoFilePath != null)
                        Player.Open(_vm.VideoFilePath);
                    if (e.PropertyName == nameof(_vm.WaveformSamples))
                        Timeline.WaveformSamples = _vm.WaveformSamples;
                };
        };
    }
}
