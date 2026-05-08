using System.Globalization;
using System.Windows.Data;

namespace KaraokeStudio.WPF.Converters;

[ValueConversion(typeof(TimeSpan), typeof(string))]
public class TimeSpanConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is TimeSpan ts)
            return $"{(int)ts.TotalMinutes}:{ts.Seconds:D2}.{ts.Milliseconds / 10:D2}";
        return "0:00.00";
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is string s && TimeSpan.TryParseExact(s, @"m\:ss\.ff", culture, out var ts))
            return ts;
        return TimeSpan.Zero;
    }
}

[ValueConversion(typeof(bool), typeof(System.Windows.Visibility))]
public class BoolToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        bool invert = parameter is string p && p == "invert";
        bool val = value is bool b && b;
        if (invert) val = !val;
        return val ? System.Windows.Visibility.Visible : System.Windows.Visibility.Collapsed;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        value is System.Windows.Visibility v && v == System.Windows.Visibility.Visible;
}

[ValueConversion(typeof(bool), typeof(string))]
public class BoolToStringConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        var parts = parameter?.ToString()?.Split('|');
        if (parts?.Length == 2 && value is bool b)
            return b ? parts[0] : parts[1];
        return value?.ToString() ?? string.Empty;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotImplementedException();
}
