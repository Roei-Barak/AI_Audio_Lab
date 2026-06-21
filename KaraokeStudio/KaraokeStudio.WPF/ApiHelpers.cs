using System.Text.Json;

namespace KaraokeStudio.WPF;

internal static class ApiHelpers
{
    internal static string ParseDetail(string json)
    {
        try
        {
            var el = JsonSerializer.Deserialize<JsonElement>(json);
            return el.TryGetProperty("detail", out var d) ? d.GetString() ?? json : json;
        }
        catch { return json; }
    }
}
