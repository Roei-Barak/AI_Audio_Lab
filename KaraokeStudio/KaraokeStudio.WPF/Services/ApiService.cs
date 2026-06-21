using System.IO;
using System.Net.Http;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace KaraokeStudio.WPF.Services;

// ── Request / Response models ─────────────────────────────────────────────────

public record VideoInfo(string Title, string Url, string Id, string Folder);

public record DownloadRequest(string Url, string Fmt = "wav", string? OutputDir = null);
public record DownloadResponse(string Path, string Relative);

public record SeparateRequest(string AudioPath, int Mode = 2, string? OutputDir = null, bool Force = false);
public record SeparateResponse(string Vocals, string VocalsRel, string Playback, string PlaybackRel);

public record TranscribeRequest(
    string AudioPath,
    string Lang = "he",
    List<string>? OutputFormats = null,
    string? Title = null,
    string? OutputDir = null,
    bool Force = false);

public record RenderRequest(
    string VideoPath,
    string AudioPath,
    string SubtitlesPath,
    string? OutputDir = null,
    string? OutputName = null,
    bool UseBidi = false,
    int? FontSize = null,
    string? ColorHex = null,
    string Position = "bottom",
    bool Force = false);

public record PipelineRequest(
    string Url,
    string Lang = "he",
    List<string>? OutputFormats = null,
    bool Save4Stems = false,
    bool UseBidi = false,
    bool Force = false);

public record AnalyzeRequest(string AudioPath);

// SSE event types
public record SseProgressEvent(string Type, int Idx, int Total, string Text);
public record SseLogEvent(string Type, string Text);
public record SseDoneEvent(string Type, bool Success, string? Path, string? Relative,
    Dictionary<string, string>? Files = null);

public record WaveformResponse(List<float> Samples, int SampleRate);
public record ExportRequest(string AssPath, string Format = "srt");

// ── Service ───────────────────────────────────────────────────────────────────

/// <summary>
/// Typed HTTP client wrapping all endpoints exposed by api/server.py.
/// </summary>
public class ApiService
{
    private readonly HttpClient _http;
    private readonly string _base;

    private static readonly JsonSerializerOptions _json = new()
    {
        PropertyNamingPolicy        = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition      = JsonIgnoreCondition.WhenWritingNull,
        PropertyNameCaseInsensitive = true,
    };

    public ApiService(string? baseUrl = null)
    {
        _base = (baseUrl ?? AppConfig.Instance.ApiBaseUrl).TrimEnd('/');
        _http = new HttpClient { BaseAddress = new Uri(_base + "/"), Timeout = TimeSpan.FromMinutes(30) };
        var token = AppConfig.Instance.AuthToken;
        if (token is not null)
            _http.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
    }

    // ── Health ────────────────────────────────────────────────────────────────

    public async Task<bool> IsHealthyAsync()
    {
        try
        {
            var r = await _http.GetAsync("/health");
            return r.IsSuccessStatusCode;
        }
        catch { return false; }
    }

    // ── Info ──────────────────────────────────────────────────────────────────

    public async Task<VideoInfo?> GetInfoAsync(string url)
    {
        var r = await PostJsonAsync<dynamic>("/info", new { url });
        if (r is null) return null;
        var info = r["info"];
        return new VideoInfo(
            (string)info["title"], (string)info["url"],
            (string)info["id"],    (string)info["folder"]);
    }

    // ── Download ──────────────────────────────────────────────────────────────

    public async Task<string?> DownloadAsync(string url, string fmt = "wav", string? outputDir = null)
    {
        var r = await PostJsonAsync<DownloadResponse>("/download",
            new DownloadRequest(url, fmt, outputDir));
        return r?.Relative;
    }

    // ── Separate ──────────────────────────────────────────────────────────────

    public async Task<SeparateResponse?> SeparateAsync(string audioPath, int mode = 2, bool force = false)
        => await PostJsonAsync<SeparateResponse>("/separate",
               new SeparateRequest(audioPath, mode, Force: force));

    // ── Transcribe (non-streaming) ────────────────────────────────────────────

    public async Task<Dictionary<string, string?>?> TranscribeAsync(TranscribeRequest req)
    {
        var r = await PostJsonAsync<JsonElement>("/transcribe", req);
        if (r is null) return null;
        var files = r.Value.GetProperty("relative");
        return JsonSerializer.Deserialize<Dictionary<string, string?>>(files.GetRawText());
    }

    // ── Transcribe (SSE stream) ───────────────────────────────────────────────

    public async IAsyncEnumerable<object> TranscribeStreamAsync(
        TranscribeRequest req,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var ev in StreamSseAsync("/transcribe/stream", req, ct))
            yield return ev;
    }

    // ── Render ────────────────────────────────────────────────────────────────

    public async Task<string?> RenderAsync(RenderRequest req)
    {
        var r = await PostJsonAsync<JsonElement>("/render", req);
        if (r is null) return null;
        return r.Value.GetProperty("relative").GetString();
    }

    // ── Pipeline (SSE stream) ─────────────────────────────────────────────────

    public async IAsyncEnumerable<object> PipelineStreamAsync(
        PipelineRequest req,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var ev in StreamSseAsync("/pipeline/stream", req, ct))
            yield return ev;
    }

    // ── Analyze ───────────────────────────────────────────────────────────────

    public async Task<string?> AnalyzeAsync(string audioPath)
    {
        var r = await PostJsonAsync<JsonElement>("/analyze", new AnalyzeRequest(audioPath));
        if (r is null) return null;
        return r.Value.GetProperty("result").GetString();
    }

    // ── Waveform ──────────────────────────────────────────────────────────────

    public async Task<WaveformResponse?> GetWaveformAsync(string relativeFilename)
        => await GetJsonAsync<WaveformResponse>($"/waveform/{Uri.EscapeDataString(relativeFilename)}");

    // ── Export (ASS → SRT / VTT) ──────────────────────────────────────────────

    public async Task<string?> ExportAsync(string assPath, string format = "srt")
    {
        var r = await PostJsonAsync<JsonElement>("/export", new ExportRequest(assPath, format));
        if (r is null) return null;
        return r.Value.TryGetProperty("path", out var p) ? p.GetString() : null;
    }

    // ── Thumbnail ─────────────────────────────────────────────────────────────

    public string ThumbnailUrl(string relativeFilename)
        => $"{_base}/thumbnail/{Uri.EscapeDataString(relativeFilename)}";

    // ── File serving ──────────────────────────────────────────────────────────

    public string FileUrl(string relativePath)
        => $"{_base}/files/{Uri.EscapeDataString(relativePath.Replace('\\', '/'))}";

    public async Task DownloadFileAsync(string relativePath, string localDest)
    {
        using var response = await _http.GetAsync(FileUrl(relativePath), HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();
        await using var src  = await response.Content.ReadAsStreamAsync();
        await using var dest = File.OpenWrite(localDest);
        await src.CopyToAsync(dest);
    }

    // ── SSE generic reader ────────────────────────────────────────────────────

    private async IAsyncEnumerable<object> StreamSseAsync<TReq>(
        string endpoint,
        TReq body,
        [EnumeratorCancellation] CancellationToken ct)
    {
        var content  = new StringContent(JsonSerializer.Serialize(body, _json), Encoding.UTF8, "application/json");
        using var response = await _http.PostAsync(endpoint, content, ct);
        response.EnsureSuccessStatusCode();

        using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var reader = new StreamReader(stream);

        while (!reader.EndOfStream && !ct.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(ct);
            if (string.IsNullOrEmpty(line) || !line.StartsWith("data: ")) continue;

            var json = line[6..];
            JsonElement el;
            try { el = JsonSerializer.Deserialize<JsonElement>(json); }
            catch { continue; }

            var type = el.TryGetProperty("type", out var t) ? t.GetString() : "";
            object ev = type switch
            {
                "progress" => new SseProgressEvent(
                    type,
                    el.TryGetProperty("idx",   out var i) ? i.GetInt32()    : 0,
                    el.TryGetProperty("total", out var tot) ? tot.GetInt32() : 0,
                    el.TryGetProperty("text",  out var tx)  ? tx.GetString() ?? "" : ""),
                "log"  => new SseLogEvent(type,
                    el.TryGetProperty("text", out var lt) ? lt.GetString() ?? "" : ""),
                "done" => new SseDoneEvent(
                    type,
                    el.TryGetProperty("success", out var s) && s.GetBoolean(),
                    el.TryGetProperty("path",     out var p) ? p.GetString() : null,
                    el.TryGetProperty("relative", out var rel) ? rel.GetString() : null),
                _ => (object)json,
            };
            yield return ev;
            if (type == "done") yield break;
        }
    }

    // ── HTTP helpers ──────────────────────────────────────────────────────────

    private async Task<T?> PostJsonAsync<T>(string url, object body)
    {
        var content  = new StringContent(JsonSerializer.Serialize(body, _json), Encoding.UTF8, "application/json");
        var response = await _http.PostAsync(url, content);
        if (!response.IsSuccessStatusCode) return default;
        return await response.Content.ReadFromJsonAsync<T>(_json);
    }

    private async Task<T?> GetJsonAsync<T>(string url)
    {
        var response = await _http.GetAsync(url);
        if (!response.IsSuccessStatusCode) return default;
        return await response.Content.ReadFromJsonAsync<T>(_json);
    }
}
