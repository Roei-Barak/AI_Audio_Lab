using System.Diagnostics;
using System.IO;
using System.Net.Http;

namespace KaraokeStudio.WPF.Services;

/// <summary>
/// Manages the lifecycle of the Python FastAPI backend process.
/// Finds venv/python, launches api/server.py, waits for /health, and
/// kills the process on application exit.
/// </summary>
public class BackendProcess : IDisposable
{
    private Process? _process;
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(2) };

    public const string ApiBaseUrl = "http://127.0.0.1:8000";
    public bool IsRunning => _process is { HasExited: false };

    public event Action<string>? LogReceived;

    /// <summary>
    /// Locate the Python interpreter inside the project's venv, then start
    /// api/server.py.  Waits up to 30 s for the /health endpoint to respond.
    /// </summary>
    public async Task StartAsync(string projectRoot, CancellationToken ct = default)
    {
        if (IsRunning) return;

        string python = FindPython(projectRoot);
        string script  = Path.Combine(projectRoot, "api", "server.py");

        if (!File.Exists(script))
            throw new FileNotFoundException($"api/server.py לא נמצא ב: {projectRoot}");

        var psi = new ProcessStartInfo
        {
            FileName  = python,
            Arguments = $"\"{script}\" --host 127.0.0.1 --port 8000",
            WorkingDirectory      = projectRoot,
            UseShellExecute       = false,
            CreateNoWindow        = true,
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
        };

        _process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        _process.OutputDataReceived += (_, e) => { if (e.Data != null) LogReceived?.Invoke(e.Data); };
        _process.ErrorDataReceived  += (_, e) => { if (e.Data != null) LogReceived?.Invoke(e.Data); };
        _process.Exited += (_, _) => LogReceived?.Invoke("⚠️ Python backend יצא.");

        _process.Start();
        _process.BeginOutputReadLine();
        _process.BeginErrorReadLine();

        LogReceived?.Invoke("⏳ ממתין ל-Python backend...");
        await WaitForHealthAsync(ct);
        LogReceived?.Invoke("✅ Python backend מוכן.");
    }

    public void Stop()
    {
        if (_process is { HasExited: false })
        {
            try { _process.Kill(entireProcessTree: true); }
            catch { /* ignore */ }
        }
    }

    private async Task WaitForHealthAsync(CancellationToken ct)
    {
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (DateTime.UtcNow < deadline && !ct.IsCancellationRequested)
        {
            try
            {
                var r = await _http.GetAsync($"{ApiBaseUrl}/health", ct);
                if (r.IsSuccessStatusCode) return;
            }
            catch { /* not ready yet */ }
            await Task.Delay(500, ct);
        }
        throw new TimeoutException("Python backend לא השיב תוך 30 שניות.");
    }

    private static string FindPython(string projectRoot)
    {
        // 1. venv inside project root
        foreach (var rel in new[] { @"venv\Scripts\python.exe", @".venv\Scripts\python.exe" })
        {
            var p = Path.Combine(projectRoot, rel);
            if (File.Exists(p)) return p;
        }
        // 2. system python
        foreach (var name in new[] { "python", "python3", "py" })
        {
            try
            {
                var which = Process.Start(new ProcessStartInfo("where", name)
                    { RedirectStandardOutput = true, UseShellExecute = false })!;
                var line = which.StandardOutput.ReadLine();
                if (!string.IsNullOrEmpty(line) && File.Exists(line)) return line;
            }
            catch { /* not found */ }
        }
        return "python"; // hope for the best
    }

    public void Dispose()
    {
        Stop();
        _process?.Dispose();
        _http.Dispose();
    }
}
