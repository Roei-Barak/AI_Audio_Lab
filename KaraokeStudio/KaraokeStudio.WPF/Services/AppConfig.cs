using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace KaraokeStudio.WPF.Services;

/// <summary>
/// Persists app settings to %APPDATA%\KaraokeStudio\config.json.
/// The JWT token is DPAPI-encrypted (Windows user scope).
/// </summary>
public class AppConfig
{
    private static readonly string _dir  = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "KaraokeStudio");
    private static readonly string _path = Path.Combine(_dir, "config.json");

    private static readonly JsonSerializerOptions _json = new()
    {
        WriteIndented           = true,
        PropertyNamingPolicy    = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition  = JsonIgnoreCondition.WhenWritingNull,
    };

    // ── Persisted fields ──────────────────────────────────────────────────────

    public string  ApiBaseUrl       { get; set; } = BackendProcess.ApiBaseUrl;
    public string  AuthMode         { get; set; } = "required";   // 'required' | 'none'
    public string? EncryptedToken   { get; set; }
    public string? Username         { get; set; }
    public bool    AutoStartBackend { get; set; } = false;
    public string  Theme            { get; set; } = "dark";

    // ── Runtime (not persisted) ───────────────────────────────────────────────

    [JsonIgnore]
    public string? AuthToken
    {
        get
        {
            if (EncryptedToken is null) return null;
            try
            {
                var enc   = Convert.FromBase64String(EncryptedToken);
                var plain = ProtectedData.Unprotect(enc, null, DataProtectionScope.CurrentUser);
                return Encoding.UTF8.GetString(plain);
            }
            catch { return null; }
        }
        set
        {
            if (value is null) { EncryptedToken = null; return; }
            var plain = Encoding.UTF8.GetBytes(value);
            var enc   = ProtectedData.Protect(plain, null, DataProtectionScope.CurrentUser);
            EncryptedToken = Convert.ToBase64String(enc);
        }
    }

    // ── Singleton load/save ───────────────────────────────────────────────────

    private static AppConfig? _instance;
    public  static AppConfig  Instance => _instance ??= Load();

    public static AppConfig Load()
    {
        if (File.Exists(_path))
        {
            try
            {
                var cfg = JsonSerializer.Deserialize<AppConfig>(File.ReadAllText(_path), _json);
                if (cfg != null) return cfg;
            }
            catch { /* corrupt → use defaults */ }
        }
        return new AppConfig();
    }

    public void Save()
    {
        Directory.CreateDirectory(_dir);
        File.WriteAllText(_path, JsonSerializer.Serialize(this, _json));
    }

    public void ClearAuth()
    {
        AuthToken = null;
        Username  = null;
        Save();
    }
}
