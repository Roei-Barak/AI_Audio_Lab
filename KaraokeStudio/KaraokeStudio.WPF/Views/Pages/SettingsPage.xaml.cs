using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KaraokeStudio.WPF.Services;

namespace KaraokeStudio.WPF.Views.Pages;

// ── Simple display model for users DataGrid ────────────────────────────────

public record UserInfo(int Id, string Username, string Role, DateTime CreatedAt);

// ── Code-behind ────────────────────────────────────────────────────────────

public partial class SettingsPage : System.Windows.Controls.Page
{
    private readonly SettingsViewModel _vm;

    public SettingsPage()
    {
        InitializeComponent();
        _vm = new SettingsViewModel();
        DataContext = _vm;
        _ = _vm.LoadUsersAsync();
    }

    private void ChangePassword_Click(object sender, RoutedEventArgs e)
        => _ = _vm.ChangePasswordAsync(PwdCurrent.Password, PwdNew.Password, PwdConfirm.Password);

    private void AddUser_Click(object sender, RoutedEventArgs e)
    {
        _ = _vm.AddUserAsync(NewPwdBox.Password);
        NewPwdBox.Clear();
    }
}

// ── ViewModel ─────────────────────────────────────────────────────────────

public partial class SettingsViewModel : ObservableObject
{
    // Shared HttpClient — avoids per-call socket exhaustion.
    // URL and token are captured at construction; restart required after URL change
    // (same behaviour as ApiService, which also fixes BaseAddress at construction).
    private readonly HttpClient _authHttp;

    [ObservableProperty] private string  _serverUrl        = AppConfig.Instance.ApiBaseUrl;
    [ObservableProperty] private bool    _serverMsgVisible = false;
    [ObservableProperty] private string  _serverMessage    = "";

    [ObservableProperty] private bool    _pwdMsgVisible    = false;
    [ObservableProperty] private string  _pwdMessage       = "";
    [ObservableProperty] private bool    _pwdErrVisible    = false;
    [ObservableProperty] private string  _pwdError         = "";

    [ObservableProperty] private System.Collections.ObjectModel.ObservableCollection<UserInfo> _users = new();
    [ObservableProperty] private UserInfo? _selectedUser;
    [ObservableProperty] private string  _newUsername      = "";
    [ObservableProperty] private string  _newRole          = "user";
    [ObservableProperty] private bool    _userMsgVisible   = false;
    [ObservableProperty] private string  _userMessage      = "";
    [ObservableProperty] private bool    _userErrVisible   = false;
    [ObservableProperty] private string  _userError        = "";

    // These reflect AuthMode which is session-stable (set at startup, unchanged until restart).
    public bool ShowServerSection => AppConfig.Instance.AuthMode != "none";
    public bool ShowLogout        => AppConfig.Instance.AuthMode != "none";
    public bool IsAdmin           => AppConfig.Instance.AuthMode == "none" ||
                                     GetRoleFromToken() == "admin";

    public SettingsViewModel()
    {
        _authHttp = new HttpClient
        {
            BaseAddress = new Uri(AppConfig.Instance.ApiBaseUrl.TrimEnd('/') + "/"),
            Timeout     = TimeSpan.FromSeconds(30),
        };
        var token = AppConfig.Instance.AuthToken;
        if (token is not null)
            _authHttp.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", token);
    }

    // ── Save server URL ───────────────────────────────────────────────────

    [RelayCommand]
    private void SaveServer()
    {
        AppConfig.Instance.ApiBaseUrl = ServerUrl.TrimEnd('/');
        AppConfig.Instance.Save();
        ShowServerMsg("כתובת השרת נשמרה");
    }

    // ── Change password (called from code-behind) ─────────────────────────

    public async Task ChangePasswordAsync(string currentPwd, string newPwd, string confirmPwd)
    {
        PwdMsgVisible = false;
        PwdErrVisible = false;
        if (string.IsNullOrEmpty(currentPwd) || string.IsNullOrEmpty(newPwd))
        {
            ShowPwdErr("יש למלא את כל שדות הסיסמה"); return;
        }
        if (newPwd != confirmPwd)
        {
            ShowPwdErr("הסיסמאות החדשות אינן תואמות"); return;
        }

        try
        {
            var body = JsonSerializer.Serialize(new { current_password = currentPwd, new_password = newPwd });
            var resp = await _authHttp.PostAsync("/api/auth/change-password",
                new StringContent(body, Encoding.UTF8, "application/json"));
            if (!resp.IsSuccessStatusCode)
            {
                var err = await resp.Content.ReadAsStringAsync();
                ShowPwdErr(ApiHelpers.ParseDetail(err)); return;
            }
            ShowPwdMsg("הסיסמה שונתה בהצלחה");
        }
        catch (Exception ex) { ShowPwdErr($"שגיאת חיבור: {ex.Message}"); }
    }

    // ── Load users ────────────────────────────────────────────────────────

    public async Task LoadUsersAsync()
    {
        if (!IsAdmin) return;
        try
        {
            var list = await _authHttp.GetFromJsonAsync<List<ApiUser>>("/api/auth/users",
                new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower });
            if (list is null) return;
            App.Dispatch(() =>
            {
                Users.Clear();
                foreach (var u in list)
                    Users.Add(new UserInfo(u.Id, u.Username, u.Role,
                        DateTime.TryParse(u.CreatedAt, out var dt) ? dt : DateTime.MinValue));
            });
        }
        catch { /* silently ignore if server unreachable */ }
    }

    // ── Add user (called from code-behind) ────────────────────────────────

    public async Task AddUserAsync(string password)
    {
        UserMsgVisible = false;
        UserErrVisible = false;
        if (string.IsNullOrWhiteSpace(NewUsername) || string.IsNullOrEmpty(password))
        {
            ShowUserErr("יש למלא שם משתמש וסיסמה"); return;
        }
        try
        {
            var body = JsonSerializer.Serialize(new { username = NewUsername, password, role = NewRole });
            var resp = await _authHttp.PostAsync("/api/auth/users",
                new StringContent(body, Encoding.UTF8, "application/json"));
            if (!resp.IsSuccessStatusCode)
            {
                var err = await resp.Content.ReadAsStringAsync();
                ShowUserErr(ApiHelpers.ParseDetail(err)); return;
            }
            NewUsername = "";
            ShowUserMsg("המשתמש נוצר בהצלחה");
            await LoadUsersAsync();
        }
        catch (Exception ex) { ShowUserErr($"שגיאת חיבור: {ex.Message}"); }
    }

    // ── Delete user ───────────────────────────────────────────────────────

    [RelayCommand]
    private async Task DeleteUser()
    {
        if (SelectedUser is null) { ShowUserErr("בחר משתמש למחיקה"); return; }
        if (SelectedUser.Username == AppConfig.Instance.Username)
        {
            ShowUserErr("לא ניתן למחוק את עצמך"); return;
        }
        try
        {
            var resp = await _authHttp.DeleteAsync($"/api/auth/users/{SelectedUser.Id}");
            if (!resp.IsSuccessStatusCode)
            {
                var err = await resp.Content.ReadAsStringAsync();
                ShowUserErr(ApiHelpers.ParseDetail(err)); return;
            }
            ShowUserMsg("המשתמש נמחק");
            await LoadUsersAsync();
        }
        catch (Exception ex) { ShowUserErr($"שגיאת חיבור: {ex.Message}"); }
    }

    // ── Logout ────────────────────────────────────────────────────────────

    [RelayCommand]
    private void Logout()
    {
        AppConfig.Instance.ClearAuth();
        App.ShowLoginWindow();

        foreach (Window w in System.Windows.Application.Current.Windows)
        {
            if (w is MainWindow) { w.Close(); break; }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static string? GetRoleFromToken()
    {
        var token = AppConfig.Instance.AuthToken;
        if (token is null) return null;
        var parts = token.Split('.');
        if (parts.Length != 3) return null;
        try
        {
            var payload = parts[1];
            // JWT uses Base64Url (no padding); add padding before decoding
            payload = payload.PadRight(payload.Length + (4 - payload.Length % 4) % 4, '=');
            var json = Encoding.UTF8.GetString(Convert.FromBase64String(payload));
            var el = JsonSerializer.Deserialize<JsonElement>(json);
            return el.TryGetProperty("role", out var r) ? r.GetString() : null;
        }
        catch { return null; }
    }

    private void ShowServerMsg(string msg) { ServerMessage = msg; ServerMsgVisible = true; }
    private void ShowPwdMsg(string msg)    { PwdMessage = msg;    PwdMsgVisible    = true; }
    private void ShowPwdErr(string msg)    { PwdError   = msg;    PwdErrVisible    = true; }
    private void ShowUserMsg(string msg)   { UserMessage = msg;   UserMsgVisible   = true; }
    private void ShowUserErr(string msg)   { UserError   = msg;   UserErrVisible   = true; }

    private record ApiUser(int Id, string Username, string Role, string CreatedAt);
}
