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

    public bool ShowServerSection => AppConfig.Instance.AuthMode != "none";
    public bool ShowLogout        => AppConfig.Instance.AuthMode != "none";
    public bool IsAdmin           => AppConfig.Instance.AuthMode == "none" ||
                                     GetRoleFromToken() == "admin";

    // ── Save server URL ───────────────────────────────────────────────────

    [RelayCommand]
    private void SaveServer()
    {
        AppConfig.Instance.ApiBaseUrl = ServerUrl.TrimEnd('/');
        AppConfig.Instance.Save();
        ShowServerMsg("כתובת השרת נשמרה");
    }

    // ── Change password (called from code-behind) ─────────────────────────

    public async Task ChangePasswordAsync(string oldPwd, string newPwd, string confirmPwd)
    {
        PwdMsgVisible = false;
        PwdErrVisible = false;
        if (string.IsNullOrEmpty(oldPwd) || string.IsNullOrEmpty(newPwd))
        {
            ShowPwdErr("יש למלא את כל שדות הסיסמה"); return;
        }
        if (newPwd != confirmPwd)
        {
            ShowPwdErr("הסיסמאות החדשות אינן תואמות"); return;
        }

        try
        {
            using var http = CreateHttpClient();
            var body = JsonSerializer.Serialize(new { old_password = oldPwd, new_password = newPwd });
            var resp = await http.PostAsync("/api/auth/change-password",
                new StringContent(body, Encoding.UTF8, "application/json"));
            if (!resp.IsSuccessStatusCode)
            {
                var err = await resp.Content.ReadAsStringAsync();
                ShowPwdErr(ParseDetail(err)); return;
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
            using var http = CreateHttpClient();
            var list = await http.GetFromJsonAsync<List<ApiUser>>("/api/auth/users",
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
            using var http = CreateHttpClient();
            var body = JsonSerializer.Serialize(new { username = NewUsername, password, role = NewRole });
            var resp = await http.PostAsync("/api/auth/users",
                new StringContent(body, Encoding.UTF8, "application/json"));
            if (!resp.IsSuccessStatusCode)
            {
                var err = await resp.Content.ReadAsStringAsync();
                ShowUserErr(ParseDetail(err)); return;
            }
            NewUsername = "";
            ShowUserMsg($"המשתמש נוצר בהצלחה");
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
            using var http = CreateHttpClient();
            var resp = await http.DeleteAsync($"/api/auth/users/{SelectedUser.Id}");
            if (!resp.IsSuccessStatusCode)
            {
                var err = await resp.Content.ReadAsStringAsync();
                ShowUserErr(ParseDetail(err)); return;
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

        var frame = new System.Windows.Controls.Frame
        {
            NavigationUIVisibility = System.Windows.Navigation.NavigationUIVisibility.Hidden
        };
        frame.Navigate(new LoginPage());
        var win = new Window
        {
            Title = "KaraokeStudio – כניסה",
            Content = frame,
            Width = 440, Height = 520,
            ResizeMode = ResizeMode.NoResize,
            WindowStartupLocation = WindowStartupLocation.CenterScreen,
            Background = System.Windows.Media.Brushes.Transparent,
            FlowDirection = FlowDirection.RightToLeft
        };
        win.Show();

        foreach (Window w in System.Windows.Application.Current.Windows)
        {
            if (w is MainWindow) { w.Close(); break; }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static HttpClient CreateHttpClient()
    {
        var http = new HttpClient
        {
            BaseAddress = new Uri(AppConfig.Instance.ApiBaseUrl.TrimEnd('/')),
            Timeout = TimeSpan.FromSeconds(30)
        };
        var token = AppConfig.Instance.AuthToken;
        if (token is not null)
            http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        return http;
    }

    private static string? GetRoleFromToken()
    {
        var token = AppConfig.Instance.AuthToken;
        if (token is null) return null;
        var parts = token.Split('.');
        if (parts.Length != 3) return null;
        try
        {
            var payload = parts[1];
            payload = payload.PadRight(payload.Length + (4 - payload.Length % 4) % 4, '=');
            var json = Encoding.UTF8.GetString(Convert.FromBase64String(payload));
            var el = JsonSerializer.Deserialize<JsonElement>(json);
            return el.TryGetProperty("role", out var r) ? r.GetString() : null;
        }
        catch { return null; }
    }

    private static string ParseDetail(string json)
    {
        try
        {
            var el = JsonSerializer.Deserialize<JsonElement>(json);
            return el.TryGetProperty("detail", out var d) ? d.GetString() ?? json : json;
        }
        catch { return json; }
    }

    private void ShowServerMsg(string msg) { ServerMessage = msg; ServerMsgVisible = true; }
    private void ShowPwdMsg(string msg)    { PwdMessage = msg;    PwdMsgVisible    = true; }
    private void ShowPwdErr(string msg)    { PwdError   = msg;    PwdErrVisible    = true; }
    private void ShowUserMsg(string msg)   { UserMessage = msg;   UserMsgVisible   = true; }
    private void ShowUserErr(string msg)   { UserError   = msg;   UserErrVisible   = true; }

    private record ApiUser(int Id, string Username, string Role, string CreatedAt);
}
