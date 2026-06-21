using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Windows;
using System.Windows.Controls;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using KaraokeStudio.WPF.Services;

namespace KaraokeStudio.WPF.Views.Pages;

public partial class LoginPage : System.Windows.Controls.Page
{
    public LoginPage()
    {
        InitializeComponent();
        DataContext = new LoginViewModel(ShowError);
    }

    private void ShowError(string msg)
    {
        TxtError.Text        = msg;
        ErrBorder.Visibility = msg.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
    }
}

public partial class LoginViewModel : ObservableObject
{
    private readonly Action<string> _showError;

    [ObservableProperty] private string _serverUrl = AppConfig.Instance.ApiBaseUrl;
    [ObservableProperty] private string _username  = AppConfig.Instance.Username ?? "";
    [ObservableProperty] private string _btnText   = "כניסה";

    public LoginViewModel(Action<string> showError) => _showError = showError;

    [RelayCommand]
    private async Task Login(PasswordBox? pwd)
    {
        _showError("");
        var password = pwd?.Password ?? "";
        if (string.IsNullOrWhiteSpace(ServerUrl) ||
            string.IsNullOrWhiteSpace(Username)  ||
            string.IsNullOrEmpty(password))
        {
            _showError("יש למלא את כל השדות"); return;
        }

        BtnText = "⏳ מתחבר...";
        try
        {
            using var http = new HttpClient { BaseAddress = new Uri(ServerUrl.TrimEnd('/') + "/") };
            var body = JsonSerializer.Serialize(new { username = Username, password });
            var resp = await http.PostAsync("/api/auth/login",
                new StringContent(body, Encoding.UTF8, "application/json"));

            if (!resp.IsSuccessStatusCode)
            {
                var err = await resp.Content.ReadAsStringAsync();
                _showError($"שגיאה {(int)resp.StatusCode}: {ApiHelpers.ParseDetail(err)}"); return;
            }

            var result = await resp.Content.ReadFromJsonAsync<LoginResponse>(new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            });
            if (result?.AccessToken is null) { _showError("תגובה לא תקינה מהשרת"); return; }

            AppConfig.Instance.ApiBaseUrl = ServerUrl.TrimEnd('/');
            AppConfig.Instance.AuthToken  = result.AccessToken;
            AppConfig.Instance.Username   = Username;
            AppConfig.Instance.Save();

            var main = new MainWindow();
            main.Show();
            // Windows[0] is the login host window (shown before MainWindow).
            App.Current.Windows[0].Close();
        }
        catch (Exception ex) { _showError($"שגיאת חיבור: {ex.Message}"); }
        finally { BtnText = "כניסה"; }
    }

    private record LoginResponse(
        [property: JsonPropertyName("access_token")] string AccessToken,
        [property: JsonPropertyName("token_type")]   string TokenType);
}
