; KaraokeStudio Standalone Installer
; Bundles the WPF client + Python embedded + backend.
; No login screen (KARAOKE_AUTH_MODE=none).
;
; Prerequisites before running ISCC:
;   1. dotnet publish → WPF EXE in BuildDir
;   2. npm run build  → web/dist/
;   3. Download Python 3.11 embeddable package to EmbedDir
;   4. pip install -r requirements.txt --target EmbedDir\Lib\site-packages
;      (or run prepare-standalone.ps1)

#define AppName    "KaraokeStudio Standalone"
#define AppVersion "1.0.0"
#define AppExeName "KaraokeStudio.exe"
#define BuildDir   "..\KaraokeStudio\KaraokeStudio.WPF\bin\Release\net8.0-windows\win-x64\publish"
#define EmbedDir   "..\standalone-python"   ; Python 3.11 embedded + packages
#define BackendDir ".."                      ; repo root (api/ + web/dist/)

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\KaraokeStudio
DefaultGroupName=KaraokeStudio
OutputDir=dist
OutputBaseFilename=KaraokeStudio-Standalone-Setup-{#AppVersion}
PrivilegesRequired=lowest
Compression=lzma2/ultra64
SolidCompression=yes
MinVersion=10.0.17763
ArchitecturesInstallIn64BitMode=x64

[Tasks]
Name: desktopicon; Description: "צור קיצור דרך בשולחן עבודה"

[Files]
; WPF client
Source: "{#BuildDir}\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildDir}\*";             DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.pdb"

; Python embedded distribution + installed packages
Source: "{#EmbedDir}\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs

; Backend source (api/)
Source: "{#BackendDir}\api\*"; DestDir: "{app}\api"; Flags: ignoreversion recursesubdirs createallsubdirs

; Web SPA (served by FastAPI)
Source: "{#BackendDir}\web\dist\*"; DestDir: "{app}\web\dist"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: WebDistExists

; Requirements file (for reference / repair)
Source: "{#BackendDir}\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}";       Filename: "{app}\{#AppExeName}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[INI]
; Write config.json so the app starts without login
Filename: "{userappdata}\KaraokeStudio\config.json"; Section: ""; Key: ""; String: ""

[Code]
function WebDistExists: Boolean;
begin
  Result := DirExists(ExpandConstant('{src}\..\web\dist'));
end;

procedure WriteStandaloneConfig();
var
  ConfigDir, ConfigFile, Json: String;
begin
  ConfigDir  := ExpandConstant('{userappdata}\KaraokeStudio');
  ConfigFile := ConfigDir + '\config.json';
  if not DirExists(ConfigDir) then
    CreateDir(ConfigDir);
  // Only write if file doesn't already exist (don't overwrite user config on upgrade)
  if not FileExists(ConfigFile) then
  begin
    Json := '{' + #13#10 +
            '  "apiBaseUrl": "http://127.0.0.1:8000",' + #13#10 +
            '  "authMode": "none",' + #13#10 +
            '  "autoStartBackend": true,' + #13#10 +
            '  "theme": "dark"' + #13#10 +
            '}';
    SaveStringToFile(ConfigFile, Json, False);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    WriteStandaloneConfig();
end;

[Run]
Filename: "{app}\{#AppExeName}"; Description: "הפעל את {#AppName}"; Flags: nowait postinstall skipifsilent
