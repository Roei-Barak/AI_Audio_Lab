; KaraokeStudio Client Installer — connects to remote server
#define AppName    "KaraokeStudio"
#define AppVersion "1.0.0"
#define AppExeName "KaraokeStudio.exe"
#define BuildDir   "..\KaraokeStudio\KaraokeStudio.WPF\bin\Release\net8.0-windows\win-x64\publish"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=dist
OutputBaseFilename=KaraokeStudio-Client-Setup-{#AppVersion}
PrivilegesRequired=lowest
Compression=lzma2/ultra64
SolidCompression=yes
MinVersion=10.0.17763
ArchitecturesInstallIn64BitMode=x64

[Tasks]
Name: desktopicon; Description: "צור קיצור דרך בשולחן עבודה"

[Files]
Source: "{#BuildDir}\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.pdb"

[Icons]
Name: "{group}\{#AppName}";       Filename: "{app}\{#AppExeName}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "הפעל את {#AppName}"; Flags: nowait postinstall skipifsilent
