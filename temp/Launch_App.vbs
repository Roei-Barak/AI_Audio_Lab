' Karaoke Studio Pro - No Console Launcher
' Double-click this file to launch the app without a terminal window

Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

strScriptPath = objFSO.GetParentFolderName(WScript.ScriptFullName)
objShell.CurrentDirectory = strScriptPath

' Run Python with hidden window
Set objProcess = objShell.Exec("python web_ui.py")
