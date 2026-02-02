# Karaoke Studio Pro V16 - New Features Quick Guide

## 🎬 Updated Features

### 1. Enhanced Automatic Creation (⚡ יצירה אוטומטית Tab)

#### Before:
- Click button → get video (no editing option)

#### Now:
- **Step 1**: Click "התחל תהליך" 
  - YouTube URL/local file → Downloads → Separates audio → Transcribes → Creates video
  - Takes ~10-20 minutes depending on length

- **Step 2**: Review & Edit (NEW!)
  - Subtitles automatically appear in the text editor
  - Make any text corrections or adjustments
  - Adjust font size (20-150px) and color using the picker

- **Step 3**: Render (NEW!)
  - Click "📝 עדכן סגנון ורנדר מחדש"
  - Video re-renders in seconds with your edits
  - No need to re-download, re-separate, or re-transcribe!

---

### 2. Playlist Processor (🎼 רשימת השמעה - NEW TAB!)

#### What it does:
Downloads an entire YouTube playlist and transcribes all videos to TXT files

#### How to use:
1. Paste YouTube playlist URL (e.g., `https://www.youtube.com/playlist?list=...`)
2. Select language (Hebrew/English)
3. Click "עבד רשימה"
4. Wait for download + transcription to complete

#### Outputs:
- **Individual TXT Files**: One file per video
  - Named: `VideoTitle.txt`
  - Contains: `[00:00:05.00 - 00:00:10.50] Text content`
  
- **Combined TXT File**: `Combined_Transcriptions.txt`
  - Includes all videos with section separators
  - Easy to review entire playlist content at once

#### Example Combined File:
```
============================================================
Song 1 Title
============================================================
[00:00:05.00 - 00:00:10.50] First line of lyrics
[00:00:11.00 - 00:00:15.20] Second line of lyrics

============================================================
Song 2 Title
============================================================
[00:00:06.00 - 00:00:12.30] Different song content
...
```

---

### 3. Dynamic Letter Colors (Real Karaoke Effect)

#### What it does:
Letters automatically change color as they're sung - creating authentic karaoke effect

#### How it works:
- **Initial State**: Letters appear **white** ⚪
- **While Singing**: Letters smoothly **transition to blue** 🔵
- **Timing**: Perfectly synced with audio playback

#### Example:
```
As audio plays for word "Hello":
⚪ Hello  →  🔵 Hello  →  🔵 Hello
(before)     (singing)     (complete)
```

#### This is automatic!
- No configuration needed
- Enabled by default in all transcriptions
- Creates professional karaoke effect

---

## File Organization

```
Web_Output/
├── Sep_[filename]/         (from automatic creation)
│   ├── Vocals.wav
│   └── karaoke.ass
├── AutoEdit_Render/        (edited versions saved here)
│   └── Karaoke_Edit_*.mp4
├── Playlist_Transcriptions/
│   ├── VideoTitle1.txt
│   ├── VideoTitle2.txt
│   └── Combined_Transcriptions.txt
└── ... (other directories)
```

---

## 💡 Usage Tips

### For Automatic Creation:
- **Timing issue?** Edit text to match what you hear
- **Font too small?** Adjust size slider before clicking render
- **Need different color?** Change picker and click render (instant!)

### For Playlist Processing:
- **Public playlists only** - ensure videos are accessible
- **Large playlists?** Start with 2-3 videos to test
- **Download time**: ~2-3 minutes per hour of content

### For Dynamic Colors:
- **White → Blue** is perfect for Hebrew text
- **Works with any language** English, Hebrew, Arabic, etc.
- **Can't be disabled** - it's part of the karaoke effect

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Subtitles don't show in editor | Check transcription completed successfully (logs show "Transcription complete") |
| Edit button doesn't work | Make sure playback audio path is stored (happens automatically after initial creation) |
| Playlist fails to download | Check YouTube playlist is public, try with smaller playlist first |
| Blue color looks wrong | Use color picker to choose preferred color, click render |
| TXT files are empty | Check transcription language matches audio content |

---

## Keyboard Shortcuts (if using Gradio)
- **Tab**: Move between fields
- **Enter**: In textbox → focus render button
- **Ctrl+A**: Select all text in editor for copying

---

## File Locations
- **Automatic videos**: `Web_Output/AutoEdit_Render/`
- **Playlist TXT files**: `Web_Output/Playlist_Transcriptions/`
- **Custom path**: Whatever folder you specified (auto-copies there)

---

## Technical Specs

| Component | Spec |
|-----------|------|
| Color Transition | White (`#FFFFFF`) → Blue (`#0000FF`) |
| Transition Time | 30% of word duration |
| Font Size Range | 20-150 pixels |
| Supported Languages | Hebrew (he), English (en) |
| Playlist Limit | No hard limit (limited by YouTube API) |
| Output Format | MP4 (video), TXT (transcription) |

---

**Questions?** Check the IMPLEMENTATION_SUMMARY.md for technical details!
