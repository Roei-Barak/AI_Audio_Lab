# Lyra Studio · V1 (R4) frontend

React 18 + Vite + TypeScript implementation of `project/Lyra Studio V1 R4.html` (Claude Design handoff), core V1 flow.
Styling uses the Nocturne token sheet (`src/styles/nocturne.css`), Phosphor icons, and Inter + Noto Sans Hebrew. The UI is RTL; timelines run left to right.

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests (store, exports, drag logic)
npm run build && npm run preview   # then, in another shell:
npm run e2e        # browser acceptance tests A/B/C (Playwright)
node e2e/screens.cjs /tmp   # every route + auth, invite, find/replace, versions, drag, outputs
```

## What's in scope
- **Simple Mode** (the default): source (YouTube link or local file) → processing progress (fail, retry, cancel) → preview with quick fixes (line, word, timing, template, color, size, position) → downloads.
- **Advanced Studio**: processing options → queue (crash, resume, fast-profile retry, cancel) → lyrics editor (Timeline, Lyrics First, Word Zoom) → karaoke design and export.
- **One shared project** (`src/store/store.ts`) is used by every screen. Edits are undoable (60 steps), go into the history log and autosave to `localStorage`. Media is never stored.
- **Save states**: saving, saved, failed with retry, offline browser save with pending count and sync on reconnect, conflict resolution. All are labelled "נשמר בדפדפן", never "saved on server".
- **AI re-run as a suggestion**: changes are accepted or rejected per line, and manually fixed words are locked.
- **Real SRT/TXT/JSON downloads** built from the edited data.

## Processing server
`../server` is the real backend: import (yt-dlp or upload), separation, Whisper transcription with word timing, alignment and ffmpeg MP4 render, with one GPU job at a time, crash checkpoints, cancel and resume, and save conflicts detected by server revision. When `/api/health` answers, the app switches to server mode automatically. The "שרת Lyra" badge appears, and:
- Sign-in, invites, roles, settings, version checkpoints and the outputs library live on the server.
- Saves go to the server.
- MP4, Vocals and Instrumental downloads are real files.
- Pasted lyrics are aligned to the transcription. Without a server, everything below stays simulated (demo mode). See `../server/README.md`.

## Simulated in demo mode (no server)
YouTube import, stem separation, transcription, alignment, MP4 render and Vocals/Instrumental downloads; the server, the connection state, and the other user in a conflict. The UI labels each of these. Uploaded local media plays for real, but with its original audio.

- **Advanced screens on the same data**: dashboard, library, project details (files, version checkpoints with restore, activity, re-run as suggestion), Audio Studio mixer (saved in the project), video editor (tracks and a global subtitle offset), export (MP4 simulated; SRT/LRC/TXT/JSON real), outputs library, settings (server, general defaults used by Simple Mode, models, storage, local API token), team and permissions.
- **Private access**: sign-in, admin-only one-time invite links (copied by hand, no emails), accept invite, password reset through the admin. Auth is simulated locally.
- **Modals**: conflict, AI suggestion comparison, share, invite, find & replace (⌘F), restore version, delete project, shortcuts (?).
- **States**: UI closed while work continued (real catch-up on reload), server unavailable, crash, content failure, draft recovery, empty, loading, forbidden, 404.

## Not included
A real processing backend (see "Simulated"). Cover Studio stays a future capability (designs are in `../project`). Other library entries and waiting queue jobs are illustrative only.
