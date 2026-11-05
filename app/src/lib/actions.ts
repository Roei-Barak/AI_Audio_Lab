import { store } from '../store/store';
import { ui } from '../store/ui';
import { ws } from '../store/workspace';
import { downloadExport, type ExportKind } from './format';
import { fileUrl, flushSave, isServer, renderFresh, submitJob } from '../store/server';

/** Real download from the edited project data, recorded in the outputs library. */
export function exportFile(kind: ExportKind) {
  const p = store.get().project;
  if (!p) return;
  downloadExport(p, kind);
  ws.addOutput(kind);
}

function download(url: string) {
  const a = document.createElement('a');
  a.href = url;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export const serverProject = () => {
  const p = store.get().project;
  return isServer() && p?.serverRev != null && p.processed ? p : null;
};

/**
 * MP4: with the server, downloads the render when it matches the current edits, otherwise renders again.
 * Without the server it is recorded as simulated, never a fake download.
 */
export async function exportMp4(): Promise<'downloaded' | 'rendering' | 'simulated' | 'busy'> {
  const p = serverProject();
  if (!p) {
    ws.addOutput('mp4', true);
    ui.toast('סימולציה: רינדור MP4 דורש את שרת העיבוד. נרשם בספריית התוצרים כמדומה.');
    return 'simulated';
  }
  const j = store.get().job;
  if (j?.serverId && j.status === 'running') {
    ui.toast('השרת עדיין מעבד · ההורדה תהיה זמינה בסיום');
    return 'busy';
  }
  await flushSave();
  const cur = store.get().project!;
  if (renderFresh(cur)) {
    download(fileUrl(cur.id, 'karaoke', true));
    ws.addOutput('mp4');
    return 'downloaded';
  }
  try {
    await submitJob('render');
    ui.toast('מרנדר MP4 עם העריכות האחרונות · ההורדה תהיה זמינה בסיום');
    return 'rendering';
  } catch (e) {
    ui.toast((e as Error).message);
    return 'busy';
  }
}

/** Separated stems are real files on the server. */
export function downloadStem(name: 'vocals' | 'instrumental' | 'drums' | 'bass' | 'other'): boolean {
  const p = serverProject();
  if (!p) return false;
  download(fileUrl(p.id, name, true));
  return true;
}

/** Manual save (⌘S): saves and creates a version checkpoint. */
export function saveCheckpoint() {
  store.saveNow();
  ws.saveVersion('נקודת שמירה ידנית');
}
