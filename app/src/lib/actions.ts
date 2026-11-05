import { store } from '../store/store';
import { ui } from '../store/ui';
import { ws } from '../store/workspace';
import { downloadExport, type ExportKind } from './format';

/** Real download from the edited project data, recorded in the outputs library. */
export function exportFile(kind: ExportKind) {
  const p = store.get().project;
  if (!p) return;
  downloadExport(p, kind);
  ws.addOutput(kind);
}

/** MP4 needs the processing server: recorded as a simulated output, never a fake download. */
export function exportMp4() {
  ws.addOutput('mp4', true);
  ui.toast('סימולציה: רינדור MP4 דורש את שרת העיבוד. נרשם בספריית התוצרים כמדומה.');
}

/** Manual save (⌘S): saves and creates a version checkpoint. */
export function saveCheckpoint() {
  store.saveNow();
  ws.saveVersion('נקודת שמירה ידנית');
}
