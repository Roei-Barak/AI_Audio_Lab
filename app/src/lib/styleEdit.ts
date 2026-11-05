import { store } from '../store/store';
import type { SubtitleStyle } from '../store/types';

export function setStyle<K extends keyof SubtitleStyle>(k: K, v: SubtitleStyle[K]) {
  store.edit('עיצוב כתוביות', (P) => {
    if (P.style[k] === v) return false;
    P.style[k] = v;
  });
}
export const previewStyle = <K extends keyof SubtitleStyle>(k: K, v: SubtitleStyle[K]) =>
  store.preview((P) => {
    P.style[k] = v;
  });
