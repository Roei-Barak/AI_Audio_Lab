import { store, useStore } from '../../store/store';
import { ui, type StateKind } from '../../store/ui';

interface Def {
  icon: string;
  title: string;
  text: string;
  actions: [string, () => void, boolean?][];
}

function defs(kind: StateKind, onClear?: () => void): Def {
  const S = store.get();
  const remote = S.sim.conn === 'remote';
  const toSimple = () => { ui.setStep('input'); ui.go('simple'); };
  const D: Record<StateKind, Def> = {
    background: {
      icon: 'ph ph-moon',
      title: 'הממשק נסגר · העבודה המשיכה',
      text: (remote ? 'השרת המרוחק' : 'שירות Lyra במחשב הזה') + ' המשיך לעבד גם כשהדפדפן היה סגור. ההתקדמות עודכנה כשחזרת.',
      actions: [['לתור העיבוד', () => ui.go('queue'), true]],
    },
    offline: {
      icon: 'ph ph-plugs',
      title: remote ? 'השרת המרוחק לא זמין' : 'שירות Lyra במחשב הזה לא מגיב',
      text: 'עריכות נשמרות בדפדפן ויסונכרנו כשהחיבור יחזור. עיבוד חדש וייצוא MP4 מושהים. מצב העבודות שרצות לא ידוע.',
      actions: [['נסה להתחבר שוב', () => { store.setSim('offline', false); ui.toast('החיבור חזר · מסנכרן'); }, true], ['המשך לערוך', () => ui.go('subs-word')]],
    },
    crash: {
      icon: 'ph ph-lightning-slash',
      title: 'קריסת עיבוד · דורש התאוששות',
      text: 'תהליך ה־GPU הופסק באופן לא צפוי. השלבים שהושלמו נשמרו, ואפשר להמשיך מהשלב שנעצר או לנסות בפרופיל מהיר. זו קריסה, לא בעיה בקובץ.',
      actions: [['לתור העיבוד', () => ui.go('queue'), true]],
    },
    failed: {
      icon: 'ph ph-warning-circle',
      title: 'העיבוד נכשל בגלל תוכן הקובץ',
      text: 'לא זוהתה שירה בקובץ. אפשר לנסות שוב ברגישות נמוכה, להפריד מחדש ל־4 ערוצים, או להדביק מילים ידנית ולסנכרן אותן.',
      actions: [['הדבק מילים ידנית', () => { ui.go('subs-lyrics'); ui.open('lyrics'); }, true], ['העלה קובץ אחר', toSimple]],
    },
    recovery: {
      icon: 'ph ph-clock-counter-clockwise',
      title: 'שחזור טיוטה',
      text: 'נמצאו בדפדפן שינויים שלא סונכרנו מהפעם הקודמת. הם נטענו לפרויקט ויסונכרנו בשמירה הבאה.',
      actions: [['פתח את הפרויקט', () => ui.go('subs-word'), true], ['השווה גרסאות', () => ui.go('project-versions')]],
    },
    emptyDashboard: {
      icon: 'ph ph-microphone-stage',
      title: 'עדיין אין פרויקטים',
      text: 'הדבק קישור YouTube או העלה קובץ, ו־Lyra תיצור סרטון קריוקי עם מילים מסונכרנות.',
      actions: [['צור קריוקי ראשון', toSimple, true]],
    },
    emptyLibrary: {
      icon: 'ph ph-magnifying-glass',
      title: 'אין תוצאות',
      text: 'לא נמצא שיר שמתאים לחיפוש.',
      actions: onClear ? [['נקה חיפוש', onClear, true]] : [],
    },
    loading: { icon: 'ph ph-spinner-gap spin', title: 'טוען…', text: 'טוען את נתוני הפרויקט.', actions: [] },
    forbidden: {
      icon: 'ph ph-lock-simple',
      title: 'אין לך הרשאה למסך הזה',
      text: 'הפעולה זמינה למנהלים בלבד. בקש מהמנהל לשנות את התפקיד שלך.',
      actions: [['ללוח הבקרה', () => ui.go('dashboard'), true]],
    },
    notfound: {
      icon: 'ph ph-compass',
      title: 'הדף לא נמצא',
      text: 'הקישור שגוי או שהפרויקט נמחק.',
      actions: [['ללוח הבקרה', () => ui.go('dashboard'), true]],
    },
  };
  return D[kind];
}

export function StateView({ kind, inline, onClear }: { kind: StateKind; inline?: boolean; onClear?: () => void }) {
  useStore();
  const d = defs(kind, onClear);
  const body = (
    <div className="panel" role={kind === 'crash' || kind === 'offline' ? 'alert' : undefined} style={{ padding: 28, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, textAlign: 'center', maxWidth: 520, margin: '0 auto' }}>
      <i className={d.icon} style={{ fontSize: 34, color: 'var(--color-accent-300)' }} />
      <h3 style={{ margin: 0, fontSize: 20, fontWeight: 500 }}>{d.title}</h3>
      <p className="muted" style={{ margin: 0, fontSize: 14, textWrap: 'pretty' }}>{d.text}</p>
      {d.actions.length > 0 && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
          {d.actions.map(([l, f, primary]) => <button key={l} className={primary ? 'btn btn-primary' : 'btn btn-secondary'} onClick={f}>{l}</button>)}
        </div>
      )}
    </div>
  );
  if (inline) return body;
  return <div dir="rtl" style={{ height: '100%', overflow: 'auto', display: 'flex', alignItems: 'center', padding: 16, background: 'var(--color-bg)' }}><div style={{ width: '100%' }}>{body}</div></div>;
}
