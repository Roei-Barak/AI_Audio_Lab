// Acceptance tests A/B/C from the R4 brief. Run `npm run build && npm run preview` first, then `npm run e2e`.
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ acceptDownloads: true, viewport: { width: 1440, height: 900 } });
  const pg = await ctx.newPage();
  const errs = [];
  pg.on('pageerror', (e) => errs.push(e.message));
  pg.on('console', (m) => m.type() === 'error' && errs.push(m.text()));
  const log = (k, v) => console.log((v ? 'PASS ' : 'FAIL ') + k);
  await pg.goto(process.env.LYRA_URL || 'http://localhost:4173/');
  log('A2 simple mode default', await pg.getByRole('button', { name: 'מצב פשוט' }).getAttribute('aria-pressed') === 'true');
  await pg.getByText('קישור לדוגמה').click();
  await pg.getByRole('button', { name: 'צור קריוקי' }).click();
  await pg.getByText('מפריד שירה ומוזיקה').waitFor();
  log('A5 progress visible', true);
  await pg.getByRole('heading', { name: 'בדיקה ותיקון מהיר' }).waitFor({ timeout: 60000 });
  log('A6 review opened automatically', true);
  // fix 1999 -> 1998 via UI
  await pg.getByRole('button', { name: /בקיץ של 1999/ }).click();
  await pg.getByRole('button', { name: '1999', exact: true }).click();
  const wi = pg.getByLabel('מילה', { exact: true });
  await wi.fill('1998'); await wi.press('Enter');
  await pg.locator('[aria-label="התחלה מאוחר"]').first().click();
  await pg.locator('[aria-label="התחלה מאוחר"]').first().click();
  await pg.waitForTimeout(300);
  // seek into the line and check karaoke shows 1998
  await pg.getByRole('button', { name: 'השמע' }).click();
  await pg.waitForTimeout(400);
  const kl = await pg.getByTestId('karaoke-line').first().innerText();
  log('A8 captions updated: ' + kl.replace(/\s+/g,' '), kl.includes('1998'));
  await pg.getByRole('button', { name: 'המשך להורדה' }).click();
  for (const [lbl, ext] of [['הורד כתוביות SRT', 'srt'], ['הורד מילות שיר TXT', 'txt'], ['הורד פרויקט JSON', 'json']]) {
    const [d] = await Promise.all([pg.waitForEvent('download'), pg.getByRole('button', { name: new RegExp(lbl) }).click()]);
    const p = await d.path(); const txt = fs.readFileSync(p, 'utf8');
    log('A9-10 ' + ext + ' file contains fix (' + d.suggestedFilename() + ')', txt.includes('1998') && !txt.includes('1999 '));
    if (ext === 'srt') console.log(txt.split('\n').slice(28, 32).join(' | '));
  }
  await pg.getByRole('button', { name: 'פתח בסטודיו המקצועי' }).click();
  await pg.getByRole('tab', { name: /זום מילה/ }).waitFor();
  log('A11-12 advanced shows same edit', (await pg.getByTestId('zoom-word').allInnerTexts()).includes('1998'));
  await pg.waitForTimeout(1200);
  await pg.reload();
  await pg.getByRole('tab', { name: /זום מילה/ }).waitFor();
  log('A13-14 restored after refresh', (await pg.getByTestId('zoom-word').allInnerTexts()).includes('1998'));
  // Test B: edit in Lyrics First, check timeline, undo/redo
  await pg.keyboard.press('2');
  const row8 = pg.getByLabel('שורה 9');
  await row8.fill('וכל אחד נושא את שמך'); await row8.press('Enter');
  await pg.waitForTimeout(200);
  await pg.keyboard.press('1');
  const clips = await pg.getByTestId('clip').allInnerTexts();
  log('B3 timeline reflects lyrics edit', clips.some((c) => c.includes('נושא')));
  await pg.keyboard.press('Control+z');
  log('B4 undo', !(await pg.getByTestId('clip').allInnerTexts()).some((c) => c.includes('נושא')));
  await pg.keyboard.press('Control+Shift+z');
  log('B4 redo', (await pg.getByTestId('clip').allInnerTexts()).some((c) => c.includes('נושא')));
  await pg.getByRole('button', { name: '4 · עיצוב' }).click();
  await pg.getByRole('button', { name: /מודגש/ }).first().click();
  log('B5 template changed', await pg.getByRole('button', { name: /מודגש/ }).first().getAttribute('aria-pressed') === 'true');
  // Test C: offline + conflict
  await pg.getByRole('button', { name: /שרת לא זמין/ }).click();
  await pg.getByRole('button', { name: /מינימלי/ }).first().click();
  await pg.getByText(/ממתינים לסנכרון/).waitFor();
  log('C offline browser save', true);
  await pg.getByRole('button', { name: /התנגשות בשמירה הבאה/ }).click();
  await pg.getByRole('button', { name: /שרת לא זמין/ }).click();
  await pg.getByRole('dialog', { name: 'התנגשות גרסאות' }).waitFor();
  await pg.getByRole('button', { name: 'שמור את שלי' }).click();
  await pg.getByText(/^נשמר בדפדפן ·/).waitFor();
  log('C conflict resolved', true);
  await pg.setViewportSize({ width: 390, height: 844 });
  await pg.getByRole('button', { name: 'מצב פשוט' }).click();
  const sw = await pg.evaluate(() => document.documentElement.scrollWidth);
  log('mobile no horizontal page scroll (' + sw + ')', sw <= 390);
  console.log('errors:', JSON.stringify(errs));
  await b.close();
})().catch((e) => { console.error('E2E ERROR', e.message); process.exit(1); });
