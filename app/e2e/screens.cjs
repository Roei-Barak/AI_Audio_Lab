// Visits every route, checks for runtime errors and horizontal page overflow, and exercises the new flows.
const { chromium } = require('playwright');
const URL = process.env.LYRA_URL || 'http://localhost:4173/';
const OUT = process.argv[2];
const ROUTES = ['dashboard', 'library', 'queue', 'outputs', 'team', 'options', 'subs', 'subs-lyrics', 'subs-word', 'karaoke', 'export', 'project', 'project-versions', 'audio', 'video',
  'settings', 'settings-general', 'settings-models', 'settings-storage', 'settings-api', 'invite', 'forgot',
  'state:background', 'state:offline', 'state:crash', 'state:failed', 'state:recovery', 'state:emptyDashboard', 'state:emptyLibrary', 'state:loading', 'state:forbidden', 'state:notfound', 'login'];
(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ acceptDownloads: true, viewport: { width: 1440, height: 900 } });
  const pg = await ctx.newPage();
  const errs = [];
  let fails = 0;
  pg.on('pageerror', (e) => errs.push(e.message));
  pg.on('console', (m) => m.type() === 'error' && !/fonts\.g/.test(m.text()) && errs.push(m.text()));
  const log = (k, v) => { if (!v) fails++; console.log((v ? 'PASS ' : 'FAIL ') + k); };
  await pg.goto(URL);
  await pg.evaluate(() => localStorage.clear());
  await pg.goto(URL + '#dashboard');
  await pg.reload();
  for (const r of ROUTES) {
    await pg.evaluate((r) => { location.hash = r; }, r);
    await pg.reload();
    await pg.waitForTimeout(250);
    const sw = await pg.evaluate(() => document.documentElement.scrollWidth);
    const txt = await pg.locator('main.stage').innerText();
    log('route ' + r + ' renders', txt.trim().length > 20 && sw <= 1440);
  }
  // login gate -> sign in
  await pg.evaluate(() => { location.hash = 'dashboard'; }); await pg.reload();
  await pg.getByRole('button', { name: 'התנתקות' }).click();
  await pg.getByLabel('סיסמה').fill('1234');
  await pg.getByRole('button', { name: 'התחבר' }).click();
  log('logout + login', await pg.getByRole('heading', { name: /שלום/ }).isVisible());
  // invite flow
  await pg.evaluate(() => { location.hash = 'team'; }); await pg.reload();
  await pg.getByRole('button', { name: 'צור קישור הזמנה' }).first().click();
  await pg.getByPlaceholder('למשל: אורי אלון').fill('אורי אלון');
  await pg.getByRole('dialog').getByRole('button', { name: 'צור קישור הזמנה' }).click();
  const link = await pg.getByRole('textbox', { name: 'קישור' }).inputValue();
  await pg.getByRole('button', { name: 'סיום' }).click();
  await pg.getByRole('button', { name: 'התנתקות' }).click();
  await pg.getByText('יש לי קישור הזמנה').click();
  await pg.getByLabel('קישור או קוד הזמנה').fill(link);
  await pg.getByLabel('שם משתמש').fill('uri');
  await pg.getByLabel('סיסמה').fill('password1');
  await pg.getByRole('button', { name: 'צור חשבון' }).click();
  log('invite accepted, signed in as new user', await pg.getByText('אורי אלון').first().isVisible());
  await pg.getByRole('button', { name: 'התנתקות' }).click();
  await pg.getByLabel('סיסמה').fill('1234'); await pg.getByRole('button', { name: 'התחבר' }).click();
  // find & replace
  await pg.evaluate(() => { location.hash = 'subs-word'; }); await pg.reload();
  await pg.waitForTimeout(400);
  await pg.keyboard.press('Control+f');
  await pg.getByLabel('חפש').fill('נשא'); await pg.getByLabel('החלף ב').fill('נושא');
  await pg.getByRole('button', { name: 'החלף הכל' }).click();
  await pg.keyboard.press('2');
  log('find & replace applied', (await pg.getByLabel('שורה 9').inputValue()).includes('נושא'));
  // checkpoint + restore
  await pg.keyboard.press('Control+s');
  await pg.getByLabel('שורה 9').fill('שורה ששונתה'); await pg.getByLabel('שורה 9').press('Enter');
  await pg.evaluate(() => { location.hash = 'project-versions'; }); await pg.reload();
  await pg.getByRole('button', { name: 'שחזר' }).first().click();
  await pg.getByRole('dialog').getByRole('button', { name: 'שחזר' }).click();
  await pg.evaluate(() => { location.hash = 'subs-lyrics'; }); await pg.reload();
  log('version restored', (await pg.getByLabel('שורה 9').inputValue()).includes('נושא'));
  // word zoom drag
  await pg.keyboard.press('3');
  const w = pg.getByTestId('zoom-word').nth(1);
  const bb = await w.boundingBox();
  const before = await pg.locator('.num').filter({ hasText: /^\d+\.\d\d$/ }).first().innerText().catch(() => '');
  await pg.mouse.move(bb.x + bb.width / 2, bb.y + bb.height / 2); await pg.mouse.down();
  await pg.mouse.move(bb.x + bb.width / 2 + 40, bb.y + bb.height / 2, { steps: 5 }); await pg.mouse.up();
  const hist = await pg.locator('aside').innerText();
  log('word zoom drag recorded', /הזזת מילה/.test(hist));
  // outputs record
  await pg.evaluate(() => { location.hash = 'export'; }); await pg.reload();
  const [d] = await Promise.all([pg.waitForEvent('download'), pg.getByRole('button', { name: 'LRC' }).first().click()]);
  log('LRC download ' + d.suggestedFilename(), d.suggestedFilename().endsWith('.lrc'));
  await pg.evaluate(() => { location.hash = 'outputs'; }); await pg.reload();
  log('output listed', await pg.getByText('yareach-shel-august.lrc').isVisible());
  // waiting queue reorder
  await pg.evaluate(() => { location.hash = 'queue'; }); await pg.reload();
  const qtxt = await pg.locator('main.stage').innerText();
  log('queue shows waiting list', /Neon Rain/.test(qtxt) || /אין עבודה בתור/.test(qtxt));
  // background catch-up
  await pg.evaluate(() => { const j = { id: 'jx', name: 'bg', profile: 'מהיר', stems: 2, status: 'running', stage: 0, pct: 0, attempt: 1, startedAt: Date.now() - 60000, updatedAt: Date.now() - 60000 }; localStorage.setItem('lyra.v1.job', JSON.stringify(j)); location.hash = 'queue'; });
  await pg.reload(); await pg.waitForTimeout(1600);
  log('background catch-up notice', /העבודה המשיכה ברקע/.test(await pg.locator('.toast').innerText().catch(() => '')));
  for (const [r, f] of [['dashboard', 'dashboard'], ['project', 'project'], ['audio', 'audio'], ['team', 'team']]) {
    await pg.evaluate((r) => { location.hash = r; }, r); await pg.reload(); await pg.waitForTimeout(300);
    await pg.screenshot({ path: OUT + '/' + f + '.png' });
  }
  console.log('errors:', JSON.stringify(errs));
  await b.close();
  process.exit(fails || errs.length ? 1 : 0);
})().catch((e) => { console.error('E2E ERROR', e.message); process.exit(1); });
