// Offline browser verification; uses the desktop's bundled Playwright.
const fs = require('fs');
const assert = require('assert/strict');
const crypto = require('crypto');
const path = require('path');
const {pathToFileURL, fileURLToPath} = require('url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'C:/Users/iiyee/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const report = process.env.C2B_REPORT_PATH
  ? path.resolve(process.env.C2B_REPORT_PATH)
  : path.resolve(__dirname, '../../Explore/runs/reports/C2b_results_20261110.html');
(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.EDGE_PATH || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  const page = await browser.newPage({viewport: {width: 1400, height: 980}});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto(pathToFileURL(report).href);
  assert.deepEqual(errors, []);
  for (const ds of ['searchqa', 'hotpotqa']) {
    await page.selectOption('#case-dataset', ds);
    await page.selectOption('#case-filter', 'all');
    assert.equal(await page.locator('#case-index option').count(), 30);
    const ids = await page.locator('#case-index option').evaluateAll(xs => xs.map(x => x.value));
    for (const id of ids) {
      await page.selectOption('#case-index', id);
      assert.equal(await page.locator('#responses .response').count(), 3);
      assert.ok((await page.locator('#question').textContent()).length > 5);
    }
    for (const filter of ['wrong', 'changed']) {
      await page.selectOption('#case-filter', filter);
      const n = await page.locator('#case-index option').count();
      assert.equal(n, {searchqa: {wrong: 8, changed: 3}, hotpotqa: {wrong: 11, changed: 0}}[ds][filter]);
    }
    for (const method of ['B', 'C', 'C2b']) {
      await page.selectOption('#skill-dataset', ds);
      await page.selectOption('#skill-method', method);
      assert.equal(await page.locator('#skill-index option').count(), 6);
      for (let i = 0; i < 6; i++) {
        await page.selectOption('#skill-index', String(i));
        assert.ok((await page.locator('#skill-meta').textContent()).length);
        if (ds === 'hotpotqa' && method === 'C2b') {
          const anomalyText = await page.locator('#skill-anomalies').textContent();
          const expected = [0, 2, 0, 1, 0, 1][i];
          if (expected) {
            const anomalies = JSON.parse(anomalyText);
            assert.equal(anomalies.length, expected);
            assert.ok(anomalies.every(row => !row.format_ok && row.attempts.some(a => a.usage?.finish_reason === 'length')));
          } else assert.ok(anomalyText.includes('没有输出截断'));
        }
      }
    }
    for (const method of ['C', 'C2b']) {
      await page.selectOption('#feedback-dataset', ds);
      await page.selectOption('#feedback-method', method);
      assert.equal(await page.locator('#feedback-index option').count(), 6);
      for (let i = 0; i < 6; i++) {
        await page.selectOption('#feedback-index', String(i));
        assert.ok((await page.locator('#feedback-text').textContent()).length > 0);
      }
    }
  }
  const links = await page.locator('a[download]').evaluateAll(xs => xs.map(x => x.href));
  for (const link of links) assert.ok(fs.existsSync(fileURLToPath(link)), link);
  const images = await page.locator('img').evaluateAll(xs => xs.map(x => ({src: x.src, loaded: x.complete && x.naturalWidth > 0})));
  for (const im of images) {
    assert.ok(fs.existsSync(fileURLToPath(im.src)));
    assert.ok(im.loaded, im.src);
  }
  assert.deepEqual(errors, []);
  assert.ok(!/__([A-Z_]+)__/.test(await page.locator('body').textContent()));
  await page.locator('#evolution .grid').screenshot({path: report + '.curves.png'});
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({path: report + '.desktop.png'});
  await page.setViewportSize({width: 390, height: 844});
  await page.evaluate(() => window.scrollTo(0, 0));
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.screenshot({path: report + '.mobile.png'});
  const validation = {status: 'passed', report_sha256: crypto.createHash('sha256').update(fs.readFileSync(report)).digest('hex'),
    dev_questions: 60, responses: 180, candidate_slots: 36, feedback_exposures: 24,
    download_links: links.length, images: images.length, page_errors: errors, mobile_overflow: false};
  fs.writeFileSync(report.replace('.html', '.validation.json'), JSON.stringify(validation, null, 2));
  console.log(JSON.stringify(validation));
  await browser.close();
})().catch(e => {console.error(e); process.exit(1);});
