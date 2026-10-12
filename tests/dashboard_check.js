// 執行（先以 --fake 啟動服務並用 playwright-cli 開啟頁面）：
//   npx --package @playwright/cli playwright-cli -s=miroxen-ui run-code --filename tests/dashboard_check.js
async (page) => {
  let checks = 0;
  const errors = [], onError = (error) => errors.push(error.message);
  page.on("pageerror", onError);
  const check = (condition, message) => { if (!condition) throw new Error(message); checks++; };
  const rejects = () => page.evaluate(async () => { try { await buildRequest(); return false; } catch { return true; } });
  const task = (group, text) => page.locator("#" + group + " label", { hasText: text }).click();
  const view = (name) => page.locator(".views button", { hasText: name }).click();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.reload();
  await page.waitForFunction(() => state.models.length === 3 && state.loraRows.length >= 4);
  await page.locator("#prompt").fill("一隻紅熊貓");
  check((await page.locator("#prompt-count").innerText()).startsWith("5 /"), "提示詞字數");
  const png = Buffer.from(await page.evaluate(() => { const c = document.createElement("canvas"); c.width = c.height = 256; c.getContext("2d").fillRect(0, 0, 256, 256); return c.toDataURL("image/png").split(",")[1]; }), "base64");
  const fixture = { name: "reference.png", mimeType: "image/png", buffer: png };

  // 圖片：模型、LoRA、任務
  const imageLoras = await page.evaluate(() => state.loraRows.filter((r) => r.model === "qwen-image-2.1").length);
  check(await page.locator("#loras input[type=checkbox]").count() === imageLoras && imageLoras > 0, "圖片 LoRA 篩選");
  await page.locator("label.pick", { hasText: "Turbo" }).click();
  let body = await page.evaluate(() => buildRequest());
  check(body.model === "qwen-image-2.1-turbo" && body.steps === 8 && !body.sigmas && body.shift_terminal === undefined, "Turbo 模型固定 8 步");
  check(await page.locator("#steps").evaluate((el) => el.readOnly) && await page.locator("#sched-wrap").isHidden(), "Turbo 隱藏排程、鎖定步數");
  await page.locator("#loras input[type=checkbox]").first().check();
  body = await page.evaluate(() => buildRequest());
  check(body.loras.length === 1 && body.loras[0].scale === 1, "畫風 LoRA 可疊在 Turbo");
  check((await page.locator("#lora-summary").innerText()).includes("已選 1"), "LoRA 摘要");
  await page.locator("#loras input[type=checkbox]").first().uncheck();
  await page.locator("label.pick", { hasText: "標準" }).click();
  body = await page.evaluate(() => buildRequest());
  check(body.model === "qwen-image-2.1" && body.steps === 40 && body.shift_terminal === true, "切回標準 40 步");
  await page.locator("#lora-install > summary").click();
  await page.locator("#lora-url").fill("https://example.com/x.safetensors");
  await page.locator("#lora-url-install").click();
  await page.waitForFunction(() => document.getElementById("lora-help").textContent.includes("只支援"));
  check(true, "LoRA 網址驗證錯誤顯示");
  check((await page.locator("#gpu-chip").innerText()).includes("卡"), "顯示卡配置標籤");
  await page.locator("#sizes button", { hasText: "自訂" }).click();
  await page.locator("#width").fill("720");
  await page.locator("#height").fill("1280");
  check(await rejects(), "720 寬度須阻擋");
  await page.locator("#width").fill("736");
  await page.locator("#height").fill("1312");
  await task("image-task", "透明背景");
  body = await page.evaluate(() => buildRequest());
  check(body.size === "736x1312" && body.prompt.includes("RGBA"), "透明圖任務與自訂尺寸");
  await task("image-task", "去背");
  check(await rejects(), "去背需原圖");
  await task("image-task", "局部修改");
  await page.locator("#ref-input").setInputFiles(fixture);
  await page.waitForFunction(() => state.refs.length === 1);
  check(await rejects(), "遮罩需兩張圖");
  await page.locator("#ref-input").setInputFiles({ ...fixture, name: "mask.png" });
  await page.waitForFunction(() => state.refs.length === 2);
  body = await page.evaluate(() => buildRequest());
  check(body.images.length === 2 && body.prompt.includes("Picture 2 is a mask"), "局部修改素材");
  check(await page.evaluate(() => splitTask(requestSettings().prompt)[0] === "mask"), "任務提示可拆回");
  await task("image-task", "生成／編輯");
  await page.locator("#advanced > summary").click();
  await page.getByText("模型檔與量化版本", { exact: true }).click();
  await page.locator('#model-files select[data-role="dit"]').selectOption("test-Q4_K.gguf");
  body = await page.evaluate(() => buildRequest());
  check(body.model_files.dit === "test-Q4_K.gguf", "模型檔選擇");

  // 實際生成（假引擎）→ 詳情 → 重用設定
  await page.locator("#seed").fill("123");
  await page.locator("#sizes button", { hasText: "1:1 · 1024" }).click();
  await page.locator("#generate").click();
  await page.waitForFunction(() => !document.getElementById("result-img").hidden, null, { timeout: 15000 });
  check(await page.locator("#detail").isVisible() && (await page.locator("#detail-meta").innerText()).includes("123"), "作品詳情顯示種子");
  check(await page.locator("#result-tools").isVisible() && await page.locator("#history .hist").count() >= 1, "結果工具列與佇列");
  await page.locator("#prompt").fill("另一個提示");
  await page.locator("#seed").fill("-1");
  await page.locator("#reuse").click();
  check(await page.locator("#prompt").inputValue() === "一隻紅熊貓" && await page.locator("#seed").inputValue() === "123", "重用設定還原提示詞與種子");
  check(await page.evaluate(() => state.fileSelections["qwen-image-2.1"].dit === "test-Q4_K.gguf"), "重用設定還原模型檔");
  await view("作品庫");
  check(await page.locator("#library-grid .card").count() >= 1, "作品庫卡片");
  await page.locator("#library-filter button", { hasText: "影片" }).click();
  check(await page.locator("#library-grid .card").count() === 0, "作品庫篩選");
  await page.locator("#library-filter button", { hasText: "全部" }).click();
  await view("創作");

  // 影音
  await page.locator("#mode-video").click();
  await page.locator("#steps").fill("1");
  check(await rejects(), "影片至少兩個取樣點");
  check(await page.locator("#loras input[type=checkbox]").count() === 2, "FL2VA LoRA 篩選");
  await page.getByRole("checkbox", { name: /8step/ }).click();
  await page.locator("#duration").fill("10");
  body = await page.evaluate(() => buildRequest());
  check(body.steps === 9 && body.video_shift === 6 && body.num_frames === 240, "影片秒數與加速排程");
  check((await page.locator("#duration-help").innerText()).includes("243"), "顯示實際對齊幀數");
  await page.locator("#first-frame").setInputFiles(fixture);
  await page.waitForFunction(() => state.frames.first);
  check(await page.locator("#first-slot img").isVisible(), "首幀縮圖");
  body = await page.evaluate(() => buildRequest());
  check(body.first_frame.startsWith("data:image/png"), "首幀送出");
  await task("video-task", "參考素材生成");
  await page.getByRole("checkbox", { name: /ref2v/ }).click();
  await page.locator("#vref-input").setInputFiles(fixture);
  await page.waitForFunction(() => state.vrefs.length === 1);
  await page.locator("#vref-list .ref-tag").click();
  check((await page.locator("#prompt").inputValue()).includes("<Picture 1>"), "點標籤插入提示詞");
  await page.getByText("參考素材設定", { exact: true }).click();
  check(await page.locator("#ref-edge").inputValue() === "1024", "參考圖短邊預設 1024");
  await page.locator("#ref-edge").fill("512");
  await page.locator("#ref-native").check();
  body = await page.evaluate(() => buildRequest());
  check(body.video_shift === 12 && body.references.length === 1 && !body.first_frame && body.ref_image_short_edge === 512 && body.ref_native_canvas, "Ref2VA 參數與首幀互斥");

  // 模型與硬體、API
  await page.locator("#model-toggle").click();
  await page.waitForFunction(() => document.querySelectorAll("#gpus .gpu-row").length >= 1);
  check(await page.locator("#models .model-row").count() === 3, "模型清單");
  check((await page.locator("#host-metrics").innerText()).includes("GiB"), "主機監控資料");
  await view("API");
  check((await page.locator("#lan-help").innerText()).includes("API Base URL"), "其他電腦連線說明");
  check((await page.locator("#request-preview").innerText()).includes("ref_image_short_edge"), "API 預覽");
  check(await page.evaluate(() => getComputedStyle(document.querySelector("#api-panel .columns")).gridTemplateColumns.split(" ").length === 2), "桌面左右兩欄");
  check((await page.locator("#network-state").innerText()).length > 0, "網路存取狀態");

  // 響應式與對比
  for (const width of [1100, 390]) {
    await page.setViewportSize({ width, height: 844 });
    for (const name of ["API", "模型與硬體", "作品庫", "創作"]) {
      await view(name);
      check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), width + " " + name + " 橫向溢出");
    }
  }
  await page.screenshot({ path: "out/qa/dashboard-studio-mobile.png", fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1000 });
  for (const colorScheme of ["light", "dark"]) {
    await page.emulateMedia({ colorScheme, reducedMotion: "reduce" });
    await page.locator("#generate").hover();
    const contrast = await page.evaluate(() => {
      const style = getComputedStyle(document.getElementById("generate"));
      const canvas = document.createElement("canvas"), ctx = canvas.getContext("2d");
      const luminance = (color) => {
        ctx.fillStyle = color; ctx.fillRect(0, 0, 1, 1);
        const rgb = Array.from(ctx.getImageData(0, 0, 1, 1).data).slice(0, 3).map((v) => v / 255).map((v) => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
        return rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722;
      };
      const a = luminance(style.color), b = luminance(style.backgroundColor);
      return (Math.max(a, b) + .05) / (Math.min(a, b) + .05);
    });
    check(contrast >= 4.5, colorScheme + " 生成按鈕滑鼠移入後對比不足");
  }
  await page.emulateMedia({ colorScheme: "light" });
  await page.reload();
  await page.waitForFunction(() => state.models.length === 3);
  page.off("pageerror", onError);
  check(!errors.length, "JavaScript 錯誤：" + errors.join("; "));
  return { passed: true, checks };
}
