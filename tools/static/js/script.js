// 切换标签页
function showTab(name) {
  document
    .querySelectorAll(".tab-content")
    .forEach((el) => el.classList.remove("active"));
  document
    .querySelectorAll(".tab-btn")
    .forEach((el) => el.classList.remove("active"));
  document.getElementById(name).classList.add("active");

  // Handle click on children elements
  const btn = event.target.closest(".tab-btn");
  if (btn) btn.classList.add("active");
}

// 设置上传区域（点击+拖拽）
function setupUpload(dropId, inputId, previewId) {
  const drop = document.getElementById(dropId);
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);

  // 点击上传
  drop.addEventListener("click", () => input.click());

  // 文件选择后预览
  input.addEventListener("change", () => {
    if (input.files[0]) showPreview(input.files[0], preview, drop);
  });

  // 拖拽事件
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("dragover");
  });

  drop.addEventListener("dragleave", () => {
    drop.classList.remove("dragover");
  });

  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (
      file &&
      (file.type.startsWith("image/") || file.type.startsWith("video/"))
    ) {
      // 更新input的files
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      showPreview(file, preview, drop);
    }
  });
}

function showPreview(file, preview, drop) {
  const objectUrl = URL.createObjectURL(file);

  if (file.type.startsWith("video/")) {
    // Warning: preview element MUST be a <video> tag
    preview.src = objectUrl;
    preview.style.display = "block";
    preview.style.opacity = "1";
    preview.controls = true;

    // Try to play automatically (muted) to verify decoding
    preview.muted = true;
    preview.play().catch((e) => {
      console.warn("Auto-play failed:", e);
    });

    preview.onerror = () => {
      console.error("Video load error");
      const drop = preview.closest(".upload-box");
      const nameEl = drop ? drop.querySelector("#video-name") : null;
      if (nameEl) {
        nameEl.innerHTML +=
          ' <span style="color:orange; font-size:12px;">(无法在浏览器预览，但不影响检测)</span>';
      }
    };

    // Find or create a name element
    const nameEl =
      drop.querySelector("#video-name") ||
      document.getElementById("video-name");
    if (nameEl) {
      nameEl.textContent = `选中文件: ${file.name}`;
      nameEl.style.display = "block";
    }
  } else {
    // Image Preview handling
    preview.src = objectUrl;
    preview.style.display = "block";
    preview.style.opacity = "0";
    setTimeout(() => (preview.style.opacity = "1"), 10);
  }

  // 隐藏上传提示
  const icon = drop.querySelector(".upload-icon");
  const text = drop.querySelector(".upload-text");
  if (icon) icon.style.display = "none";
  if (text) text.style.display = "none";
}

// 初始化上传区域
setupUpload("detect-drop", "detect-file", "detect-preview");
setupUpload("source-drop", "source-file", "source-preview");
setupUpload("target-drop", "target-file", "target-preview");
setupUpload("video-drop", "video-file", "video-preview");

// 加载模型列表
// 全局存储模型数据
let allModels = {};
let currentCategory = "";
let lastSwappedImage = null; // 存储最近一次换脸结果，用于重新检测

// 加载模型列表
fetch("/api/models")
  .then((r) => r.json())
  .then((data) => {
    allModels = data;
    const categories = Object.keys(data);
    if (categories.length === 0) return;

    // 渲染分类
    const catContainer = document.getElementById("model-categories");
    catContainer.innerHTML = "";

    categories.forEach((cat, index) => {
      const btn = document.createElement("button");
      btn.className = "cat-btn";
      btn.textContent = cat; // 这里后端返回的是 Category Key
      if (index === 0) {
        btn.classList.add("active");
        currentCategory = cat;
      }

      btn.onclick = () => {
        document
          .querySelectorAll(".cat-btn")
          .forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        renderModels(cat);
      };
      catContainer.appendChild(btn);
    });

    // 初始渲染第一个分类的模型
    renderModels(currentCategory);

    // 默认选中第一个分类的第一个模型
    if (data[currentCategory] && data[currentCategory].length > 0) {
      selectModel(data[currentCategory][0].id);
    }

    // 同时填充换脸页面的迷你模型选择器
    populateMiniModelSelect(data);
  })
  .catch((e) => {
    console.error(e);
    document.getElementById("model-options-grid").innerHTML =
      '<div style="color:red">Loading failed</div>';
  });

// 填充迷你模型选择器（用于换脸后自动检测）
function populateMiniModelSelect(data) {
  const select = document.getElementById("swap-detect-model");
  if (!select) return;

  select.innerHTML = "";
  for (const [group, models] of Object.entries(data)) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = group;
    models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      optgroup.appendChild(opt);
    });
    select.appendChild(optgroup);
  }
}

function renderModels(category) {
  const grid = document.getElementById("model-options-grid");
  grid.innerHTML = "";

  const models = allModels[category] || [];
  const currentSelected = document.getElementById("model-select").value;

  models.forEach((m) => {
    const div = document.createElement("div");
    div.className = "model-option";
    div.dataset.id = m.id; // Store ID

    if (m.id === currentSelected) {
      div.classList.add("active");
    }

    div.textContent = m.label;
    div.onclick = () => selectModel(m.id);
    grid.appendChild(div);
  });
}

function selectModel(id) {
  // 更新隐藏Input
  document.getElementById("model-select").value = id;

  // 更新UI高亮 state
  document.querySelectorAll(".model-option").forEach((el) => {
    if (el.dataset.id === id) {
      el.classList.add("active");
    } else {
      el.classList.remove("active");
    }
  });
}

// 检测
async function runDetect() {
  const file = document.getElementById("detect-file").files[0];
  if (!file) return alert("请先上传图片");

  const btn = document.getElementById("detect-btn");
  const result = document.getElementById("detect-result");

  btn.disabled = true;
  btn.textContent = "Processing...";
  result.className = "result show";
  result.innerHTML = '<div style="color:#6b7280;">Analyzing image...</div>';

  const form = new FormData();
  form.append("image", file);
  form.append("model_name", document.getElementById("model-select").value);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();

    if (data.error) {
      result.innerHTML = '<span class="error">错误: ' + data.error + "</span>";
    } else {
      const fakeProb = data.fake * 100;
      const realProb = data.real * 100;
      const isFake = data.fake > 0.5;
      const prob = isFake ? fakeProb : realProb;
      const colorClass = isFake ? "fake" : "real";
      const label = isFake ? "Fake (伪造)" : "Real (真实)";

      result.innerHTML = `
                <div class="result-header">
                    <span class="result-badge ${colorClass}">${
        isFake ? "DETECTED FAKE" : "LEGITIMATE IMAGE"
      }</span>
                </div>
                
                <div class="progress-container">
                    <div class="progress-label">
                        <span>${label}</span>
                        <span>${prob.toFixed(1)}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill ${colorClass}" style="width: ${prob}%"></div>
                    </div>
                </div>

                ${
                  data.note
                    ? '<div class="result-detail" style="margin-top:16px">' +
                      data.note +
                      "</div>"
                    : ""
                }
            `;
    }
  } catch (e) {
    result.innerHTML =
      '<span class="error">请求失败，请检查服务是否正常</span>';
  }

  btn.disabled = false;
  btn.textContent = "开始检测";
}

// 视频检测 - 根据当前模式选择不同的检测方式
let currentVideoInputMode = "video"; // 'video' 或 'frames'

// 切换视频输入模式
function switchVideoInputMode(mode) {
  currentVideoInputMode = mode;

  // 更新按钮样式
  document
    .getElementById("mode-video-btn")
    .classList.toggle("active", mode === "video");
  document
    .getElementById("mode-frames-btn")
    .classList.toggle("active", mode === "frames");

  // 切换显示的上传区域
  document.getElementById("video-upload-section").style.display =
    mode === "video" ? "block" : "none";
  document.getElementById("frames-upload-section").style.display =
    mode === "frames" ? "block" : "none";

  // 清空之前的结果
  document.getElementById("video-result").className = "result";
  document.getElementById("video-result").innerHTML = "";
  document.getElementById("feature-details").style.display = "none";
}

// 设置多图片上传
function setupFramesUpload() {
  const drop = document.getElementById("frames-drop");
  const input = document.getElementById("frames-files");
  const countEl = document.getElementById("frames-count");
  const previewGrid = document.getElementById("frames-preview-grid");

  if (!drop || !input) return;

  // 点击上传
  drop.addEventListener("click", () => input.click());

  // 文件选择后预览
  input.addEventListener("change", () => {
    if (input.files.length > 0) {
      showFramesPreview(input.files, countEl, previewGrid, drop);
    }
  });

  // 拖拽事件
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("dragover");
  });

  drop.addEventListener("dragleave", () => {
    drop.classList.remove("dragover");
  });

  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("dragover");
    const files = Array.from(e.dataTransfer.files).filter((f) =>
      f.type.startsWith("image/")
    );
    if (files.length > 0) {
      // 更新input的files
      const dt = new DataTransfer();
      files.forEach((f) => dt.items.add(f));
      input.files = dt.files;
      showFramesPreview(files, countEl, previewGrid, drop);
    }
  });
}

// 显示多图片预览
function showFramesPreview(files, countEl, previewGrid, drop) {
  const fileList = Array.from(files);

  // 按文件名排序（确保帧顺序正确）
  fileList.sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { numeric: true })
  );

  // 更新计数
  const isEnough = fileList.length >= 10;
  countEl.textContent = `已选择 ${fileList.length} 帧 ${
    isEnough ? "✓" : "(至少需要10帧)"
  }`;
  countEl.style.color = isEnough ? "#16a34a" : "#dc2626";
  countEl.style.display = "block";

  // 隐藏上传提示
  const icon = drop.querySelector(".upload-icon");
  const text = drop.querySelector(".upload-text");
  if (icon) icon.style.display = "none";
  if (text) text.style.display = "none";

  // 显示预览网格（最多显示前20张）
  previewGrid.innerHTML = "";
  previewGrid.style.display = "grid";

  const previewCount = Math.min(fileList.length, 20);
  for (let i = 0; i < previewCount; i++) {
    const file = fileList[i];
    const div = document.createElement("div");
    div.className = "frame-preview-item";

    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.alt = file.name;
    img.title = file.name;

    const label = document.createElement("span");
    label.className = "frame-label";
    label.textContent = `#${i + 1}`;

    div.appendChild(img);
    div.appendChild(label);
    previewGrid.appendChild(div);
  }

  // 如果超过20张，显示提示
  if (fileList.length > 20) {
    const moreDiv = document.createElement("div");
    moreDiv.className = "frame-preview-more";
    moreDiv.innerHTML = `<span>+${fileList.length - 20} 更多</span>`;
    previewGrid.appendChild(moreDiv);
  }
}

// 初始化多图片上传
setupFramesUpload();

async function runVideoDetect() {
  // 根据当前模式选择检测方式
  if (currentVideoInputMode === "frames") {
    return runVideoFramesDetect();
  }

  // 原始视频文件检测逻辑
  const file = document.getElementById("video-file").files[0];
  if (!file) return alert("请先上传视频文件");

  const btn = document.getElementById("video-detect-btn");
  const result = document.getElementById("video-result");
  const featureDetails = document.getElementById("feature-details");

  btn.disabled = true;
  btn.textContent = "正在提取运动特征并分析...";
  result.className = "result show";
  result.innerHTML =
    '<div style="color:#6b7280; text-align:center;">正在分析 (可能需要几十秒)...<br><small>提取 EAR 和 Jitter 特征中</small></div>';
  featureDetails.style.display = "none";

  const form = new FormData();
  form.append("video", file);
  const motionModelEl = document.getElementById("video-motion-model");
  if (motionModelEl && motionModelEl.value) {
    form.append("motion_model", motionModelEl.value);
  }

  try {
    const res = await fetch("/api/predict_video", {
      method: "POST",
      body: form,
    });
    const data = await res.json();

    if (data.error) {
      result.innerHTML = '<span class="error">错误: ' + data.error + "</span>";
    } else {
      displayVideoResult(data, result, featureDetails);
    }
  } catch (e) {
    result.innerHTML =
      '<span class="error">请求失败，请检查服务是否正常</span>';
  }

  btn.disabled = false;
  btn.textContent = "开始分析视频";
}

// 视频帧检测
async function runVideoFramesDetect() {
  const input = document.getElementById("frames-files");
  const files = input.files;

  if (!files || files.length === 0) {
    return alert("请先上传视频帧图片");
  }

  if (files.length < 10) {
    return alert(`帧数不足，当前只有 ${files.length} 帧，至少需要 10 帧`);
  }

  const btn = document.getElementById("video-detect-btn");
  const result = document.getElementById("video-result");
  const featureDetails = document.getElementById("feature-details");

  btn.disabled = true;
  btn.textContent = `正在分析 ${files.length} 帧图片...`;
  result.className = "result show";
  result.innerHTML = `<div style="color:#6b7280; text-align:center;">正在从 ${files.length} 帧图片中提取运动特征...<br><small>分析 EAR 眨眼和 Jitter 抖动</small></div>`;
  featureDetails.style.display = "none";

  // 按文件名排序
  const fileList = Array.from(files);
  fileList.sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { numeric: true })
  );

  const form = new FormData();
  // 使用 'frames' 作为字段名，按顺序添加所有帧
  fileList.forEach((file, index) => {
    form.append("frames", file);
  });
  const motionModelEl = document.getElementById("video-motion-model");
  if (motionModelEl && motionModelEl.value) {
    form.append("motion_model", motionModelEl.value);
  }

  try {
    const res = await fetch("/api/predict_video_frames", {
      method: "POST",
      body: form,
    });
    const data = await res.json();

    if (data.error) {
      result.innerHTML = '<span class="error">错误: ' + data.error + "</span>";
      if (data.note) {
        result.innerHTML += `<div style="margin-top:8px; color:#6b7280; font-size:13px;">${data.note}</div>`;
      }
    } else {
      displayVideoResult(data, result, featureDetails);
    }
  } catch (e) {
    console.error("视频帧检测请求失败:", e);
    result.innerHTML =
      '<span class="error">请求失败，请检查服务是否正常</span>';
  }

  btn.disabled = false;
  btn.textContent = "开始分析视频";
}

// 统一的结果显示函数
function displayVideoResult(data, resultEl, featureDetailsEl) {
  const fakeProb = data.fake * 100;
  const realProb = data.real * 100;
  const isFake =
    (data.label || "").toLowerCase() === "fake" ? true : data.fake > 0.5;
  const prob = isFake ? fakeProb : realProb;
  const colorClass = isFake ? "fake" : "real";
  const label = isFake ? "Fake (伪造)" : "Real (真实)";

  resultEl.innerHTML = `
        <div class="result-header">
            <span class="result-badge ${colorClass}">${
    isFake ? "DETECTED FAKE" : "LEGITIMATE VIDEO"
  }</span>
        </div>
        
        <div class="progress-container">
            <div class="progress-label">
                <span>${label}</span>
                <span>${prob.toFixed(1)}%</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill ${colorClass}" style="width: ${prob}%"></div>
            </div>
        </div>

        ${
          data.note
            ? '<div class="result-detail" style="margin-top:16px">' +
              data.note +
              "</div>"
            : ""
        }
    `;

  if (
    data.input_info &&
    (data.input_info.motion_model_file ||
      data.input_info.motion_model_id ||
      data.input_info.valid_frames != null)
  ) {
    const infoDiv = document.createElement("div");
    infoDiv.style.cssText =
      "margin-top:12px; font-size:12px; color:#6b7280; background:#f3f4f6; padding:8px 12px; border-radius:6px;";
    const parts = [];
    if (data.input_info.motion_model_id) {
      parts.push(`motion_model: <strong>${data.input_info.motion_model_id}</strong>`);
    }
    if (data.input_info.motion_model_file) {
      parts.push(
        `model_file: <strong>${data.input_info.motion_model_file}</strong>`
      );
    }
    if (
      data.input_info.valid_frames != null ||
      data.input_info.uploaded_frames != null ||
      data.input_info.sampled_frames != null ||
      data.input_info.total_frames != null
    ) {
      const vf =
        data.input_info.valid_frames != null ? data.input_info.valid_frames : "-";
      const uf =
        data.input_info.uploaded_frames != null
          ? data.input_info.uploaded_frames
          : null;
      const sf =
        data.input_info.sampled_frames != null
          ? data.input_info.sampled_frames
          : "-";
      const tf =
        data.input_info.total_frames != null ? data.input_info.total_frames : "-";
      if (uf != null) {
        parts.push(
          `frames(valid/sampled/uploaded): <strong>${vf}</strong> / ${sf} / ${uf}`
        );
      } else {
        parts.push(
          `frames(valid/sampled/total): <strong>${vf}</strong> / ${sf} / ${tf}`
        );
      }
    }
    if (data.score != null) {
      parts.push(`decision_score: <strong>${Number(data.score).toFixed(3)}</strong>`);
    }
    resultEl.appendChild(infoDiv);
    infoDiv.innerHTML = parts.join(" | ");
  }

  // 显示特征详情
  if (data.features) {
    featureDetailsEl.style.display = "block";
    document.getElementById(
      "ear-stats"
    ).textContent = `${data.features.ear_features} dim`;
    document.getElementById(
      "jitter-stats"
    ).textContent = `${data.features.jitter_features} dim`;
  }
}

// 换脸
async function runSwap() {
  const src = document.getElementById("source-file").files[0];
  const tgt = document.getElementById("target-file").files[0];
  if (!src || !tgt) return alert("请上传两张图片");

  const btn = document.getElementById("swap-btn");
  const result = document.getElementById("swap-result");
  const detectResult = document.getElementById("swap-detect-result");

  btn.disabled = true;
  btn.textContent = "换脸处理中...";
  result.className = "result show";
  result.innerHTML =
    '<div style="text-align:center; color:#6b7280;">正在生成换脸图像，请稍候...</div>';
  detectResult.className = "result detection-result"; // Hide detection result initially
  detectResult.innerHTML = "";

  const form = new FormData();
  form.append("source_image", src);
  form.append("target_image", tgt);
  form.append("backend", document.getElementById("swap-backend").value);

  let swapSuccess = false;
  let swapImageBase64 = null;

  try {
    const res = await fetch("/api/swap", { method: "POST", body: form });
    const data = await res.json();

    if (data.error) {
      result.innerHTML = '<span class="error">错误: ' + data.error + "</span>";
    } else {
      swapSuccess = true;
      swapImageBase64 = data.image;
      lastSwappedImage = data.image; // 存储到全局变量
      result.innerHTML = `
                <img src="data:image/jpeg;base64,${data.image}">
                ${
                  data.note
                    ? '<div class="result-detail" style="margin-top:12px">' +
                      data.note +
                      "</div>"
                    : ""
                }
            `;

      // 显示重新检测按钮
      document.getElementById("redetect-btn").style.display = "inline-block";
    }
  } catch (e) {
    result.innerHTML =
      '<span class="error">请求失败，请检查服务是否正常</span>';
  }

  // 如果换脸成功且开启了自动检测
  const autoDetect = document.getElementById("auto-detect-toggle");
  if (swapSuccess && autoDetect && autoDetect.checked && swapImageBase64) {
    await runAutoDetect(swapImageBase64, detectResult);
  }

  btn.disabled = false;
  btn.textContent = "开始换脸";
}

// 自动检测换脸结果
async function runAutoDetect(base64Image, resultContainer) {
  const modelId = document.getElementById("swap-detect-model").value;
  if (!modelId) {
    resultContainer.className = "result detection-result show";
    resultContainer.innerHTML = '<span class="error">未选择检测模型</span>';
    return;
  }

  resultContainer.className = "result detection-result show";
  resultContainer.innerHTML =
    '<div class="result-title">伪造检测分析</div><div style="color:#6b7280;">正在分析换脸结果...</div>';

  try {
    // 将 base64 转换为 Blob
    const byteCharacters = atob(base64Image);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: "image/jpeg" });

    const form = new FormData();
    form.append("image", blob, "swapped.jpg");
    form.append("model_name", modelId);

    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();

    if (data.error) {
      resultContainer.innerHTML =
        '<div class="result-title">伪造检测分析</div><span class="error">检测错误: ' +
        data.error +
        "</span>";
    } else {
      const fakeProb = data.fake * 100;
      const realProb = data.real * 100;
      const isFake = data.fake > 0.5;
      const prob = isFake ? fakeProb : realProb;
      const colorClass = isFake ? "fake" : "real";
      const label = isFake ? "Fake (伪造)" : "Real (真实)";

      resultContainer.innerHTML = `
                <div class="result-title">伪造检测分析</div>
                <div class="result-header">
                    <span class="result-badge ${colorClass}">${
        isFake ? "DETECTED FAKE" : "LEGITIMATE IMAGE"
      }</span>
                </div>
                
                <div class="progress-container">
                    <div class="progress-label">
                        <span>${label}</span>
                        <span>${prob.toFixed(1)}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill ${colorClass}" style="width: ${prob}%"></div>
                    </div>
                </div>
            `;
    }
  } catch (e) {
    resultContainer.innerHTML =
      '<div class="result-title">伪造检测分析</div><span class="error">检测请求失败</span>';
  }
}

// 重新检测换脸结果（使用当前选择的模型）
async function redetectSwapResult() {
  if (!lastSwappedImage) {
    alert("没有可检测的换脸结果");
    return;
  }

  const detectResult = document.getElementById("swap-detect-result");
  const btn = document.getElementById("redetect-btn");

  btn.disabled = true;
  btn.textContent = "检测中...";

  await runAutoDetect(lastSwappedImage, detectResult);

  btn.disabled = false;
  btn.textContent = "重新检测";
}

// ========== 可视化功能 ==========

// 检测时同时生成可视化
const originalRunDetect = runDetect;
runDetect = async function () {
  // 先运行原始检测
  await originalRunDetect();

  // 检测完成后生成可视化
  const file = document.getElementById("detect-file").files[0];
  const modelId = document.getElementById("model-select").value;

  if (file && modelId) {
    await generateVisualization(file, modelId);
  }
};

// 生成并显示可视化
async function generateVisualization(file, modelId) {
  const visSection = document.getElementById("visualization-section");
  const visGrid = document.getElementById("visualization-grid");
  const visDescription = document.getElementById("visualization-description");

  // 显示可视化区域
  visSection.style.display = "block";

  // 显示加载状态
  visGrid.innerHTML = `
        <div class="vis-loading">
            <div class="spinner"></div>
            <div>正在生成可视化分析...</div>
        </div>
    `;
  visDescription.innerHTML = "";

  const form = new FormData();
  form.append("image", file);
  form.append("model_name", modelId);

  try {
    const res = await fetch("/api/visualize", { method: "POST", body: form });
    const data = await res.json();

    if (data.error) {
      visGrid.innerHTML = `<div class="error">可视化生成失败: ${data.error}</div>`;
      if (data.note) {
        visGrid.innerHTML += `<div style="margin-top:8px; color:#6b7280;">${data.note}</div>`;
      }
      return;
    }

    // 显示描述
    const typeBadgeClass = data.detector_type || "unknown";
    visDescription.innerHTML = `
            <span class="vis-type-badge ${typeBadgeClass}">${
      data.detector_type || "unknown"
    }</span>
            <span>${data.description || "可视化检测依据"}</span>
        `;

    // 显示可视化结果
    const visualizations = data.visualizations || {};
    const visCount = Object.keys(visualizations).length;

    if (visCount === 0) {
      visGrid.innerHTML =
        '<div style="color:#6b7280; text-align:center; padding:20px;">暂无可视化结果</div>';
      return;
    }

    visGrid.innerHTML = "";

    for (const [name, base64Image] of Object.entries(visualizations)) {
      const visItem = document.createElement("div");
      visItem.className = "vis-item";

      const img = document.createElement("img");
      img.src = `data:image/jpeg;base64,${base64Image}`;
      img.alt = name;
      img.title = "点击查看大图";
      img.style.cursor = "pointer";

      // 点击放大
      img.onclick = () => showVisualizationModal(img.src, formatVisName(name));

      const label = document.createElement("div");
      label.className = "vis-item-label";
      label.textContent = formatVisName(name);

      visItem.appendChild(img);
      visItem.appendChild(label);
      visGrid.appendChild(visItem);
    }
  } catch (e) {
    console.error("可视化请求失败:", e);
    visGrid.innerHTML =
      '<div class="error">可视化请求失败，请检查服务状态</div>';
  }
}

// 格式化可视化名称（将下划线转为空格，首字母大写）
function formatVisName(name) {
  const nameMap = {
    grad_cam: "Grad-CAM 热力图",
    grad_cam_overlay: "Grad-CAM 叠加图",
    edge_detection: "边缘检测",
    dct_spectrum: "DCT 频谱",
    high_frequency: "高频成分",
    srm_noise_residual: "SRM 噪声残差",
    attention_map: "注意力图",
    attention_overlay: "注意力叠加图",
    spatial_edges: "空间边缘",
    texture_pattern: "纹理模式",
    srm_features: "SRM 特征",
    gradient_features: "梯度特征",
    lbp_texture: "LBP 纹理",
    edge_analysis: "边缘分析",
    sobel_magnitude: "Sobel 梯度幅值",
    canny_edges: "Canny 边缘",
    laplacian: "Laplacian 算子",
    srm_edge: "SRM 边缘",
    srm_noise: "SRM 噪声",
    srm_horizontal: "SRM 水平",
    srm_combined: "SRM 综合",
  };

  return (
    nameMap[name] ||
    name
      .split("_")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ")
  );
}

// 显示可视化放大预览
function showVisualizationModal(imageSrc, title) {
  // 创建模态框
  const modal = document.createElement("div");
  modal.className = "vis-modal";

  const img = document.createElement("img");
  img.src = imageSrc;
  img.alt = title;

  const closeBtn = document.createElement("div");
  closeBtn.className = "vis-modal-close";
  closeBtn.innerHTML = "✕";
  closeBtn.title = "关闭";

  modal.appendChild(img);
  modal.appendChild(closeBtn);
  document.body.appendChild(modal);

  // 点击任何地方关闭
  modal.onclick = () => {
    document.body.removeChild(modal);
  };

  // 阻止图片点击事件冒泡
  img.onclick = (e) => {
    e.stopPropagation();
  };
}

// 可视化开关控制
const showVisToggle = document.getElementById("show-visualization");
if (showVisToggle) {
  showVisToggle.addEventListener("change", (e) => {
    const visGrid = document.getElementById("visualization-grid");
    const visDescription = document.getElementById("visualization-description");

    if (e.target.checked) {
      visGrid.style.display = "grid";
      visDescription.style.display = "block";
    } else {
      visGrid.style.display = "none";
      visDescription.style.display = "none";
    }
  });
}

// 键盘ESC关闭模态框
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    const modal = document.querySelector(".vis-modal");
    if (modal) {
      document.body.removeChild(modal);
    }
  }
});
