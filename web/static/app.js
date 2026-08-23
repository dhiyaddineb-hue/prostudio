const $ = (id) => document.getElementById(id);

const state = {
  meta: null,
  jobId: null,
  locale: "ar",
};

function applyLocale() {
  const ar = state.locale === "ar";
  document.documentElement.lang = ar ? "ar" : "en";
  document.documentElement.dir = ar ? "rtl" : "ltr";
  $("langToggle").textContent = ar ? "EN" : "عربي";
  document.querySelectorAll("[data-ar]").forEach((el) => {
    el.textContent = ar ? el.dataset.ar : el.dataset.en;
  });
}

function logLine(message) {
  const box = $("log");
  const time = new Date().toLocaleTimeString();
  box.textContent = `[${time}] ${message}\n` + box.textContent;
}

function renderStages(active) {
  const stages = state.meta?.stages || [];
  $("stages").innerHTML = stages.map((stage) => {
    const current = active === stage.id;
    const doneOrder = ["download", "transcribe", "chunk", "translate", "tts", "mix", "render", "done"];
    const isDone = doneOrder.indexOf(active) > doneOrder.indexOf(stage.id);
    const cls = active === "error" ? "" : current ? "active" : isDone ? "done" : "";
    const label = state.locale === "ar" ? stage.ar : stage.en;
    return `<li class="${cls}" data-id="${stage.id}"><span>${label}</span><b>${isDone ? "✓" : current ? "…" : ""}</b></li>`;
  }).join("");
}

function fillLanguages() {
  const select = $("lang");
  select.innerHTML = (state.meta.languages || []).map((lang) => {
    const selected = lang.code === "ar" ? "selected" : "";
    return `<option value="${lang.code}" ${selected}>${lang.label} (${lang.code})</option>`;
  }).join("");
}

function fillVoices() {
  const lang = $("lang").value || "ar";
  const gender = $("gender").value || "male";
  const voices = state.meta.voices?.[lang]?.[gender] || [];
  const preferred = lang === "ar" && gender === "male"
    ? "ar-SA-HamedNeural"
    : lang === "ar" && gender === "female"
      ? "ar-EG-SalmaNeural"
      : voices[0]?.id;
  $("voice").innerHTML = voices.map((voice) => {
    const selected = voice.id === preferred ? "selected" : "";
    return `<option value="${voice.id}" ${selected}>${voice.label}</option>`;
  }).join("");
}

function setBusy(busy) {
  $("startBtn").disabled = busy;
}

async function loadMeta() {
  const res = await fetch("/api/meta");
  state.meta = await res.json();
  fillLanguages();
  fillVoices();
  renderStages("queued");
}

function attachEvents() {
  $("langToggle").addEventListener("click", () => {
    state.locale = state.locale === "ar" ? "en" : "ar";
    applyLocale();
    renderStages(document.querySelector(".stages li.active")?.dataset.id || "queued");
  });
  $("lang").addEventListener("change", fillVoices);
  $("gender").addEventListener("change", fillVoices);

  const drop = $("dropzone");
  const file = $("file");
  drop.addEventListener("click", () => file.click());
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.style.borderColor = "#e8b86d";
  });
  drop.addEventListener("dragleave", () => {
    drop.style.borderColor = "";
  });
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.style.borderColor = "";
    if (e.dataTransfer.files[0]) {
      file.files = e.dataTransfer.files;
      $("fileName").textContent = e.dataTransfer.files[0].name;
    }
  });
  file.addEventListener("change", () => {
    $("fileName").textContent = file.files[0]?.name || $("fileName").dataset.ar;
  });

  $("dubForm").addEventListener("submit", onSubmit);
}

async function onSubmit(event) {
  event.preventDefault();
  const data = new FormData($("dubForm"));
  if (!data.get("url") && !data.get("file")?.name) {
    $("statusLine").textContent = "أدخل رابط يوتيوب أو ارفع ملف فيديو";
    return;
  }
  data.set("bg_music", $("bg_music").checked ? "true" : "false");
  setBusy(true);
  $("result").classList.add("hidden");
  $("statusLine").textContent = "جاري إنشاء المهمة…";
  logLine("بدء مهمة دبلجة جديدة");

  const res = await fetch("/api/jobs", { method: "POST", body: data });
  const job = await res.json();
  if (!res.ok) {
    setBusy(false);
    $("statusLine").textContent = job.detail || "تعذر إنشاء المهمة";
    return;
  }
  state.jobId = job.id;
  listen(job.id);
}

function listen(jobId) {
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  source.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.percent != null) $("percent").textContent = `${payload.percent}%`;
    if (payload.stage) renderStages(payload.stage === "done" ? "render" : payload.stage);
    if (payload.message) {
      $("statusLine").textContent = payload.message;
      logLine(payload.message);
    }
    if (payload.type === "done") {
      source.close();
      setBusy(false);
      showResult(jobId, payload.output_name);
    }
    if (payload.type === "error") {
      source.close();
      setBusy(false);
      renderStages("error");
      $("statusLine").textContent = payload.message || "فشلت المهمة";
    }
  };
  source.onerror = () => {
    // Keep polling the job if the stream drops.
    poll(jobId);
  };
}

async function poll(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) return;
  const job = await res.json();
  if (job.percent != null) $("percent").textContent = `${job.percent}%`;
  if (job.message) $("statusLine").textContent = job.message;
  if (job.stage) renderStages(job.stage === "done" ? "render" : job.stage);
  if (job.status === "done") {
    setBusy(false);
    showResult(jobId, job.output_name);
  } else if (job.status === "error") {
    setBusy(false);
  } else {
    setTimeout(() => poll(jobId), 2500);
  }
}

function showResult(jobId, name) {
  $("result").classList.remove("hidden");
  $("downloadBtn").href = `/api/jobs/${jobId}/download`;
  $("downloadBtn").download = name || "output.mp4";
  $("player").src = `/api/jobs/${jobId}/preview`;
  $("srtBtn").href = `/api/jobs/${jobId}/srt`;
  $("srtBtn").classList.remove("hidden");
  $("statusLine").textContent = "الفيديو جاهز للتحميل";
}

loadMeta().then(attachEvents).catch((err) => {
  $("statusLine").textContent = "تعذر تحميل إعدادات الاستوديو";
  logLine(String(err));
});
