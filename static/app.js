(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  let uploadedVideoPath = null;
  let pollTimer = null;
  let logsSeen = 0;

  // ---------------------------------------------------------------------
  // Radio toggles
  // ---------------------------------------------------------------------
  function wireRadioToggle(groupName, valueToPanel) {
    document.querySelectorAll(`input[name="${groupName}"]`).forEach((input) => {
      input.addEventListener("change", () => {
        Object.entries(valueToPanel).forEach(([value, el]) => {
          if (!el) return;
          el.classList.toggle("hidden", input.value !== value && value !== "__default__");
        });
        const panel = valueToPanel[input.value];
        if (panel) panel.classList.remove("hidden");
      });
    });
  }

  wireRadioToggle("execution_mode", {
    remote: $("remote-config"),
    local: null,
  });
  document.querySelectorAll('input[name="execution_mode"]').forEach((input) => {
    input.addEventListener("change", () => {
      $("remote-config").classList.toggle("hidden", input.value !== "remote" || !input.checked);
    });
  });

  document.querySelectorAll('input[name="video_source"]').forEach((input) => {
    input.addEventListener("change", () => {
      const isYoutube = document.querySelector('input[name="video_source"]:checked').value === "youtube";
      $("local-video-block").classList.toggle("hidden", isYoutube);
      $("youtube-block").classList.toggle("hidden", !isYoutube);
    });
  });

  // ---------------------------------------------------------------------
  // System info / environment status
  // ---------------------------------------------------------------------
  function statusClass(level) {
    return level === "good" ? "status-good" : level === "warn" ? "status-warn" : "status-bad";
  }

  async function loadSystemInfo() {
    try {
      const res = await fetch("/api/system-info");
      const data = await res.json();

      const sysRows = $("system-info").querySelectorAll(".kv-value");
      sysRows[0].textContent = `${data.ram_gb} GB`;
      sysRows[1].textContent = data.gpu_name;
      sysRows[2].textContent = `${data.chunk_duration}s (${Math.round(data.chunk_duration / 60)} min)`;

      const envRows = $("env-status").querySelectorAll(".kv-value");
      envRows[0].textContent = data.hf_token ? "✓ Found" : "✗ Missing";
      envRows[0].className = "kv-value " + (data.hf_token ? "status-good" : "status-bad");

      envRows[1].textContent = data.groq_token ? "✓ Found" : "✗ Missing";
      envRows[1].className = "kv-value " + (data.groq_token ? "status-good" : "status-warn");

      envRows[2].textContent = data.cloning_engine.label;
      envRows[2].className = "kv-value " + statusClass(data.cloning_engine.level);
    } catch (e) {
      console.error("Failed to load system info", e);
    }
  }

  // ---------------------------------------------------------------------
  // Remote connection test
  // ---------------------------------------------------------------------
  $("test-connection-btn").addEventListener("click", async () => {
    const url = $("remote-url").value.trim();
    const statusEl = $("remote-status");
    if (!url) {
      statusEl.textContent = "Please paste a Cloud GPU tunnel URL first.";
      statusEl.className = "hint status-bad";
      return;
    }
    statusEl.textContent = "Connecting to Cloud GPU server…";
    statusEl.className = "hint";
    try {
      const res = await fetch("/api/test-connection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      if (data.status === "healthy") {
        const gpu = data.gpu || {};
        statusEl.textContent = `✓ Connected: ${gpu.gpu_name || "GPU"} (${gpu.vram_gb || 0} GB VRAM)`;
        statusEl.className = "hint status-good";
      } else {
        statusEl.textContent = `✗ Connection failed: ${data.message || "Unknown error"}`;
        statusEl.className = "hint status-bad";
      }
    } catch (e) {
      statusEl.textContent = `✗ Error: ${e}`;
      statusEl.className = "hint status-bad";
    }
  });

  // ---------------------------------------------------------------------
  // File upload
  // ---------------------------------------------------------------------
  const dropzone = $("upload-drop");
  const fileInput = $("video-file-input");

  dropzone.addEventListener("click", () => fileInput.click());
  ["dragover", "dragenter"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) uploadFile(fileInput.files[0]);
  });

  async function uploadFile(file) {
    const statusEl = $("upload-status");
    $("upload-label").textContent = `Uploading ${file.name}…`;
    statusEl.textContent = "";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");

      uploadedVideoPath = data.path;
      $("upload-label").textContent = `✓ ${data.filename}`;
      statusEl.textContent = "Ready to process.";
      statusEl.className = "hint status-good";
    } catch (e) {
      $("upload-label").textContent = "Click to choose a video file, or drag one here";
      statusEl.textContent = `✗ ${e.message}`;
      statusEl.className = "hint status-bad";
      uploadedVideoPath = null;
    }
  }

  // ---------------------------------------------------------------------
  // Start processing
  // ---------------------------------------------------------------------
  $("start-btn").addEventListener("click", async () => {
    const executionMode = document.querySelector('input[name="execution_mode"]:checked').value;
    const videoSource = document.querySelector('input[name="video_source"]:checked').value;
    const isYoutube = videoSource === "youtube";

    if (executionMode === "remote" && !$("remote-url").value.trim()) {
      alert("Please enter a Cloud GPU tunnel URL.");
      return;
    }
    if (isYoutube && !$("youtube-url").value.trim()) {
      alert("Please enter a YouTube URL.");
      return;
    }
    if (!isYoutube && !uploadedVideoPath) {
      alert("Please upload a video file first.");
      return;
    }

    const payload = {
      execution_mode: executionMode,
      remote_url: $("remote-url").value.trim(),
      video_source: videoSource,
      video_path: isYoutube ? $("youtube-url").value.trim() : uploadedVideoPath,
      source_lang: $("source-lang").value,
      target_lang: $("target-lang").value,
      whisper_model: $("whisper-model").value,
      voice_quality: $("voice-quality").value,
      enable_lipsync: $("enable-lipsync").checked,
      preserve_bg: $("preserve-bg").checked,
      use_context: $("use-context").checked,
    };

    $("start-btn").disabled = true;
    $("start-btn").textContent = "Processing…";
    $("progress-card").classList.remove("hidden");
    $("result-block").classList.add("hidden");
    $("log-view").textContent = "";
    $("status-text").textContent = "Starting…";
    $("progress-fill").classList.add("indeterminate");
    logsSeen = 0;

    try {
      const res = await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to start");

      startPolling();
    } catch (e) {
      $("status-text").textContent = `✗ ${e.message}`;
      resetStartButton();
    }
  });

  function resetStartButton() {
    $("start-btn").disabled = false;
    $("start-btn").textContent = "▶ Start Processing";
    $("progress-fill").classList.remove("indeterminate");
  }

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, 1000);
    pollStatus();
  }

  async function pollStatus() {
    try {
      const res = await fetch(`/api/logs?since=${logsSeen}`);
      const data = await res.json();

      if (data.logs.length) {
        const logView = $("log-view");
        logView.textContent += data.logs.join("\n") + "\n";
        logView.scrollTop = logView.scrollHeight;
        logsSeen = data.total;
      }

      if (data.status === "processing") {
        $("status-text").textContent = "Processing…";
      } else if (data.status === "completed") {
        $("status-text").textContent = "✓ Completed!";
        clearInterval(pollTimer);
        resetStartButton();
        $("progress-fill").style.width = "100%";
        showResult(data);
      } else if (data.status === "failed") {
        $("status-text").textContent = `✗ Failed: ${data.error || "See log above"}`;
        clearInterval(pollTimer);
        resetStartButton();
      }
    } catch (e) {
      console.error("Polling error", e);
    }
  }

  function showResult(data) {
    if (!data.result_video) return;
    const block = $("result-block");
    block.classList.remove("hidden");
    const src = `/results/${encodeURIComponent(data.result_video)}`;
    $("result-video").src = src;
    $("download-link").href = src;
    $("download-link").setAttribute("download", data.result_video);
  }

  // ---------------------------------------------------------------------
  loadSystemInfo();
})();
