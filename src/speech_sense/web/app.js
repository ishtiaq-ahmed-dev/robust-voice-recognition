/* ═══════════════════════════════════════════════════════════════════════
   SpeechSense Dashboard — client logic
   Engineered by Ishtiaq Ahmed

   Four things here are deliberate rather than incidental:

   1. Nothing untrusted is ever concatenated into HTML. Speaker names come
      from enrolment forms and from directory names on disk, so building
      markup out of them is a stored-XSS hole. Every dynamic value goes in
      through `textContent` or an element property.

   2. Recording releases what it acquires. One AudioContext is created lazily
      and reused for the whole session (browsers cap concurrent contexts at a
      handful — a context per take breaks recording after ~6), and the
      microphone tracks are stopped on every path out of a recording, so the
      browser's "in use" indicator actually goes away.

   3. Audio is converted to real PCM WAV in the browser. MediaRecorder emits
      WebM/Opus regardless of the MIME type you ask for; sending that with a
      .wav name made the server fall back to librosa, which needs ffmpeg
      installed. Decoding here means the server only ever sees a format
      libsndfile handles natively.

   4. Every request is bounded and checked. Each fetch has an abort timeout
      and its status is inspected — a hung or failing backend surfaces as a
      toast, not as a spinner that never resolves.
   ═══════════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  var REQUEST_TIMEOUT_MS = 120000;
  var ANALYTICS_INTERVAL_MS = 8000;
  var HEALTH_INTERVAL_MS = 15000;
  var ENROLL_TAKES = 3;
  var AVATAR_CLASSES = ['av-1', 'av-2', 'av-3', 'av-4', 'av-5', 'av-6'];
  var API_KEY_STORAGE = 'speechsense.apiKey';

  var state = {
    prompts: [],
    promptIdx: 0,
    mode: 'identify',
    recorder: null,
    stream: null,
    chunks: [],
    pendingBlob: null,
    isRecording: false,
    enrollClips: [],
    audioCtx: null,
    analyser: null,
    animId: null,
    clockOffset: 0,        // serverTime - clientTime, so "x ago" never goes negative
    analyticsFailures: 0,
    apiKey: ''
  };

  // ── DOM helpers ────────────────────────────────────────────────────────

  function $(id) { return document.getElementById(id); }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function setText(id, value) {
    var node = $(id);
    if (node) { node.textContent = value; }
  }

  function replaceChildren(node, children) {
    node.textContent = '';
    for (var i = 0; i < children.length; i++) { node.appendChild(children[i]); }
  }

  function avatarFor(name, index) {
    var initial = (name || '?').trim().charAt(0).toUpperCase() || '?';
    return el('span', 'avatar ' + AVATAR_CLASSES[index % AVATAR_CLASSES.length], initial);
  }

  function toast(message, type) {
    var box = $('toasts');
    var node = el('div', 'toast ' + (type || 'info'), message);
    box.appendChild(node);
    setTimeout(function () {
      node.style.opacity = '0';
      setTimeout(function () { node.remove(); }, 300);
    }, 4500);
  }

  // ── HTTP ───────────────────────────────────────────────────────────────

  /**
   * fetch with an abort timeout, the optional API key, and status checking.
   * Resolves to the parsed body; rejects with an Error carrying a message
   * suitable for a toast.
   */
  function api(path, options) {
    options = options || {};
    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, options.timeout || REQUEST_TIMEOUT_MS);

    var headers = options.headers || {};
    if (state.apiKey) { headers['X-API-Key'] = state.apiKey; }

    return fetch(path, {
      method: options.method || 'GET',
      body: options.body,
      headers: headers,
      signal: controller.signal
    }).then(function (response) {
      return response.text().then(function (raw) {
        var body = null;
        try { body = raw ? JSON.parse(raw) : null; } catch (e) { body = null; }
        if (!response.ok) {
          var detail = (body && body.detail) ? body.detail : (raw || response.statusText);
          if (typeof detail !== 'string') { detail = JSON.stringify(detail); }
          if (response.status === 401) {
            detail = 'This server requires an API key. Click the 🔑 button to set one.';
          }
          var err = new Error(detail);
          err.status = response.status;
          throw err;
        }
        return body;
      });
    }).catch(function (error) {
      if (error.name === 'AbortError') { throw new Error('Request timed out.'); }
      if (error instanceof TypeError) { throw new Error('Cannot reach the server.'); }
      throw error;
    }).finally(function () {
      clearTimeout(timer);
    });
  }

  // ── Audio conversion ───────────────────────────────────────────────────

  /** Encode mono float samples as a 16-bit PCM WAV blob. */
  function encodeWav(samples, sampleRate) {
    var buffer = new ArrayBuffer(44 + samples.length * 2);
    var view = new DataView(buffer);
    function writeAscii(offset, text) {
      for (var i = 0; i < text.length; i++) { view.setUint8(offset + i, text.charCodeAt(i)); }
    }
    writeAscii(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeAscii(8, 'WAVE');
    writeAscii(12, 'fmt ');
    view.setUint32(16, 16, true);            // PCM header size
    view.setUint16(20, 1, true);             // format = PCM
    view.setUint16(22, 1, true);             // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);             // block align
    view.setUint16(34, 16, true);            // bits per sample
    writeAscii(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    var offset = 44;
    for (var i = 0; i < samples.length; i++, offset += 2) {
      var s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([view], { type: 'audio/wav' });
  }

  /**
   * Decode any blob the browser understands and re-encode it as PCM WAV.
   * Returns the original blob unchanged if decoding fails, so the server's
   * own decoders still get a chance.
   */
  function toWavBlob(blob) {
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx || !blob) { return Promise.resolve(blob); }
    var ctx = new Ctx();
    return blob.arrayBuffer()
      .then(function (bytes) { return ctx.decodeAudioData(bytes); })
      .then(function (decoded) {
        var length = decoded.length;
        var mono = new Float32Array(length);
        for (var c = 0; c < decoded.numberOfChannels; c++) {
          var data = decoded.getChannelData(c);
          for (var i = 0; i < length; i++) { mono[i] += data[i]; }
        }
        if (decoded.numberOfChannels > 1) {
          for (var j = 0; j < length; j++) { mono[j] /= decoded.numberOfChannels; }
        }
        return encodeWav(mono, decoded.sampleRate);
      })
      .catch(function (error) {
        console.warn('Client-side decode failed; sending the original file.', error);
        return blob;
      })
      .finally(function () { ctx.close().catch(function () {}); });
  }

  // ── Health & analytics ─────────────────────────────────────────────────

  /**
   * `force` distinguishes an explicit refresh from a timer tick. Skipping
   * timer ticks in a hidden tab is polite; skipping the *first* load because
   * the dashboard happened to open in a background tab just leaves it showing
   * placeholder dashes.
   */
  function pollHealth(force) {
    if (document.hidden && !force) { return; }
    api('/healthz', { timeout: 8000 }).then(function (data) {
      var backend = (data && data.backend) ? String(data.backend).toUpperCase() : 'UNKNOWN';
      setText('statusText', backend + ' Engine Online');
      $('statusPill').classList.remove('is-down');
    }).catch(function (error) {
      setText('statusText', error.status === 503 ? 'Engine Starting…' : 'Engine Offline');
      $('statusPill').classList.add('is-down');
    });
  }

  function loadAnalytics(force) {
    if (document.hidden && !force) { return Promise.resolve(); }
    // Back off once the backend has failed repeatedly so a dashboard left open
    // against a dead server doesn't hammer it (or the console) forever.
    if (!force && state.analyticsFailures > 3 && (state.analyticsFailures % 5) !== 0) {
      state.analyticsFailures++;
      return Promise.resolve();
    }
    return api('/analytics').then(function (d) {
      state.analyticsFailures = 0;
      if (typeof d.server_time === 'number') {
        state.clockOffset = d.server_time - (Date.now() / 1000);
      }

      setText('statSpeakers', d.speakers_enrolled);
      setText('statSpeakersChange', d.speakers_enrolled > 0
        ? d.speakers_enrolled + ' voice profiles active'
        : 'No profiles enrolled');
      setText('statOperations', d.total_operations);
      setText('statMatchRate', d.total_operations > 0 ? d.match_rate + '%' : '—');
      setText('statMatchRateLabel', d.successful_matches + ' matched / ' + d.failed_matches + ' failed');
      setText('statConfidence', d.avg_confidence > 0 ? Math.round(d.avg_confidence * 100) + '%' : '—');
      setText('statBackend', 'Backend: ' + (d.backend || 'unknown'));

      if (d.config) {
        setText('cfgBackend', d.backend || '—');
        setText('cfgSampleRate', d.config.sample_rate + ' Hz');
        setText('cfgThreshold', d.config.similarity_threshold);
        setText('cfgMargin', d.config.known_speaker_margin);
        setText('cfgVAD', d.config.vad_enabled ? 'Active' : 'Disabled');
        setText('cfgUptime', formatUptime(d.uptime_seconds));
      }

      renderSpeakers(d.speaker_names || []);
      renderFeed(d.recent_events || []);
    }).catch(function (error) {
      state.analyticsFailures++;
      if (state.analyticsFailures === 1) { toast('Analytics unavailable: ' + error.message, 'error'); }
    });
  }

  function formatUptime(seconds) {
    var total = Math.max(0, Math.floor(seconds || 0));
    var days = Math.floor(total / 86400);
    var hours = Math.floor((total % 86400) / 3600);
    var mins = Math.floor((total % 3600) / 60);
    if (days) { return days + 'd ' + hours + 'h'; }
    if (hours) { return hours + 'h ' + mins + 'm'; }
    return mins + 'm';
  }

  function timeSince(timestamp) {
    // Client and server clocks disagree; without the measured offset a fresh
    // event renders as "-3s ago".
    var seconds = Math.max(0, Math.floor((Date.now() / 1000 + state.clockOffset) - timestamp));
    if (seconds < 60) { return seconds + 's ago'; }
    if (seconds < 3600) { return Math.floor(seconds / 60) + 'm ago'; }
    return Math.floor(seconds / 3600) + 'h ago';
  }

  function renderSpeakers(names) {
    var host = $('speakersList');
    if (!names.length) {
      replaceChildren(host, [el('p', 'empty', 'No enrolled voice profiles.')]);
      return;
    }
    var grid = el('div', 'speaker-grid');
    names.forEach(function (name, index) {
      var chip = el('span', 'speaker-chip');
      chip.appendChild(avatarFor(name, index));
      chip.appendChild(el('span', 'chip-name', name));

      var remove = el('button', 'del-btn', '✕');
      remove.type = 'button';
      remove.title = 'Delete ' + name;
      remove.setAttribute('aria-label', 'Delete ' + name);
      // A listener, not an onclick attribute: a speaker called
      // `'); fetch(...)` would otherwise become executable markup.
      remove.addEventListener('click', function () { deleteSpeaker(name); });

      chip.appendChild(remove);
      grid.appendChild(chip);
    });
    replaceChildren(host, [grid]);
  }

  function renderFeed(events) {
    setText('feedCount', events.length + (events.length === 1 ? ' event' : ' events'));
    var host = $('activityFeed');
    if (!events.length) {
      replaceChildren(host, [el('p', 'empty', 'No activity yet.')]);
      return;
    }
    replaceChildren(host, events.map(function (event) {
      var item = el('div', 'feed-item');
      item.appendChild(el('span', 'feed-dot' + (event.success ? '' : ' is-fail')));
      var body = el('div');
      body.appendChild(el('div', 'feed-details', event.details));
      body.appendChild(el('div', 'feed-time', timeSince(event.timestamp)));
      item.appendChild(body);
      return item;
    }));
  }

  function deleteSpeaker(name) {
    if (!window.confirm('Delete speaker "' + name + '"?')) { return; }
    api('/speakers/' + encodeURIComponent(name), { method: 'DELETE' })
      .then(function () {
        toast('Deleted ' + name, 'success');
        return loadAnalytics(true);
      })
      .catch(function (error) { toast('Delete failed: ' + error.message, 'error'); });
  }

  // ── Prompts ────────────────────────────────────────────────────────────

  function loadPrompts() {
    return api('/prompts').then(function (data) {
      state.prompts = (data && data.prompts) || [];
      if (state.prompts.length) { renderPrompt(0); }
    }).catch(function () {
      setText('promptText', 'Prompts unavailable.');
    });
  }

  function renderPrompt(index) {
    if (!state.prompts.length) { return; }
    state.promptIdx = ((index % state.prompts.length) + state.prompts.length) % state.prompts.length;
    var prompt = state.prompts[state.promptIdx];
    setText('promptCat', prompt.category);
    setText('promptText', '“' + prompt.text + '”');
    setText('promptCounter', (state.promptIdx + 1) + ' / ' + state.prompts.length);
  }

  function copyPrompt() {
    if (!state.prompts.length) { return; }
    var text = state.prompts[state.promptIdx].text;
    // navigator.clipboard is undefined outside a secure context, which is
    // exactly where this dashboard often runs (plain http on a LAN address).
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text)
        .then(function () { toast('Sentence copied to clipboard', 'success'); })
        .catch(function () { toast('Clipboard blocked by the browser', 'error'); });
    } else {
      toast('Clipboard needs HTTPS — select the sentence and copy it manually', 'info');
    }
  }

  // ── Studio ─────────────────────────────────────────────────────────────

  function setMode(mode) {
    state.mode = mode;
    var enrolling = mode === 'enroll';
    $('modeEnroll').classList.toggle('is-active', enrolling);
    $('modeIdentify').classList.toggle('is-active', !enrolling);
    $('enrollNameGroup').classList.toggle('is-hidden', !enrolling);
    $('takesBar').classList.toggle('is-hidden', !enrolling);
    setText('processBtn', enrolling ? 'Capture Take' : 'Analyze Audio');
    state.enrollClips = [];
    updatePips();
  }

  function updatePips() {
    var pips = document.querySelectorAll('#takesBar .take-pip');
    for (var i = 0; i < pips.length; i++) {
      pips[i].classList.toggle('is-filled', i < state.enrollClips.length);
    }
  }

  function getAudioContext() {
    // One context for the whole session. Browsers allow only a handful at a
    // time, so creating one per take silently kills recording after ~6 takes.
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) { return null; }
    if (!state.audioCtx || state.audioCtx.state === 'closed') { state.audioCtx = new Ctx(); }
    if (state.audioCtx.state === 'suspended') { state.audioCtx.resume().catch(function () {}); }
    return state.audioCtx;
  }

  function releaseMicrophone() {
    if (state.stream) {
      state.stream.getTracks().forEach(function (track) { track.stop(); });
      state.stream = null;
    }
    if (state.animId) {
      cancelAnimationFrame(state.animId);
      state.animId = null;
    }
    if (state.analyser) {
      try { state.analyser.disconnect(); } catch (e) { /* already detached */ }
      state.analyser = null;
    }
  }

  function startRecord() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast('Microphone access needs HTTPS (or localhost). Upload a file instead.', 'error');
      return;
    }
    if (typeof MediaRecorder === 'undefined') {
      toast('This browser has no MediaRecorder. Upload a file instead.', 'error');
      return;
    }

    navigator.mediaDevices.getUserMedia({ audio: true }).then(function (stream) {
      state.stream = stream;
      state.chunks = [];
      state.recorder = new MediaRecorder(stream);

      var ctx = getAudioContext();
      if (ctx) {
        state.analyser = ctx.createAnalyser();
        state.analyser.fftSize = 256;
        ctx.createMediaStreamSource(stream).connect(state.analyser);
      }

      state.recorder.ondataavailable = function (event) {
        if (event.data && event.data.size) { state.chunks.push(event.data); }
      };
      state.recorder.onerror = function (event) {
        toast('Recording error: ' + (event.error ? event.error.name : 'unknown'), 'error');
        finishRecording(null);
      };
      state.recorder.onstop = function () {
        var blob = state.chunks.length
          ? new Blob(state.chunks, { type: state.recorder.mimeType || 'audio/webm' })
          : null;
        finishRecording(blob);
      };

      state.recorder.start();
      state.isRecording = true;
      $('recBtn').classList.add('is-recording');
      setText('recLabel', 'Stop');
      setText('vizOverlay', '');
      drawViz();
    }).catch(function (error) {
      var message = error && error.name === 'NotAllowedError'
        ? 'Microphone permission denied'
        : 'Could not open the microphone: ' + (error && error.message ? error.message : error);
      toast(message, 'error');
      releaseMicrophone();
    });
  }

  function finishRecording(blob) {
    state.isRecording = false;
    releaseMicrophone();
    $('recBtn').classList.remove('is-recording');
    setText('recLabel', 'Record');

    if (!blob) {
      setText('vizOverlay', 'No audio captured — try again');
      return;
    }
    setText('vizOverlay', 'Converting…');
    toWavBlob(blob).then(function (wav) {
      state.pendingBlob = wav;
      $('processBtn').disabled = false;
      setText('vizOverlay', '✓ Audio captured');
      toast('Voice sample recorded', 'success');
    });
  }

  function stopRecord() {
    if (state.recorder && state.isRecording) {
      try {
        state.recorder.stop();   // onstop drives the rest
      } catch (error) {
        finishRecording(null);
      }
    }
  }

  function drawViz() {
    var canvas = $('vizCanvas');
    if (!state.analyser || !canvas) { return; }
    var ctx2d = canvas.getContext('2d');
    var bins = state.analyser.frequencyBinCount;
    var data = new Uint8Array(bins);

    function render() {
      if (!state.isRecording || !state.analyser) { return; }
      state.animId = requestAnimationFrame(render);

      var width = canvas.clientWidth;
      var height = canvas.clientHeight;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      state.analyser.getByteFrequencyData(data);
      ctx2d.clearRect(0, 0, width, height);
      var gradient = ctx2d.createLinearGradient(0, height, 0, 0);
      gradient.addColorStop(0, 'rgba(34, 211, 238, 0.25)');
      gradient.addColorStop(1, 'rgba(192, 132, 252, 0.9)');
      ctx2d.fillStyle = gradient;

      var barWidth = (width / bins) * 2.5;
      var x = 0;
      for (var i = 0; i < bins && x < width; i++) {
        var barHeight = (data[i] / 255) * height;
        ctx2d.fillRect(x, height - barHeight, barWidth, barHeight);
        x += barWidth + 1;
      }
    }
    render();
  }

  function acceptFile(file) {
    if (!file) { return; }
    setText('vizOverlay', 'Converting ' + file.name + '…');
    toWavBlob(file).then(function (wav) {
      state.pendingBlob = wav;
      $('processBtn').disabled = false;
      setText('vizOverlay', '📄 ' + file.name);
      toast('File loaded: ' + file.name, 'info');
    });
  }

  function processAudio() {
    if (!state.pendingBlob) { toast('No audio sample provided', 'error'); return; }
    return state.mode === 'enroll' ? captureEnrollTake() : runIdentify();
  }

  function captureEnrollTake() {
    var name = $('enrollName').value.trim();
    if (!name) { toast('Enter a speaker name', 'error'); return Promise.resolve(); }

    state.enrollClips.push(state.pendingBlob);
    state.pendingBlob = null;
    updatePips();
    $('processBtn').disabled = true;
    setText('vizOverlay', 'Press record and speak clearly');

    if (state.enrollClips.length < ENROLL_TAKES) {
      var left = ENROLL_TAKES - state.enrollClips.length;
      toast('Take ' + state.enrollClips.length + '/' + ENROLL_TAKES + ' captured. Record ' + left + ' more.', 'info');
      renderPrompt(state.promptIdx + 1);
      return Promise.resolve();
    }

    var form = new FormData();
    form.append('name', name);
    state.enrollClips.forEach(function (clip, i) {
      form.append('files', clip, 'take_' + (i + 1) + '.wav');
    });

    setText('processBtn', 'Enrolling…');
    return api('/enroll', { method: 'POST', body: form })
      .then(function () {
        toast('Enrolled: ' + name, 'success');
        state.enrollClips = [];
        updatePips();
        return loadAnalytics(true);
      })
      .catch(function (error) {
        // Keep the takes: making the user re-record three clips because the
        // network blipped is the wrong recovery.
        toast('Enrollment failed: ' + error.message, 'error');
      })
      .finally(function () { setText('processBtn', 'Capture Take'); });
  }

  function runIdentify() {
    var form = new FormData();
    form.append('file', state.pendingBlob, 'input.wav');
    setText('processBtn', 'Analyzing…');
    return api('/identify', { method: 'POST', body: form })
      .then(function (result) {
        renderResult(result);
        return loadAnalytics(true);
      })
      .catch(function (error) { toast('Analysis failed: ' + error.message, 'error'); })
      .finally(function () { setText('processBtn', 'Analyze Audio'); });
  }

  function renderResult(result) {
    var badge = $('resultBadge');
    badge.className = 'result-badge ' + (result.is_known ? 'is-match' : 'is-nomatch');
    badge.textContent = result.is_known ? '✓ VERIFIED: ' + result.speaker : '✗ UNKNOWN VOICE';

    var confidence = Math.round((result.confidence || 0) * 100);
    setText('resultConf', confidence + '%');
    $('confBar').style.width = confidence + '%';
    $('confBar').style.background = result.is_known ? 'var(--green)' : 'var(--red)';
    setText('resSim', (result.similarity || 0).toFixed(4));
    setText('resMargin', (result.margin || 0).toFixed(4));
    setText('resReason', result.reason && result.reason !== 'ok' ? result.reason : '');

    var scores = result.scores || {};
    var ranked = Object.keys(scores).map(function (name) {
      return { name: name, score: scores[name] };
    }).sort(function (a, b) { return b.score - a.score; });

    replaceChildren($('scoresList'), ranked.map(function (entry, index) {
      var isWinner = result.is_known && entry.name === result.speaker;
      var row = el('div', 'score-row' + (isWinner ? ' is-winner' : ''));
      row.appendChild(el('span', 'score-rank', '#' + (index + 1)));
      row.appendChild(el('span', 'score-name', entry.name));
      row.appendChild(el('span', 'score-val', entry.score.toFixed(4)));
      return row;
    }));
  }

  function resetStudio() {
    stopRecord();
    state.pendingBlob = null;
    state.enrollClips = [];
    updatePips();
    $('processBtn').disabled = true;
    setText('vizOverlay', 'Press record and speak clearly');
    var badge = $('resultBadge');
    badge.className = 'result-badge is-idle';
    badge.textContent = 'Awaiting Audio Input';
    setText('resultConf', '—');
    $('confBar').style.width = '0';
    setText('resSim', '—');
    setText('resMargin', '—');
    setText('resReason', '');
    $('scoresList').textContent = '';
    toast('Studio reset', 'info');
  }

  // ── Employees ──────────────────────────────────────────────────────────

  function loadEmployees() {
    var host = $('empList');
    return api('/employees').then(function (data) {
      if (!data.exists || !data.employees.length) {
        replaceChildren(host, [el('p', 'empty',
          data.exists ? 'No employee folders found in ' + data.root
                      : 'Employee directory not found at ' + data.root)]);
        return;
      }
      replaceChildren(host, data.employees.map(function (employee, index) {
        var card = el('div', 'emp-card');
        var name = el('div', 'emp-name');
        name.appendChild(avatarFor(employee.name, index));
        name.appendChild(el('span', null, employee.name.replace(/_/g, ' ')));
        card.appendChild(name);
        card.appendChild(el('span', 'emp-clips', employee.clips + ' audio clips'));
        return card;
      }));
    }).catch(function (error) {
      replaceChildren(host, [el('p', 'empty', 'Could not load employees: ' + error.message)]);
    });
  }

  function enrollAllEmployees() {
    var button = $('enrollAllBtn');
    button.disabled = true;
    toast('Batch enrolling employees…', 'info');
    // Whole-corpus enrolment is minutes of CPU, well past the default timeout.
    return api('/employees/enroll-all', { method: 'POST', timeout: 1800000 })
      .then(function (report) {
        var count = report && report.enrolled ? Object.keys(report.enrolled).length : 0;
        toast('Enrolled ' + count + ' employee' + (count === 1 ? '' : 's'), 'success');
        return loadAnalytics(true);
      })
      .catch(function (error) { toast('Batch enrollment failed: ' + error.message, 'error'); })
      .finally(function () { button.disabled = false; });
  }

  // ── API key ────────────────────────────────────────────────────────────

  function loadApiKey() {
    try { state.apiKey = window.localStorage.getItem(API_KEY_STORAGE) || ''; }
    catch (e) { state.apiKey = ''; }   // private mode / storage disabled
    setText('apiKeyState', state.apiKey ? 'Key set' : 'No key');
  }

  function promptForApiKey() {
    var next = window.prompt('X-API-Key sent with every request (leave blank to clear):', state.apiKey);
    if (next === null) { return; }
    state.apiKey = next.trim();
    try {
      if (state.apiKey) { window.localStorage.setItem(API_KEY_STORAGE, state.apiKey); }
      else { window.localStorage.removeItem(API_KEY_STORAGE); }
    } catch (e) { /* not persisted, but still used for this session */ }
    setText('apiKeyState', state.apiKey ? 'Key set' : 'No key');
    state.analyticsFailures = 0;   // a new key deserves a fresh attempt
    pollHealth(true);
    loadAnalytics(true);
  }

  // ── Navigation ─────────────────────────────────────────────────────────

  function showPage(name) {
    document.querySelectorAll('.page').forEach(function (page) {
      page.classList.toggle('is-active', page.id === 'page-' + name);
    });
    document.querySelectorAll('.nav-link').forEach(function (link) {
      var active = link.dataset.page === name;
      link.classList.toggle('is-active', active);
      link.setAttribute('aria-selected', active ? 'true' : 'false');
    });
  }

  // ── Wiring ─────────────────────────────────────────────────────────────

  function init() {
    loadApiKey();

    document.querySelectorAll('.nav-link').forEach(function (link) {
      link.addEventListener('click', function () { showPage(link.dataset.page); });
    });

    $('apiKeyBtn').addEventListener('click', promptForApiKey);
    $('refreshSpeakersBtn').addEventListener('click', function () {
      loadAnalytics(true).then(function () { toast('Profiles refreshed', 'info'); });
    });
    $('reloadDbBtn').addEventListener('click', function () {
      api('/reload', { method: 'POST' })
        .then(function () { toast('Database reloaded', 'success'); return loadAnalytics(); })
        .catch(function (error) { toast('Reload failed: ' + error.message, 'error'); });
    });

    $('prevPromptBtn').addEventListener('click', function () { renderPrompt(state.promptIdx - 1); });
    $('nextPromptBtn').addEventListener('click', function () { renderPrompt(state.promptIdx + 1); });
    $('copyPromptBtn').addEventListener('click', copyPrompt);

    $('modeIdentify').addEventListener('click', function () { setMode('identify'); });
    $('modeEnroll').addEventListener('click', function () { setMode('enroll'); });
    $('recBtn').addEventListener('click', function () {
      if (state.isRecording) { stopRecord(); } else { startRecord(); }
    });
    $('processBtn').addEventListener('click', processAudio);
    $('resetBtn').addEventListener('click', resetStudio);

    var dropzone = $('dropzone');
    var fileInput = $('fileInput');
    dropzone.addEventListener('click', function () { fileInput.click(); });
    fileInput.addEventListener('change', function (event) {
      acceptFile(event.target.files[0]);
      event.target.value = '';   // so re-picking the same file fires change again
    });
    ['dragenter', 'dragover'].forEach(function (type) {
      dropzone.addEventListener(type, function (event) {
        event.preventDefault();
        dropzone.classList.add('is-over');
      });
    });
    ['dragleave', 'drop'].forEach(function (type) {
      dropzone.addEventListener(type, function () { dropzone.classList.remove('is-over'); });
    });
    dropzone.addEventListener('drop', function (event) {
      event.preventDefault();
      acceptFile(event.dataTransfer.files[0]);
    });

    $('refreshEmployeesBtn').addEventListener('click', loadEmployees);
    $('enrollAllBtn').addEventListener('click', enrollAllEmployees);

    // Refresh immediately when the tab comes back rather than waiting out the
    // interval that was skipped while hidden.
    document.addEventListener('visibilitychange', function () {
      if (!document.hidden) { pollHealth(true); loadAnalytics(true); }
    });
    // Leaving the page mid-recording must not leave the microphone hot.
    window.addEventListener('pagehide', releaseMicrophone);

    pollHealth(true);
    loadPrompts();
    loadAnalytics(true);
    loadEmployees();
    setInterval(function () { loadAnalytics(); }, ANALYTICS_INTERVAL_MS);
    setInterval(function () { pollHealth(); }, HEALTH_INTERVAL_MS);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
