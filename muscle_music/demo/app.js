(() => {
  "use strict";

  const BPM = 112;
  const STEPS_PER_BEAT = 4;
  const STEPS_PER_BAR = 16;
  const LOOP_STEPS = 64;
  const STAGE_LABELS = ["基础动机", "节奏变化", "装饰发展", "高潮变化"];
  const ROOTS = [
    { name: "A", pc: 9 }, { name: "C", pc: 0 }, { name: "D", pc: 2 },
    { name: "E", pc: 4 }, { name: "G", pc: 7 }
  ];
  const MODES = [
    { name: "NATURAL MINOR", intervals: [0, 2, 3, 5, 7, 8, 10], progression: [0, 5, 2, 6] },
    { name: "DORIAN", intervals: [0, 2, 3, 5, 7, 9, 10], progression: [0, 3, 6, 0] }
  ];

  const LAYER_DEFS = [
    { gesture: "forward", name: "鼓与节奏", short: "鼓", type: "drums" },
    { gesture: "back", name: "环境纹理", short: "环境", type: "texture" },
    { gesture: "left", name: "和弦铺底", short: "和弦", type: "chords" },
    { gesture: "right", name: "主旋律", short: "旋律", type: "lead" },
    { gesture: "up", name: "高音琶音", short: "琶音", type: "arp" },
    { gesture: "down", name: "贝斯", short: "贝斯", type: "bass" }
  ];

  const KEY_MAP = { w: "forward", s: "back", a: "left", d: "right", r: "up", f: "down" };
  const GESTURE_BY_ID = {
    0: { key: null, label: "无动作" },
    1: { key: "forward", label: "向前" },
    2: { key: "back", label: "向后" },
    3: { key: "left", label: "向左" },
    4: { key: "right", label: "向右" },
    5: { key: "up", label: "向上" },
    6: { key: "down", label: "向下" },
    7: { key: "transport", label: "开始 / 结束" },
    8: { key: null, label: "保留状态" }
  };
  const DEFAULT_CHANNEL_LEVELS = [82, 45, 62, 74, 55, 78];
  const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
  const midiToHz = note => 440 * Math.pow(2, (note - 69) / 12);
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

  function mulberry32(seed) {
    return function random() {
      let value = seed += 0x6D2B79F5;
      value = Math.imul(value ^ value >>> 15, value | 1);
      value ^= value + Math.imul(value ^ value >>> 7, value | 61);
      return ((value ^ value >>> 14) >>> 0) / 4294967296;
    };
  }

  function profileDefault() {
    return { motion: 0.5, hold: 450, samples: 0 };
  }

  function blendProfile(profile, motion, hold, weight) {
    const w = profile.samples === 0 ? 1 : weight;
    return {
      motion: profile.motion * (1 - w) + motion * w,
      hold: profile.hold * (1 - w) + hold * w,
      samples: profile.samples + 1
    };
  }

  function mixProfile(local, global, globalWeight = 0.3) {
    return {
      motion: local.motion * (1 - globalWeight) + global.motion * globalWeight,
      hold: local.hold * (1 - globalWeight) + global.hold * globalWeight,
      samples: local.samples + global.samples
    };
  }

  class BrowserSynth {
    constructor(onStep) {
      this.onStep = onStep;
      this.context = null;
      this.playing = false;
      this.absoluteStep = 0;
      this.nextStepTime = 0;
      this.timer = null;
      this.sources = new Set();
      this.tracks = [];
      this.noiseBuffer = null;
      this.masterLevel = 0.72;
      this.spaceAmount = 0.46;
      this.autoMix = true;
      this.channelMix = DEFAULT_CHANNEL_LEVELS.map(level => ({ level: level / 100, muted: false }));
    }

    async ensureAudio() {
      if (this.context) {
        await this.context.resume();
        return;
      }
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) throw new Error("当前浏览器不支持 Web Audio API");
      this.context = new AudioContext({ latencyHint: "interactive" });

      this.master = this.context.createGain();
      this.master.gain.value = 0.0001;
      this.compressor = this.context.createDynamicsCompressor();
      this.compressor.threshold.value = -20;
      this.compressor.knee.value = 14;
      this.compressor.ratio.value = 4;
      this.compressor.attack.value = 0.004;
      this.compressor.release.value = 0.22;
      this.limiter = this.context.createDynamicsCompressor();
      this.limiter.threshold.value = -4;
      this.limiter.knee.value = 0;
      this.limiter.ratio.value = 20;
      this.limiter.attack.value = 0.002;
      this.limiter.release.value = 0.09;
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 512;
      this.meterData = new Uint8Array(this.analyser.fftSize);
      this.master.connect(this.compressor).connect(this.limiter).connect(this.analyser).connect(this.context.destination);

      this.reverb = this.context.createConvolver();
      this.reverb.buffer = this.makeImpulse(1.9, 2.8);
      this.reverbGain = this.context.createGain();
      this.reverb.connect(this.reverbGain).connect(this.master);

      this.delayNode = this.context.createDelay(1.0);
      this.delayNode.delayTime.value = (60 / BPM) * 0.75;
      this.delayFeedback = this.context.createGain();
      this.delayFeedback.gain.value = 0.24;
      this.delayGain = this.context.createGain();
      this.delayNode.connect(this.delayFeedback).connect(this.delayNode);
      this.delayNode.connect(this.delayGain).connect(this.master);

      const baseGains = [0.58, 0.16, 0.21, 0.23, 0.17, 0.31];
      const filterBase = [12000, 2600, 2300, 3400, 4800, 700];
      const reverbSends = [0.03, 0.52, 0.28, 0.20, 0.24, 0.05];
      const delaySends = [0.01, 0.12, 0.08, 0.32, 0.22, 0.03];
      const pans = [0, -0.26, -0.16, 0.18, 0.28, 0];
      for (let index = 0; index < LAYER_DEFS.length; index += 1) {
        const input = this.context.createGain();
        const filter = this.context.createBiquadFilter();
        filter.type = "lowpass";
        filter.frequency.value = filterBase[index];
        filter.Q.value = index === 3 || index === 4 ? 1.5 : 0.7;
        const expressionGain = this.context.createGain();
        expressionGain.gain.value = baseGains[index];
        const mixGain = this.context.createGain();
        mixGain.gain.value = this.channelMix[index].muted ? 0 : this.channelMix[index].level;
        const autoGain = this.context.createGain();
        autoGain.gain.value = 1;
        const panner = this.context.createStereoPanner();
        panner.pan.value = pans[index];
        const reverbSend = this.context.createGain();
        reverbSend.gain.value = reverbSends[index];
        const delaySend = this.context.createGain();
        delaySend.gain.value = delaySends[index];
        input.connect(filter).connect(expressionGain).connect(mixGain).connect(autoGain).connect(panner).connect(this.master);
        autoGain.connect(reverbSend).connect(this.reverb);
        autoGain.connect(delaySend).connect(this.delayNode);
        this.tracks.push({
          input, filter, expressionGain, mixGain, autoGain, panner,
          reverbSend, delaySend, baseGain: baseGains[index], baseFilter: filterBase[index]
        });
      }
      this.setSpace(this.spaceAmount);
      this.noiseBuffer = this.makeNoiseBuffer(1.0);
      await this.context.resume();
    }

    makeNoiseBuffer(seconds) {
      const length = Math.floor(this.context.sampleRate * seconds);
      const buffer = this.context.createBuffer(1, length, this.context.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < length; i += 1) data[i] = Math.random() * 2 - 1;
      return buffer;
    }

    makeImpulse(seconds, decay) {
      const length = Math.floor(this.context.sampleRate * seconds);
      const impulse = this.context.createBuffer(2, length, this.context.sampleRate);
      for (let channel = 0; channel < 2; channel += 1) {
        const data = impulse.getChannelData(channel);
        for (let i = 0; i < length; i += 1) {
          data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, decay);
        }
      }
      return impulse;
    }

    register(source) {
      this.sources.add(source);
      source.onended = () => this.sources.delete(source);
    }

    async start() {
      await this.ensureAudio();
      this.stopSources();
      this.absoluteStep = 0;
      this.nextStepTime = this.context.currentTime + 0.08;
      this.playing = true;
      this.master.gain.cancelScheduledValues(this.context.currentTime);
      this.master.gain.setValueAtTime(0.0001, this.context.currentTime);
      this.master.gain.exponentialRampToValueAtTime(Math.max(0.0001, this.masterLevel), this.context.currentTime + 0.12);
      this.timer = window.setInterval(() => this.scheduler(), 25);
    }

    stop() {
      if (!this.context) return;
      this.playing = false;
      if (this.timer) window.clearInterval(this.timer);
      this.timer = null;
      const now = this.context.currentTime;
      this.master.gain.cancelScheduledValues(now);
      this.master.gain.setTargetAtTime(0.0001, now, 0.025);
      window.setTimeout(() => this.stopSources(), 100);
    }

    stopSources() {
      for (const source of [...this.sources]) {
        try { source.stop(); } catch (_) { /* source already stopped */ }
      }
      this.sources.clear();
    }

    scheduler() {
      if (!this.playing) return;
      const horizon = this.context.currentTime + 0.12;
      while (this.nextStepTime < horizon) {
        this.onStep(this.absoluteStep, this.nextStepTime);
        this.absoluteStep += 1;
        this.nextStepTime += (60 / BPM) / STEPS_PER_BEAT;
      }
    }

    setMaster(level) {
      this.masterLevel = clamp(level);
      if (!this.context || !this.playing) return;
      this.master.gain.setTargetAtTime(Math.max(0.0001, this.masterLevel), this.context.currentTime, 0.04);
    }

    setSpace(amount) {
      this.spaceAmount = clamp(amount);
      if (!this.context) return;
      const now = this.context.currentTime;
      this.reverbGain.gain.setTargetAtTime(0.05 + this.spaceAmount * 0.31, now, 0.08);
      this.delayGain.gain.setTargetAtTime(0.03 + this.spaceAmount * 0.22, now, 0.08);
    }

    setAutoMix(enabled) {
      this.autoMix = Boolean(enabled);
      if (!this.context) return;
      const now = this.context.currentTime;
      for (const track of this.tracks) track.autoGain.gain.setTargetAtTime(1, now, 0.05);
    }

    setChannel(index, level, muted) {
      if (!this.channelMix[index]) return;
      this.channelMix[index] = { level: clamp(level), muted: Boolean(muted) };
      if (!this.context || !this.tracks[index]) return;
      const target = muted ? 0.0001 : Math.max(0.0001, this.channelMix[index].level);
      this.tracks[index].mixGain.gain.setTargetAtTime(target, this.context.currentTime, 0.035);
    }

    duckForKick(time) {
      if (!this.autoMix) return;
      for (const [index, floor] of [[5, 0.68], [2, 0.78], [1, 0.86]]) {
        const gain = this.tracks[index]?.autoGain.gain;
        if (!gain) continue;
        gain.cancelScheduledValues(time);
        gain.setValueAtTime(1, time);
        gain.linearRampToValueAtTime(floor, time + 0.008);
        gain.exponentialRampToValueAtTime(1, time + 0.19);
      }
    }

    meterLevel() {
      if (!this.analyser || !this.meterData || !this.playing) return 0;
      this.analyser.getByteTimeDomainData(this.meterData);
      let sum = 0;
      for (const sample of this.meterData) {
        const value = (sample - 128) / 128;
        sum += value * value;
      }
      return clamp(Math.sqrt(sum / this.meterData.length) * 3.3);
    }

    updateTrack(index, profile, activeCount) {
      if (!this.context || !this.tracks[index]) return;
      const track = this.tracks[index];
      const now = this.context.currentTime;
      const energy = clamp(profile.motion);
      const crowding = this.autoMix ? Math.max(0.72, 1 - Math.max(0, activeCount - 3) * 0.06) : 1;
      const targetGain = track.baseGain * (0.72 + energy * 0.48) * crowding;
      const brightness = index === 0
        ? track.baseFilter
        : track.baseFilter * (0.62 + profile.motion * 0.76);
      track.expressionGain.gain.setTargetAtTime(targetGain, now, 0.08);
      track.filter.frequency.setTargetAtTime(clamp(brightness, 120, 14000), now, 0.08);
    }

    playEvent(layerIndex, event, time, profile) {
      if (event.kind === "kick" || event.kind === "snare" || event.kind === "hat" || event.kind === "openhat") {
        this.playDrum(event.kind, event.vel / 127, time);
        return;
      }
      this.playTone(layerIndex, event.note, event.dur, event.vel / 127, time, profile);
    }

    playDrum(kind, velocity, time) {
      const destination = this.tracks[0].input;
      if (kind === "kick") {
        this.duckForKick(time);
        const oscillator = this.context.createOscillator();
        const gain = this.context.createGain();
        oscillator.type = "sine";
        oscillator.frequency.setValueAtTime(135, time);
        oscillator.frequency.exponentialRampToValueAtTime(44, time + 0.16);
        gain.gain.setValueAtTime(Math.max(0.0001, 0.72 * velocity), time);
        gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.24);
        oscillator.connect(gain).connect(destination);
        oscillator.start(time); oscillator.stop(time + 0.26); this.register(oscillator);
        return;
      }

      const noise = this.context.createBufferSource();
      noise.buffer = this.noiseBuffer;
      const filter = this.context.createBiquadFilter();
      const gain = this.context.createGain();
      if (kind === "snare") {
        filter.type = "bandpass"; filter.frequency.value = 1700; filter.Q.value = 0.8;
        gain.gain.setValueAtTime(0.34 * velocity, time);
        gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.16);
      } else {
        filter.type = "highpass"; filter.frequency.value = kind === "openhat" ? 6500 : 7800;
        const length = kind === "openhat" ? 0.16 : 0.045;
        gain.gain.setValueAtTime(0.12 * velocity, time);
        gain.gain.exponentialRampToValueAtTime(0.0001, time + length);
      }
      noise.connect(filter).connect(gain).connect(destination);
      noise.start(time); noise.stop(time + (kind === "snare" ? 0.18 : 0.20)); this.register(noise);

      if (kind === "snare") {
        const body = this.context.createOscillator();
        const bodyGain = this.context.createGain();
        body.type = "triangle"; body.frequency.value = 175;
        bodyGain.gain.setValueAtTime(0.20 * velocity, time);
        bodyGain.gain.exponentialRampToValueAtTime(0.0001, time + 0.10);
        body.connect(bodyGain).connect(destination);
        body.start(time); body.stop(time + 0.11); this.register(body);
      }
    }

    playTone(layerIndex, midiNote, durationSteps, velocity, time, profile) {
      const layerType = LAYER_DEFS[layerIndex].type;
      const stepSeconds = (60 / BPM) / STEPS_PER_BEAT;
      const duration = Math.max(stepSeconds, durationSteps * stepSeconds);
      const settings = {
        texture: { waves: [["sine", -7], ["triangle", 7]], level: 0.060, attack: 0.32, release: 0.75 },
        chords: { waves: [["sawtooth", -5], ["triangle", 5]], level: 0.055, attack: 0.035, release: 0.30 },
        lead: { waves: [["square", 0], ["triangle", 8]], level: 0.080, attack: 0.008, release: 0.20 },
        arp: { waves: [["triangle", 0], ["sine", 5]], level: 0.052, attack: 0.004, release: 0.13 },
        bass: { waves: [["sine", 0], ["sawtooth", -6]], level: 0.105, attack: 0.008, release: 0.12 }
      }[layerType];
      if (!settings) return;

      const envelope = this.context.createGain();
      const peak = Math.max(0.0001, settings.level * velocity * (0.82 + profile.motion * 0.22));
      const end = time + duration;
      envelope.gain.setValueAtTime(0.0001, time);
      envelope.gain.exponentialRampToValueAtTime(peak, time + settings.attack);
      envelope.gain.setValueAtTime(peak * 0.82, Math.max(time + settings.attack, end - settings.release));
      envelope.gain.exponentialRampToValueAtTime(0.0001, end + settings.release);
      envelope.connect(this.tracks[layerIndex].input);

      for (const [wave, detune] of settings.waves) {
        const oscillator = this.context.createOscillator();
        oscillator.type = wave;
        oscillator.frequency.value = midiToHz(midiNote);
        oscillator.detune.value = detune;
        oscillator.connect(envelope);
        oscillator.start(time);
        oscillator.stop(end + settings.release + 0.02);
        this.register(oscillator);
      }
    }
  }

  class MusicVisualizer {
    constructor(canvas) {
      this.canvas = canvas;
      this.context = canvas.getContext("2d");
      this.layers = [];
      this.particles = [];
      this.gestures = [];
      this.step = 0;
      this.meter = 0;
      this.playing = false;
      this.colors = {
        drums: "#45c7a3",
        texture: "#8f82dc",
        chords: "#55aee6",
        lead: "#ed729d",
        arp: "#e8b94f",
        bass: "#ee8a55"
      };
      this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas);
      this.resize();
      window.requestAnimationFrame(time => this.draw(time));
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      this.canvas.width = Math.max(1, Math.round(rect.width * ratio));
      this.canvas.height = Math.max(1, Math.round(rect.height * ratio));
      this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.width = rect.width;
      this.height = rect.height;
    }

    reset() {
      this.particles.length = 0;
      this.gestures.length = 0;
      this.step = 0;
      this.meter = 0;
    }

    setLayers(layers) { this.layers = layers || []; }
    setPlaying(playing) { this.playing = playing; }
    setStep(step) { this.step = step; }
    setMeter(level) { this.meter = level; }

    trigger(type, motion) {
      const now = performance.now();
      const energy = clamp(motion);
      this.gestures.push({ type, energy, start: now });
      const amount = this.reducedMotion ? 5 : 14 + Math.round(energy * 14);
      for (let index = 0; index < amount; index += 1) {
        this.particles.push({
          type,
          color: this.colors[type],
          start: now + Math.random() * 120,
          life: 850 + Math.random() * 850,
          seed: Math.random(),
          energy
        });
      }
      if (this.particles.length > 180) this.particles.splice(0, this.particles.length - 180);
    }

    layer(type) {
      return this.layers.find(item => item.type === type && (item.active || item.pending));
    }

    rgba(hex, alpha) {
      const number = Number.parseInt(hex.replace("#", ""), 16);
      return `rgba(${number >> 16},${number >> 8 & 255},${number & 255},${alpha})`;
    }

    area() {
      const dock = this.width < 620 ? 225 : this.width < 900 ? 178 : 96;
      return { width: this.width, height: Math.max(260, this.height - dock) };
    }

    drawBackdrop(ctx, width, height, time) {
      const wash = ctx.createLinearGradient(0, 0, width, height);
      wash.addColorStop(0, "#fbfaf6");
      wash.addColorStop(0.48, "#f7f4f1");
      wash.addColorStop(1, "#f3f1ea");
      ctx.fillStyle = wash;
      ctx.fillRect(0, 0, width, this.height);

      this.layers.filter(layer => layer.active || layer.pending).forEach((layer, index) => {
        const phase = time * 0.00012 + index * 1.7;
        const x = width * (0.5 + Math.cos(phase) * 0.25);
        const y = height * (0.48 + Math.sin(phase * 1.2) * 0.26);
        const radius = Math.max(width, height) * 0.34;
        const glow = ctx.createRadialGradient(x, y, 0, x, y, radius);
        glow.addColorStop(0, this.rgba(this.colors[layer.type], 0.09));
        glow.addColorStop(1, this.rgba(this.colors[layer.type], 0));
        ctx.fillStyle = glow;
        ctx.fillRect(0, 0, width, height);
      });
    }

    drawTexture(ctx, width, height, time, layer) {
      if (!layer) return;
      const energy = layer.profile?.motion ?? 0.5;
      for (let index = 0; index < 5; index += 1) {
        const phase = time * (0.00013 + index * 0.000012) + index * 1.4;
        const x = width * (0.18 + index * 0.17) + Math.sin(phase) * 42;
        const y = height * (0.32 + Math.sin(phase * 1.3) * 0.20);
        const radius = 70 + index * 19 + energy * 36;
        const glow = ctx.createRadialGradient(x, y, 0, x, y, radius);
        glow.addColorStop(0, this.rgba(this.colors.texture, 0.16));
        glow.addColorStop(0.58, this.rgba(this.colors.texture, 0.055));
        glow.addColorStop(1, this.rgba(this.colors.texture, 0));
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    drawChords(ctx, width, height, time, layer) {
      if (!layer) return;
      const generation = Math.max(0, layer.generation);
      const energy = layer.profile?.motion ?? 0.5;
      ctx.lineCap = "round";
      for (let voice = 0; voice < 3; voice += 1) {
        ctx.beginPath();
        for (let x = -20; x <= width + 20; x += 14) {
          const y = height * (0.43 + voice * 0.07) + Math.sin(x * 0.012 + time * 0.001 + voice) * (7 + energy * 12);
          if (x === -20) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = this.rgba(this.colors.chords, 0.15 + generation * 0.025);
        ctx.lineWidth = 10 - voice * 2;
        ctx.stroke();
      }
    }

    drawBass(ctx, width, height, time, layer) {
      if (!layer) return;
      const energy = layer.profile?.motion ?? 0.5;
      const baseline = height * 0.79;
      ctx.beginPath();
      for (let x = 0; x <= width; x += 8) {
        const envelope = Math.sin(Math.PI * x / width);
        const y = baseline + Math.sin(x * 0.022 - time * 0.0021) * (10 + energy * 22) * envelope;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = this.rgba(this.colors.bass, 0.34);
      ctx.lineWidth = 8 + this.meter * 8;
      ctx.lineCap = "round";
      ctx.stroke();
    }

    drawLead(ctx, width, height, time, layer) {
      if (!layer) return;
      const generation = Math.max(0, layer.generation);
      const energy = layer.profile?.motion ?? 0.5;
      const points = 7 + Math.min(5, generation * 2);
      const positions = [];
      ctx.beginPath();
      for (let index = 0; index < points; index += 1) {
        const x = width * (0.16 + index / Math.max(1, points - 1) * 0.68);
        const y = height * 0.47 + Math.sin(index * 1.36 + time * 0.0017) * (35 + energy * 45);
        positions.push([x, y]);
        if (index === 0) ctx.moveTo(x, y); else ctx.quadraticCurveTo(x - 22, y - 8, x, y);
      }
      ctx.strokeStyle = this.rgba(this.colors.lead, 0.65);
      ctx.lineWidth = 3.5;
      ctx.lineCap = "round";
      ctx.stroke();
      positions.forEach(([x, y], index) => {
        ctx.fillStyle = this.rgba(this.colors.lead, 0.72);
        ctx.beginPath();
        ctx.arc(x, y, 3.5 + (index % 3 === 0 ? 2.5 : 0), 0, Math.PI * 2);
        ctx.fill();
      });
    }

    drawArp(ctx, width, height, time, layer) {
      if (!layer) return;
      const count = 9 + Math.max(0, layer.generation) * 3;
      for (let index = 0; index < count; index += 1) {
        const cycle = (time * 0.00035 + index / count) % 1;
        const x = width * (0.22 + cycle * 0.56) + Math.sin(index * 1.9) * 20;
        const y = height * (0.72 - cycle * 0.54);
        ctx.fillStyle = this.rgba(this.colors.arp, 0.28 + cycle * 0.48);
        ctx.beginPath();
        ctx.arc(x, y, 2.5 + (index % 4 === 0 ? 2 : 0), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    drawDrums(ctx, width, height, layer) {
      if (!layer) return;
      const cx = width / 2;
      const cy = height * 0.48;
      const beatPhase = (this.step % 4) / 4;
      const energy = layer.profile?.motion ?? 0.5;
      for (let ring = 0; ring < 3; ring += 1) {
        const phase = (beatPhase + ring / 3) % 1;
        ctx.strokeStyle = this.rgba(this.colors.drums, (1 - phase) * 0.32);
        ctx.lineWidth = 2 + energy * 2;
        ctx.beginPath();
        ctx.arc(cx, cy, 42 + phase * Math.min(width, height) * 0.30, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    particlePosition(particle, progress, width, height) {
      const cx = width / 2;
      const cy = height * 0.48;
      const spread = (particle.seed - 0.5) * Math.min(width, height) * 0.46;
      const ease = 1 - Math.pow(1 - progress, 3);
      if (particle.type === "drums") {
        const angle = particle.seed * Math.PI * 2;
        const radius = ease * Math.min(width, height) * 0.38;
        return [cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius];
      }
      if (particle.type === "texture") return [width * (1 - ease), cy + spread * 0.5 + Math.sin(progress * 8 + particle.seed) * 28];
      if (particle.type === "chords") return [ease * width, cy + spread * 0.42];
      if (particle.type === "lead") return [width * (0.12 + ease * 0.76), cy + Math.sin(ease * 9 + particle.seed * 4) * 65];
      if (particle.type === "arp") return [cx + spread * 0.55, height * (0.88 - ease * 0.78)];
      return [cx + spread * 0.48, height * (0.90 - ease * 0.18) + Math.sin(ease * 12) * 12];
    }

    drawParticles(ctx, width, height, time) {
      this.particles = this.particles.filter(particle => time < particle.start + particle.life);
      for (const particle of this.particles) {
        const progress = clamp((time - particle.start) / particle.life);
        if (progress <= 0) continue;
        const [x, y] = this.particlePosition(particle, progress, width, height);
        const alpha = Math.sin(progress * Math.PI) * (0.3 + particle.energy * 0.5);
        ctx.fillStyle = this.rgba(particle.color, alpha);
        ctx.beginPath();
        ctx.arc(x, y, 2 + particle.energy * 4 + particle.seed * 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    drawGestureEchoes(ctx, width, height, time) {
      this.gestures = this.gestures.filter(item => time - item.start < 950);
      for (const item of this.gestures) {
        const progress = clamp((time - item.start) / 950);
        ctx.strokeStyle = this.rgba(this.colors[item.type], (1 - progress) * 0.28);
        ctx.lineWidth = 2 + item.energy * 5;
        ctx.beginPath();
        if (item.type === "drums") {
          ctx.arc(width / 2, height * 0.48, 38 + progress * Math.min(width, height) * 0.42, 0, Math.PI * 2);
        } else {
          const y = item.type === "bass" ? height * 0.82 : height * (0.22 + progress * 0.52);
          ctx.moveTo(width * 0.14, y);
          ctx.bezierCurveTo(width * 0.34, y - 55, width * 0.66, y + 55, width * 0.86, y);
        }
        ctx.stroke();
      }
    }

    drawCore(ctx, width, height, time) {
      const cx = width / 2;
      const cy = height * 0.48;
      const pulse = this.playing ? 1 + Math.sin(time * 0.004) * 0.035 + this.meter * 0.12 : 1;
      const radius = 28 * pulse;
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius * 3.2);
      glow.addColorStop(0, `rgba(37,38,48,${0.17 + this.meter * 0.18})`);
      glow.addColorStop(0.32, "rgba(37,38,48,0.055)");
      glow.addColorStop(1, "rgba(37,38,48,0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 3.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = this.playing ? "rgba(37,38,48,0.84)" : "rgba(37,38,48,0.13)";
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    draw(time) {
      if (!this.width || !this.height) this.resize();
      const ctx = this.context;
      const { width, height } = this.area();
      ctx.clearRect(0, 0, this.width, this.height);
      this.drawBackdrop(ctx, width, height, time);
      this.drawTexture(ctx, width, height, time, this.layer("texture"));
      this.drawChords(ctx, width, height, time, this.layer("chords"));
      this.drawBass(ctx, width, height, time, this.layer("bass"));
      this.drawLead(ctx, width, height, time, this.layer("lead"));
      this.drawArp(ctx, width, height, time, this.layer("arp"));
      this.drawDrums(ctx, width, height, this.layer("drums"));
      this.drawGestureEchoes(ctx, width, height, time);
      this.drawParticles(ctx, width, height, time);
      this.drawCore(ctx, width, height, time);
      window.requestAnimationFrame(next => this.draw(next));
    }
  }

  class JilvDemo {
    constructor() {
      this.dom = {
        start: document.querySelector("#startButton"), auto: document.querySelector("#autoButton"),
        motion: document.querySelector("#motionSlider"), motionValue: document.querySelector("#motionValue"),
        motionMode: document.querySelector("#motionMode"), gestureValue: document.querySelector("#gestureValue"),
        confidenceValue: document.querySelector("#confidenceValue"), currentGesture: document.querySelector("#currentGesture"),
        sourceReadout: document.querySelector("#sourceReadout"), sourceText: document.querySelector("#sourceText"),
        liveBadge: document.querySelector("#liveBadge"),
        transportText: document.querySelector("#transportText"), transportDot: document.querySelector("#transportDot"),
        clock: document.querySelector("#clockText"), status: document.querySelector("#mainStatus"),
        personality: document.querySelector("#personalityLabel"), harmony: document.querySelector("#harmonyText"),
        beatHand: document.querySelector("#beatHand"), eventLog: document.querySelector("#eventLog"),
        profileMotion: document.querySelector("#profileMotion"), profileLegato: document.querySelector("#profileLegato"),
        profileLayers: document.querySelector("#profileLayers"), profileComplexity: document.querySelector("#profileComplexity"),
        autoMix: document.querySelector("#autoMixToggle"), space: document.querySelector("#spaceSlider"),
        spaceValue: document.querySelector("#spaceValue"), master: document.querySelector("#masterSlider"),
        masterValue: document.querySelector("#masterValue"), masterMeter: document.querySelector("#masterMeter"),
        resetMix: document.querySelector("#resetMixButton"), canvas: document.querySelector("#musicCanvas"),
        visualStage: document.querySelector("#visualStage"),
        channels: [...document.querySelectorAll("[data-mix-channel]")]
      };
      this.visual = new MusicVisualizer(this.dom.canvas);
      this.audio = new BrowserSynth((step, time) => this.onStep(step, time));
      this.playing = false;
      this.starting = false;
      this.finished = false;
      this.autoRunning = false;
      this.held = new Map();
      this.inputState = {
        connected: false,
        candidate: 0,
        candidateSince: 0,
        stable: 0,
        lastFrameAt: 0
      };
      this.resetSession();
      this.bindEvents();
      this.syncMixer();
      this.connectInputStream();
      this.renderProfile();
      this.animateMeter();
    }

    resetSession() {
      this.sessionSeed = Math.floor(Math.random() * 0x7fffffff) || 42;
      const random = mulberry32(this.sessionSeed);
      this.root = ROOTS[Math.floor(random() * ROOTS.length)];
      this.mode = MODES[Math.floor(random() * MODES.length)];
      this.globalProfile = profileDefault();
      this.layers = LAYER_DEFS.map((definition, index) => ({
        ...definition,
        index,
        generation: -1,
        active: false,
        pending: null,
        pendingTarget: null,
        pattern: [],
        profile: profileDefault(),
        seed: (this.sessionSeed + (index + 1) * 1009) >>> 0,
        element: document.querySelector(`[data-gesture="${definition.gesture}"]`)
      }));
      this.visual.reset();
      this.visual.setLayers(this.layers);
      this.renderHarmony();
      this.layers.forEach(layer => this.renderLayer(layer));
      this.renderProfile();
    }

    bindEvents() {
      this.dom.start.addEventListener("click", () => this.toggleSession());
      this.dom.auto.addEventListener("click", () => this.runAutoDemo());
      for (const layer of this.layers) {
        const begin = event => { event.preventDefault(); this.beginGesture(layer.gesture); };
        const end = event => { event.preventDefault(); this.endGesture(layer.gesture); };
        layer.element.addEventListener("pointerdown", begin);
        layer.element.addEventListener("pointerup", end);
        layer.element.addEventListener("pointercancel", end);
        layer.element.addEventListener("pointerleave", event => {
          if (this.held.has(layer.gesture)) end(event);
        });
      }
      const updateSlider = () => {
        this.dom.motionValue.value = this.dom.motion.value;
        for (const gesture of this.held.keys()) this.updateHeldExpression(gesture);
      };
      this.dom.motion.addEventListener("input", updateSlider);
      updateSlider();
      this.dom.autoMix.addEventListener("change", () => {
        this.audio.setAutoMix(this.dom.autoMix.checked);
        this.updateAllTracks();
        this.log(this.dom.autoMix.checked ? "自动平衡已开启。" : "自动平衡已关闭，可手动调整各轨比例。");
      });
      this.dom.space.addEventListener("input", () => {
        this.dom.spaceValue.value = this.dom.space.value;
        this.audio.setSpace(Number(this.dom.space.value) / 100);
      });
      this.dom.master.addEventListener("input", () => {
        this.dom.masterValue.value = this.dom.master.value;
        this.audio.setMaster(Number(this.dom.master.value) / 100);
      });
      this.dom.resetMix.addEventListener("click", () => this.resetMixer());
      for (const channel of this.dom.channels) {
        const slider = channel.querySelector('input[type="range"]');
        const output = channel.querySelector("output");
        const mute = channel.querySelector(".mute-button");
        slider.addEventListener("input", () => {
          output.value = slider.value;
          this.applyChannelMix(channel);
        });
        mute.addEventListener("click", () => {
          mute.classList.toggle("active");
          mute.setAttribute("aria-pressed", String(mute.classList.contains("active")));
          channel.classList.toggle("muted", mute.classList.contains("active"));
          this.applyChannelMix(channel);
        });
      }
      window.addEventListener("keydown", event => {
        if (event.repeat) return;
        const key = event.key.toLowerCase();
        if (key === "j") { event.preventDefault(); this.toggleSession(); return; }
        if (KEY_MAP[key]) { event.preventDefault(); this.beginGesture(KEY_MAP[key]); }
      });
      window.addEventListener("keyup", event => {
        const key = event.key.toLowerCase();
        if (KEY_MAP[key]) { event.preventDefault(); this.endGesture(KEY_MAP[key]); }
      });
    }

    syncMixer() {
      this.dom.spaceValue.value = this.dom.space.value;
      this.dom.masterValue.value = this.dom.master.value;
      this.audio.setSpace(Number(this.dom.space.value) / 100);
      this.audio.setMaster(Number(this.dom.master.value) / 100);
      this.audio.setAutoMix(this.dom.autoMix.checked);
      for (const channel of this.dom.channels) this.applyChannelMix(channel);
    }

    applyChannelMix(channel) {
      const index = Number(channel.dataset.mixChannel);
      const slider = channel.querySelector('input[type="range"]');
      const mute = channel.querySelector(".mute-button");
      this.audio.setChannel(index, Number(slider.value) / 100, mute.classList.contains("active"));
    }

    resetMixer() {
      this.dom.autoMix.checked = true;
      this.dom.space.value = 46;
      this.dom.master.value = 72;
      this.dom.channels.forEach((channel, index) => {
        const slider = channel.querySelector('input[type="range"]');
        const output = channel.querySelector("output");
        const mute = channel.querySelector(".mute-button");
        slider.value = DEFAULT_CHANNEL_LEVELS[index];
        output.value = slider.value;
        mute.classList.remove("active");
        mute.setAttribute("aria-pressed", "false");
        channel.classList.remove("muted");
      });
      this.syncMixer();
      this.updateAllTracks();
      this.log("混音已恢复为推荐比例。");
    }

    connectInputStream() {
      if (!("EventSource" in window)) {
        this.setInputConnection(false, "浏览器不支持实时数据");
        return;
      }
      this.eventSource = new EventSource("/api/stream");
      this.eventSource.onopen = () => {
        if (!this.inputState.connected) this.setInputConnection(false, "等待手环");
      };
      this.eventSource.onmessage = event => {
        try {
          this.handleInputFrame(JSON.parse(event.data));
        } catch (error) {
          console.warn("忽略无效的手环数据", error);
        }
      };
      this.eventSource.onerror = () => {
        if (performance.now() - this.inputState.lastFrameAt > 1800) {
          this.setInputConnection(false, "连接不可用", true);
        }
      };
      window.setInterval(() => {
        if (this.inputState.connected && performance.now() - this.inputState.lastFrameAt > 1700) {
          this.setInputConnection(false, "等待手环");
        }
      }, 500);
    }

    setInputConnection(connected, message, isError = false) {
      const changed = this.inputState.connected !== connected;
      this.inputState.connected = connected;
      this.dom.sourceReadout.dataset.state = connected ? "live" : isError ? "error" : "waiting";
      this.dom.sourceText.textContent = message || (connected ? "手环已连接" : "等待手环");
      this.dom.liveBadge.textContent = connected ? "实时" : "等待数据";
      this.dom.liveBadge.classList.toggle("live", connected);
      this.dom.motion.disabled = connected;
      this.dom.motionMode.textContent = connected ? "手环" : "手动";
      if (!connected) {
        this.dom.confidenceValue.textContent = "--";
        if (changed && this.inputState.stable) this.commitInputGesture(0);
        this.inputState.candidate = 0;
        this.inputState.candidateSince = performance.now();
        this.dom.gestureValue.textContent = "无动作";
        this.dom.currentGesture.textContent = "无动作";
      }
    }

    handleInputFrame(frame) {
      const now = performance.now();
      this.inputState.lastFrameAt = now;
      if (!frame.connected) {
        this.setInputConnection(false, frame.error ? "连接不可用" : "等待手环");
        return;
      }

      this.setInputConnection(true, "手环已连接");
      const rawId = Number(frame.gesture_id);
      const confidence = clamp(Number(frame.confidence));
      const motion = clamp(Number(frame.motion_energy));
      const isConfident = Boolean(GESTURE_BY_ID[rawId]) && confidence >= 0.55;
      const gestureId = isConfident ? rawId : 0;
      const gesture = GESTURE_BY_ID[rawId] || GESTURE_BY_ID[0];
      const visibleLabel = isConfident ? gesture.label : "置信度不足";

      this.dom.motion.value = String(Math.round(motion * 100));
      this.dom.motionValue.value = this.dom.motion.value;
      this.dom.gestureValue.textContent = visibleLabel;
      this.dom.confidenceValue.textContent = `${Math.round(confidence * 100)}%`;
      this.dom.currentGesture.textContent = visibleLabel;

      if (gestureId !== this.inputState.candidate) {
        this.inputState.candidate = gestureId;
        this.inputState.candidateSince = now;
        return;
      }
      if (gestureId !== this.inputState.stable && now - this.inputState.candidateSince >= 120) {
        this.commitInputGesture(gestureId);
      } else if (gestureId === this.inputState.stable) {
        const key = GESTURE_BY_ID[gestureId]?.key;
        if (key && key !== "transport" && this.held.has(key)) this.updateHeldExpression(key);
      }
    }

    commitInputGesture(nextId) {
      const previous = GESTURE_BY_ID[this.inputState.stable];
      if (previous?.key && previous.key !== "transport") this.endGesture(previous.key);

      this.inputState.stable = nextId;
      const next = GESTURE_BY_ID[nextId] || GESTURE_BY_ID[0];
      if (next.key === "transport") {
        this.toggleSession();
      } else if (next.key) {
        this.beginGesture(next.key);
      }
    }

    animateMeter() {
      const draw = () => {
        const level = this.audio.meterLevel();
        this.dom.masterMeter.style.width = `${Math.round(level * 100)}%`;
        this.visual.setMeter(level);
        window.requestAnimationFrame(draw);
      };
      window.requestAnimationFrame(draw);
    }

    values() {
      return { motion: Number(this.dom.motion.value) / 100 };
    }

    async toggleSession() {
      if (this.starting) return;
      try {
        if (this.playing) {
          this.audio.stop();
          this.playing = false;
          this.visual.setPlaying(false);
          this.finished = true;
          this.dom.transportText.textContent = "作品已完成";
          this.dom.transportDot.classList.remove("playing");
          this.dom.start.textContent = "开始新作品";
          this.dom.status.textContent = "作品已保留";
          this.log("本次作品结束，下次开始将生成新调性。");
          return;
        }
        this.starting = true;
        this.dom.start.disabled = true;
        if (this.finished) this.resetSession();
        this.finished = false;
        this.dom.status.textContent = "1 拍倒计时";
        this.dom.transportText.textContent = "准备中";
        await this.audio.ensureAudio();
        await delay(535);
        await this.audio.start();
        this.playing = true;
        this.visual.setPlaying(true);
        this.dom.transportText.textContent = "正在生成";
        this.dom.transportDot.classList.add("playing");
        this.dom.start.textContent = "结束作品";
        this.dom.status.textContent = "用方向加入声部";
        this.log("已开始。首次进入某个方向加入声部，重复进入让它发展。");
      } catch (error) {
        this.dom.status.textContent = "音频启动失败";
        this.log(error.message || String(error));
      } finally {
        this.starting = false;
        this.dom.start.disabled = false;
      }
    }

    beginGesture(gesture) {
      if (!this.inputState.connected) {
        const manualGesture = Object.values(GESTURE_BY_ID).find(item => item.key === gesture);
        if (manualGesture) {
          this.dom.gestureValue.textContent = manualGesture.label;
          this.dom.currentGesture.textContent = manualGesture.label;
        }
      }
      if (!this.playing) {
        this.log("请先点击开始，或按 J 键启用声音。");
        return;
      }
      if (this.held.has(gesture)) return;
      const layer = this.layers.find(item => item.gesture === gesture);
      if (!layer) return;
      this.held.set(gesture, performance.now());
      const { motion } = this.values();
      this.visual.trigger(layer.type, motion);
      layer.profile = blendProfile(layer.profile, motion, layer.profile.hold, 0.34);
      this.globalProfile = blendProfile(this.globalProfile, motion, this.globalProfile.hold, 0.16);
      layer.generation += 1;
      const effective = mixProfile(layer.profile, this.globalProfile);
      layer.pending = this.makePattern(layer, effective);
      const current = this.audio.absoluteStep;
      const quantum = layer.active ? STEPS_PER_BAR : STEPS_PER_BEAT;
      layer.pendingTarget = Math.ceil((current + 1) / quantum) * quantum;
      layer.element.classList.add("pending", "pulse");
      window.setTimeout(() => layer.element.classList.remove("pulse"), 190);
      this.renderLayer(layer);
      this.renderProfile();
      const timing = layer.active ? "下一小节" : "下一拍";
      this.dom.status.textContent = `${layer.name} · ${STAGE_LABELS[Math.min(3, layer.generation)]}`;
      this.log(`${layer.name}已确认，${timing}生效。`);
      this.updateAllTracks();
    }

    updateHeldExpression(gesture) {
      const layer = this.layers.find(item => item.gesture === gesture);
      if (!layer) return;
      const start = this.held.get(gesture) || performance.now();
      const { motion } = this.values();
      layer.profile = blendProfile(layer.profile, motion, performance.now() - start, 0.05);
      this.globalProfile = blendProfile(this.globalProfile, motion, performance.now() - start, 0.02);
      this.updateAllTracks();
      this.renderProfile();
    }

    endGesture(gesture) {
      if (!this.held.has(gesture)) return;
      const started = this.held.get(gesture);
      this.held.delete(gesture);
      const layer = this.layers.find(item => item.gesture === gesture);
      if (!layer) return;
      const { motion } = this.values();
      const heldMs = Math.max(100, performance.now() - started);
      layer.profile = blendProfile(layer.profile, motion, heldMs, 0.30);
      this.globalProfile = blendProfile(this.globalProfile, motion, heldMs, 0.10);
      this.updateAllTracks();
      this.renderProfile();
      if (!this.inputState.connected && this.held.size === 0) {
        this.dom.gestureValue.textContent = "无动作";
        this.dom.currentGesture.textContent = "无动作";
      }
    }

    onStep(absoluteStep, time) {
      for (const layer of this.layers) {
        if (layer.pending && layer.pendingTarget <= absoluteStep) {
          layer.pattern = layer.pending;
          layer.pending = null;
          layer.pendingTarget = null;
          layer.active = true;
          this.renderLayer(layer);
          this.updateAllTracks();
        }
      }
      const phraseStep = absoluteStep % LOOP_STEPS;
      const activeCount = this.layers.filter(layer => layer.active).length;
      for (const layer of this.layers) {
        if (!layer.active) continue;
        const effective = mixProfile(layer.profile, this.globalProfile);
        for (const event of layer.pattern) {
          if (event.step !== phraseStep || !this.shouldPlay(layer, event, activeCount, effective)) continue;
          this.audio.playEvent(layer.index, event, time, effective);
        }
      }
      const latency = Math.max(0, (time - this.audio.context.currentTime) * 1000);
      window.setTimeout(() => this.renderClock(absoluteStep), latency);
    }

    shouldPlay(layer, event, activeCount, profile) {
      const anchor = event.step % STEPS_PER_BEAT === 0 || event.kind === "kick" || event.kind === "snare";
      if (anchor) return true;
      const energy = clamp(profile.motion);
      const budget = Math.max(0.60, 1 - Math.max(0, activeCount - 3) * 0.09);
      const threshold = (0.24 + energy * 0.76) * budget;
      const stable = ((event.step * 37 + (event.note || 0) * 17 + layer.index * 29) % 101) / 100;
      return stable <= threshold;
    }

    updateAllTracks() {
      const activeCount = this.layers.filter(layer => layer.active || layer.pending).length;
      for (const layer of this.layers) {
        this.audio.updateTrack(layer.index, mixProfile(layer.profile, this.globalProfile), activeCount);
      }
    }

    degreePitch(degree, octave) {
      const count = this.mode.intervals.length;
      const normalized = ((degree % count) + count) % count;
      const octaveAdd = Math.floor(degree / count);
      return 12 * (octave + 1 + octaveAdd) + this.root.pc + this.mode.intervals[normalized];
    }

    chord(bar, octave = 3) {
      const degree = this.mode.progression[bar % 4];
      return [0, 2, 4].map(offset => this.degreePitch(degree + offset, octave));
    }

    scaleNotes(low, high) {
      const allowed = new Set(this.mode.intervals.map(value => (this.root.pc + value) % 12));
      const notes = [];
      for (let note = low; note <= high; note += 1) if (allowed.has(note % 12)) notes.push(note);
      return notes;
    }

    nearest(value, options) {
      return options.reduce((best, candidate) => Math.abs(candidate - value) < Math.abs(best - value) ? candidate : best, options[0]);
    }

    velocity(profile, base, offset = 0) {
      const energy = profile.motion;
      return Math.round(clamp(base + (energy - 0.5) * 26 + offset, 28, 118));
    }

    makePattern(layer, profile) {
      const stage = Math.min(3, layer.generation);
      const random = mulberry32(layer.seed);
      const events = [];
      const push = (step, dur, note, vel, kind = "tone") => events.push({ step, dur, note, vel, kind });

      if (layer.type === "drums") {
        for (let bar = 0; bar < 4; bar += 1) {
          const start = bar * 16;
          const kicks = [0, 8];
          if (stage >= 1) kicks.push(bar % 2 ? 10 : 6);
          if (stage >= 3 && profile.motion > 0.42) kicks.push(14);
          kicks.forEach(step => push(start + step, 1, 36, this.velocity(profile, 100), "kick"));
          [4, 12].forEach(step => push(start + step, 1, 38, this.velocity(profile, 92), "snare"));
          let stride = stage === 0 && profile.motion < 0.58 ? 4 : 2;
          if (stage >= 2 && profile.motion > 0.68) stride = 1;
          for (let step = 0; step < 16; step += stride) {
            push(start + step, 1, 42, this.velocity(profile, step % 4 === 0 ? 72 : 58), step === 14 && stage >= 2 ? "openhat" : "hat");
          }
        }
      }

      if (layer.type === "bass") {
        const allowed = this.scaleNotes(32, 55);
        for (let bar = 0; bar < 4; bar += 1) {
          const chord = this.chord(bar, 2);
          const root = this.nearest(chord[0], allowed);
          const fifth = this.nearest(chord[2], allowed);
          const pattern = [[0, root], [8, fifth]];
          if (stage >= 1) pattern.push([6, this.nearest(root + 12, allowed)], [12, root]);
          if (stage >= 2) pattern.push([3, this.nearest(root + 3, allowed)], [14, fifth]);
          if (stage >= 3 && profile.motion > 0.45) pattern.push([10, this.nearest(root + 7, allowed)]);
          pattern.sort((a, b) => a[0] - b[0]);
          pattern.forEach(([step, note], index) => {
            const next = pattern[index + 1]?.[0] ?? 16;
            const legato = clamp(profile.hold / 1400);
            push(bar * 16 + step, Math.max(1, Math.round((next - step) * (0.48 + legato * 0.38))), note, this.velocity(profile, 84));
          });
        }
      }

      if (layer.type === "chords") {
        let previousTop = null;
        for (let bar = 0; bar < 4; bar += 1) {
          const raw = this.chord(bar, 3);
          const inversions = [raw, [raw[1], raw[2], raw[0] + 12], [raw[2], raw[0] + 12, raw[1] + 12]];
          const chord = previousTop == null ? inversions[0] : inversions.reduce((best, value) => Math.abs(value[2] - previousTop) < Math.abs(best[2] - previousTop) ? value : best, inversions[0]);
          previousTop = chord[2];
          const attacks = stage === 0 ? [0] : [0, 8];
          if (stage >= 3 && profile.motion > 0.56) attacks.push(6, 14);
          attacks.sort((a, b) => a - b).forEach((step, index, all) => {
            const next = all[index + 1] ?? 16;
            chord.forEach((note, voice) => push(bar * 16 + step, Math.max(2, next - step - (stage >= 2 ? 1 : 0)), note + (stage >= 2 && voice === 2 && bar % 2 ? 12 : 0), this.velocity(profile, 66, voice * 2)));
          });
        }
      }

      if (layer.type === "lead") {
        const allowed = this.scaleNotes(60, 84);
        let cursor = allowed[Math.floor(allowed.length / 2)];
        const motif = [0, 1, 2, 3].map(() => {
          cursor = this.nearest(cursor + [-3, -2, 2, 3, 5][Math.floor(random() * 5)], allowed);
          return cursor;
        });
        for (let bar = 0; bar < 4; bar += 1) {
          const local = [[0, motif[0]], [4, motif[1]], [8, motif[2]], [12, motif[3]]];
          if (stage >= 1) { local[1][0] = bar % 2 ? 5 : 3; local.push([10, this.nearest(motif[2] + 2, allowed)]); }
          if (stage >= 2) local.push([2, this.nearest(motif[0] + 2, allowed)], [14, this.nearest(motif[3] - 3, allowed)]);
          if (stage >= 3 && profile.motion > 0.38) local.push([7, this.nearest(motif[2] + 7, allowed)], [15, motif[0]]);
          const chordPcs = new Set(this.chord(bar, 4).map(note => note % 12));
          const stableNotes = allowed.filter(note => chordPcs.has(note % 12));
          for (let [step, note] of local) {
            if (step % 8 === 0) note = this.nearest(note, stableNotes);
            push(bar * 16 + step, 1 + Math.round(clamp(profile.hold / 1400) * 3), note, this.velocity(profile, 80, Math.round((random() - 0.5) * 8)));
          }
        }
      }

      if (layer.type === "arp") {
        for (let bar = 0; bar < 4; bar += 1) {
          const chord = this.chord(bar, 4); chord.push(chord[0] + 12);
          let stride = stage === 0 ? 4 : 2;
          if (stage >= 2 && profile.motion > 0.62) stride = 1;
          let order = stage >= 1 ? [0, 2, 1, 3, 2, 1, 0, 2] : [0, 1, 2, 1];
          if (stage >= 3 && bar % 2) order = [...order].reverse();
          let index = 0;
          for (let step = 0; step < 16; step += stride) {
            push(bar * 16 + step, Math.max(1, stride - 1), chord[order[index % order.length]], this.velocity(profile, 69));
            index += 1;
          }
        }
      }

      if (layer.type === "texture") {
        const allowed = this.scaleNotes(48, 79);
        for (let bar = 0; bar < 4; bar += 1) {
          const chord = this.chord(bar, 3);
          const base = this.nearest(chord[stage % chord.length], allowed);
          push(bar * 16, 15, base, this.velocity(profile, 48));
          if (stage >= 1) push(bar * 16 + 8, 7, this.nearest(base + 12, allowed), this.velocity(profile, 39));
          if (stage >= 3 && profile.motion > 0.45) push(bar * 16 + 12, 3, this.nearest(base + 5, allowed), this.velocity(profile, 50));
        }
      }

      if (layer.generation > 3) this.mutateOrnaments(events, layer);
      return events.sort((a, b) => a.step - b.step || (a.note || 0) - (b.note || 0));
    }

    mutateOrnaments(events, layer) {
      const epoch = layer.generation - 3;
      const random = mulberry32((layer.seed + epoch * 7919) >>> 0);
      const allowed = this.scaleNotes(28, 100);
      events.forEach((event, index) => {
        const anchor = event.step % 4 === 0 || event.kind === "kick" || event.kind === "snare";
        if (anchor || (index + epoch) % 3 !== 0) return;
        const local = event.step % 16;
        event.step = event.step - local + clamp(local + (random() < 0.5 ? -1 : 1), 1, 15);
        if (event.kind === "tone") {
          const candidates = allowed.filter(note => Math.abs(note - event.note) <= 5 && note !== event.note);
          if (candidates.length) event.note = candidates[Math.floor(random() * candidates.length)];
        }
      });
    }

    renderLayer(layer) {
      if (!layer.element) return;
      layer.element.classList.toggle("active", layer.active);
      layer.element.classList.toggle("pending", Boolean(layer.pending));
      const small = layer.element.querySelector("small");
      if (layer.pending) small.textContent = "等待节拍";
      else if (layer.active) small.textContent = STAGE_LABELS[Math.min(3, layer.generation)];
      else small.textContent = layer.short;
      this.visual.setLayers(this.layers);
    }

    renderHarmony() {
      if (this.dom?.harmony) this.dom.harmony.textContent = `${this.root.name} ${this.mode.name} · ${BPM} BPM`;
    }

    renderClock(step) {
      const bar = Math.floor(step / 16) % 4 + 1;
      const beat = Math.floor((step % 16) / 4) + 1;
      const subdivision = step % 4;
      this.dom.clock.textContent = `${bar}.${beat}`;
      const progress = ((beat - 1) * 4 + subdivision + 1) / 16 * 100;
      this.dom.beatHand.style.width = `${progress}%`;
      this.visual.setStep(step);
    }

    renderProfile() {
      if (!this.dom) return;
      const profile = this.globalProfile || profileDefault();
      const movement = profile.motion < 0.38 ? "舒展" : profile.motion > 0.68 ? "跃动" : "流动";
      const complexity = this.layers ? clamp(this.layers.reduce((sum, layer) => sum + Math.max(0, layer.generation), 0) / 12) : 0;
      const legato = clamp(profile.hold / 1400);
      const activeLayers = this.layers ? this.layers.filter(layer => layer.active || layer.pending).length : 0;
      this.dom.personality.textContent = profile.samples ? movement : "等待你的动作";
      this.dom.profileMotion.textContent = profile.motion.toFixed(2);
      this.dom.profileLegato.textContent = legato.toFixed(2);
      this.dom.profileLayers.textContent = `${activeLayers} / 6`;
      this.dom.profileComplexity.textContent = complexity.toFixed(2);
    }

    log(message) {
      this.dom.eventLog.textContent = message;
    }

    async runAutoDemo() {
      if (this.autoRunning) return;
      this.autoRunning = true;
      this.dom.auto.disabled = true;
      try {
        if (!this.playing) await this.toggleSession();
        const sequence = [
          ["forward", 58], ["down", 65], ["left", 42],
          ["right", 72], ["up", 78], ["back", 32],
          ["right", 88], ["forward", 76]
        ];
        for (const [gesture, motion] of sequence) {
          this.dom.motion.value = motion;
          this.dom.motionValue.value = motion;
          this.beginGesture(gesture);
          await delay(360);
          this.endGesture(gesture);
          await delay(760);
        }
        this.log("自动演示完成。现在可以继续挥动，或点击结束作品。");
      } finally {
        this.autoRunning = false;
        this.dom.auto.disabled = false;
      }
    }
  }

  window.jilvDemo = new JilvDemo();
})();
