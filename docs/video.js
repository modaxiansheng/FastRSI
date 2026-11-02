"use strict";

/* See VIDEO_GUIDE.md for integration and the verified-recording manifest schema. */
(() => {
  const root = document.getElementById("video-experience");
  const heroRoot = document.getElementById("hero-reel");
  if (!root && !heroRoot) return;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const connection = navigator.connection;
  let autoplayPaused = reducedMotion.matches || Boolean(connection && connection.saveData);
  const media = new Map();
  const motionButtons = [];
  const motionStatuses = [];

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  function safeUrl(value) {
    if (typeof value !== "string" || !value.trim()) return null;
    try {
      const url = new URL(value, document.baseURI);
      return /^https?:$/.test(url.protocol) ? url.href : null;
    } catch { return null; }
  }
  function link(value, label) {
    const href = safeUrl(value);
    if (!href) return null;
    const node = el("a", "video-source-link", label);
    node.href = href;
    node.target = "_blank";
    node.rel = "noopener noreferrer";
    return node;
  }
  function pause(video, state) {
    if (!video.paused) {
      state.managedPause = true;
      video.pause();
    }
  }
  function hydrate(video, state) {
    if (state.hydrated) return;
    video.src = state.src;
    state.hydrated = true;
  }
  function eligible(video, state) {
    return state.visible && !video.closest("[hidden]") && !document.hidden;
  }
  function sync(video, state) {
    if (!eligible(video, state)) {
      pause(video, state);
      return;
    }
    if (autoplayPaused || state.userPaused || state.failed || state.pending) return;
    hydrate(video, state);
    if (!video.paused) return;
    state.pending = true;
    const attempt = video.play();
    if (attempt && typeof attempt.then === "function") {
      attempt.then(() => {
        if (!eligible(video, state) || autoplayPaused) pause(video, state);
      }).catch(() => {
        // Native controls remain usable when a browser declines autoplay.
      }).finally(() => { state.pending = false; });
    } else state.pending = false;
  }
  const observer = "IntersectionObserver" in window ? new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const state = media.get(entry.target);
      if (!state) return;
      state.visible = entry.isIntersecting && entry.intersectionRatio >= .35;
      if (entry.isIntersecting) hydrate(entry.target, state);
      sync(entry.target, state);
    });
  }, { threshold: [0, .35, .75] }) : null;
  function player(clip) {
    const wrap = el("div", "video-player-wrap");
    const video = el("video", "video-player");
    video.controls = true;
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.loop = true;
    video.preload = "none";
    video.setAttribute("aria-label", clip.title);
    const width = Number(clip.width);
    const height = Number(clip.height);
    if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
      video.style.aspectRatio = `${width} / ${height}`;
    }
    const poster = safeUrl(clip.poster);
    if (poster) video.poster = poster;
    const error = el("p", "video-error", "This recording could not load. Use its source link below to inspect the original.");
    error.hidden = true;
    error.setAttribute("role", "status");
    const state = { src: safeUrl(clip.src), visible: false, hydrated: false, userPaused: false, managedPause: false, failed: false, pending: false };
    media.set(video, state);
    video.addEventListener("pause", () => {
      if (state.managedPause) state.managedPause = false;
      else state.userPaused = true;
    });
    video.addEventListener("play", () => { state.userPaused = false; });
    video.addEventListener("pointerdown", () => hydrate(video, state), { passive: true });
    video.addEventListener("keydown", () => hydrate(video, state));
    video.addEventListener("error", () => {
      state.failed = true;
      error.hidden = false;
    });
    wrap.append(video, error);
    if (observer) observer.observe(video);
    else hydrate(video, state); // Native playback only without visibility detection.
    return wrap;
  }
  function unregister(container) {
    container.querySelectorAll("video").forEach(video => {
      const state = media.get(video);
      if (state) pause(video, state);
      if (observer) observer.unobserve(video);
      media.delete(video);
      video.removeAttribute("src");
      video.load();
    });
  }
  function metadata(clip) {
    const meta = el("div", "video-clip-meta");
    [clip.backend, clip.suite, clip.playbackLabel].filter(Boolean).forEach(value => meta.append(el("span", "video-meta-tag", value)));
    const outcome = ["success", "failure"].includes(clip.outcome) ? clip.outcome : "unknown";
    const labels = { success: "Recorded success", failure: "Recorded failure", unknown: "Outcome unverified" };
    const badge = el("span", "video-meta-tag video-outcome", clip.outcomeLabel || labels[outcome]);
    badge.dataset.outcome = outcome;
    meta.append(badge);
    return meta;
  }
  function details(clip, featured) {
    const panel = el("div", featured ? "video-feature-details" : "video-card-details");
    panel.append(metadata(clip), el("h3", featured ? "video-feature-title" : "video-card-title", clip.title));
    if (clip.description) panel.append(el("p", "video-clip-description", clip.description));
    const source = link(clip.sourceUrl || clip.src, "Recording source ↗");
    if (source) panel.append(source);
    return panel;
  }
  function syncMotionControls() {
    motionButtons.forEach(button => {
      button.setAttribute("aria-pressed", String(autoplayPaused));
      button.textContent = autoplayPaused ? "▷ Play visible videos" : "Ⅱ Pause videos";
    });
    motionStatuses.forEach(status => {
      status.textContent = autoplayPaused ? "Autoplay paused · individual controls remain available" : "Muted loops · paused when out of view";
    });
  }
  function motionControl() {
    const button = el("button", "video-motion-button");
    button.type = "button";
    motionButtons.push(button);
    button.addEventListener("click", () => {
      autoplayPaused = !autoplayPaused;
      syncMotionControls();
      media.forEach((state, video) => {
        if (autoplayPaused) pause(video, state);
        else { state.userPaused = false; sync(video, state); }
      });
    });
    syncMotionControls();
    return button;
  }
  function featuredClips(data, clips) {
    const selected = Array.isArray(data.featuredIds) ? data.featuredIds.map(id => clips.find(clip => clip.id === id)).filter(Boolean) : [];
    return selected.length ? [...new Set(selected)] : clips.slice(0, 4);
  }
  function renderHero(data, clips) {
    if (!heroRoot) return;
    unregister(heroRoot);
    heroRoot.replaceChildren();
    heroRoot.hidden = !clips.length;
    if (!clips.length) return;
    const selected = featuredClips(data, clips).slice(0, 3);
    heroRoot.dataset.count = String(selected.length);
    const toolbar = el("div", "hero-reel-toolbar");
    toolbar.append(el("span", "hero-reel-kicker", "LIBERO · development recordings"), motionControl());
    heroRoot.append(toolbar);
    selected.forEach((clip, index) => {
      const tile = el("article", "hero-reel-tile");
      const caption = el("div", "hero-reel-caption");
      const label = el("div", "hero-reel-label");
      label.append(el("span", "hero-reel-number", String(index + 1).padStart(2, "0")), el("h3", "hero-reel-title", clip.title));
      caption.append(label);
      const meta = [clip.backend, clip.suite, clip.playbackLabel].filter(Boolean).join(" · ");
      if (meta) caption.append(el("span", "hero-reel-meta", meta));
      tile.append(player(clip), caption);
      heroRoot.append(tile);
    });
  }
  function render(data) {
    const ids = new Set();
    const clips = Array.isArray(data.clips) ? data.clips.filter(clip => {
      if (!clip || !clip.id || !clip.title || !safeUrl(clip.src) || ids.has(clip.id)) return false;
      ids.add(clip.id);
      return true;
    }) : [];
    renderHero(data, clips);
    if (!root) return;
    root.replaceChildren();
    if (!clips.length) {
      root.append(el("p", "video-empty", "No recorded demonstrations are available in this collection yet."));
      return;
    }
    const header = el("div", "video-collection-header");
    const intro = el("div");
    intro.append(el("h3", "video-collection-title", data.title || "Recorded demonstrations"));
    if (data.context) intro.append(el("p", "video-collection-context", data.context));
    const controls = el("div", "video-controls");
    const motionButton = motionControl();
    const motionStatus = el("span", "video-motion-status");
    motionStatus.setAttribute("role", "status");
    motionStatuses.push(motionStatus);
    syncMotionControls();
    controls.append(motionButton, motionStatus);
    header.append(intro, controls);
    const feature = el("div", "video-feature");
    const stage = el("div", "video-feature-player");
    const playlist = el("div", "video-playlist");
    playlist.setAttribute("role", "group");
    playlist.setAttribute("aria-label", "Featured recorded demonstrations");
    playlist.append(el("p", "video-playlist-label", "Featured recordings"));
    const featured = featuredClips(data, clips);
    const buttons = [];
    function select(clip) {
      unregister(stage);
      stage.replaceChildren(player(clip), details(clip, true));
      buttons.forEach(({ button, id }) => button.setAttribute("aria-pressed", String(id === clip.id)));
    }
    featured.forEach(clip => {
      const button = el("button", "video-playlist-button");
      button.type = "button";
      button.setAttribute("aria-label", `Feature recording: ${clip.title}`);
      const poster = safeUrl(clip.poster);
      const thumbnail = el(poster ? "img" : "span", "video-playlist-thumb", poster ? undefined : "▷");
      if (poster) { thumbnail.src = poster; thumbnail.alt = ""; thumbnail.loading = "lazy"; }
      else thumbnail.setAttribute("aria-hidden", "true");
      const text = el("span", "video-playlist-text");
      text.append(el("strong", "", clip.title), el("small", "", [clip.backend, clip.suite].filter(Boolean).join(" · ")));
      const arrow = el("span", "video-playlist-arrow", "↗");
      arrow.setAttribute("aria-hidden", "true");
      button.append(thumbnail, text, arrow);
      button.addEventListener("click", () => select(clip));
      buttons.push({ button, id: clip.id });
      playlist.append(button);
    });
    select(featured[0]);
    feature.append(stage, playlist);
    const toolbar = el("div", "video-gallery-toolbar");
    const filters = el("div", "video-filters");
    filters.setAttribute("role", "group");
    filters.setAttribute("aria-label", "Filter recordings");
    const count = el("span", "video-gallery-count");
    count.setAttribute("role", "status");
    const grid = el("div", "video-grid");
    const cards = clips.map(clip => {
      const card = el("article", "video-card");
      card.append(player(clip), details(clip, false));
      grid.append(card);
      return { card, clip };
    });
    const filterButtons = [];
    const categoryOf = clip => clip.category || clip.backend;
    const categories = [...new Set(clips.map(categoryOf).filter(Boolean))];
    [null, ...categories].forEach(category => {
      const button = el("button", "video-filter", category || "All recordings");
      button.type = "button";
      button.setAttribute("aria-pressed", String(category === null));
      button.addEventListener("click", () => {
        filterButtons.forEach(other => other.setAttribute("aria-pressed", String(other === button)));
        cards.forEach(({ card, clip }) => {
          card.hidden = Boolean(category && categoryOf(clip) !== category);
          if (card.hidden) card.querySelectorAll("video").forEach(video => pause(video, media.get(video)));
        });
        count.textContent = `${cards.filter(({ card }) => !card.hidden).length} / ${clips.length} recordings`;
      });
      filterButtons.push(button);
      filters.append(button);
    });
    count.textContent = `${clips.length} recordings`;
    toolbar.append(filters, count);
    const footer = el("div", "video-collection-footer");
    footer.append(el("p", "", data.note || "Individual recordings illustrate the labeled runs. Their selection does not establish aggregate performance or a comparison across systems."));
    const collectionSource = link(data.sourceUrl, "Collection source ↗");
    if (collectionSource) footer.append(collectionSource);
    root.append(header, feature, toolbar, grid, footer);
  }
  document.addEventListener("visibilitychange", () => media.forEach((state, video) => sync(video, state)));
  function preferenceChanged() {
    if (reducedMotion.matches || (connection && connection.saveData)) {
      autoplayPaused = true;
      syncMotionControls();
      media.forEach((state, video) => pause(video, state));
    }
  }
  reducedMotion.addEventListener("change", preferenceChanged);
  if (connection && connection.addEventListener) connection.addEventListener("change", preferenceChanged);
  fetch((root && root.dataset.manifest) || (heroRoot && heroRoot.dataset.manifest) || "videos.json")
    .then(response => {
      if (!response.ok) throw new Error("Recording manifest unavailable");
      return response.json();
    })
    .then(render)
    .catch(() => {
      if (root) root.replaceChildren(el("p", "video-empty", "The recording collection could not be loaded. Please try again later."));
      if (heroRoot) heroRoot.hidden = true;
    });
})();
