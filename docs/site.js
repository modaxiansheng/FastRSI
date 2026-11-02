"use strict";

(() => {
  const menuButton = document.querySelector(".menu-toggle");
  const navigation = document.getElementById("primary-nav");
  const closeMenu = () => {
    menuButton.setAttribute("aria-expanded", "false");
    menuButton.setAttribute("aria-label", "Open navigation");
    navigation.classList.remove("open");
  };
  menuButton.addEventListener("click", () => {
    const expanded = menuButton.getAttribute("aria-expanded") === "true";
    menuButton.setAttribute("aria-expanded", String(!expanded));
    menuButton.setAttribute("aria-label", expanded ? "Open navigation" : "Close navigation");
    navigation.classList.toggle("open", !expanded);
  });
  navigation.querySelectorAll("a").forEach(link => link.addEventListener("click", closeMenu));
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && navigation.classList.contains("open")) {
      closeMenu();
      menuButton.focus();
    }
  });

  function bindTabs(selector, onSelect) {
    const tabs = Array.from(document.querySelectorAll(selector));
    const select = (tab, moveFocus = false) => {
      tabs.forEach(item => {
        const active = item === tab;
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", String(active));
        item.tabIndex = active ? 0 : -1;
      });
      onSelect(tab);
      if (moveFocus) tab.focus();
    };
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => select(tab));
      tab.addEventListener("keydown", event => {
        let next;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + tabs.length) % tabs.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = tabs.length - 1;
        if (next !== undefined) {
          event.preventDefault();
          select(tabs[next], true);
        }
      });
    });
    return select;
  }

  bindTabs(".method-tab", tab => {
    document.querySelectorAll(".method-panel").forEach(panel => {
      panel.hidden = panel.id !== tab.getAttribute("aria-controls");
    });
  });

  const sequenceStage = document.querySelector(".sequence-stage");
  sequenceStage.id = "sequence-stage-panel";
  sequenceStage.setAttribute("role", "tabpanel");
  sequenceStage.setAttribute("aria-labelledby", "sequence-tab-1");
  const sequenceImage = document.getElementById("sequence-image");
  const sequenceExpand = document.getElementById("sequence-expand");
  bindTabs(".sequence-step", tab => {
    const { step, title, alt } = tab.dataset;
    sequenceImage.src = `step-${step}.png`;
    sequenceImage.alt = alt;
    sequenceExpand.dataset.lightbox = `step-${step}.png`;
    sequenceExpand.dataset.caption = `Step ${step}: ${title}. Illustrative task sequence from project figures.`;
    document.getElementById("sequence-title").textContent = title;
    document.getElementById("sequence-count").textContent = `0${step} / 04`;
    sequenceStage.setAttribute("aria-labelledby", tab.id);
  });

  const dialog = document.getElementById("figure-dialog");
  const dialogImage = document.getElementById("dialog-image");
  const dialogCaption = document.getElementById("dialog-caption");
  let dialogTrigger = null;
  document.querySelectorAll("[data-lightbox]").forEach(button => {
    button.addEventListener("click", () => {
      dialogTrigger = button;
      dialogImage.src = button.dataset.lightbox;
      dialogImage.alt = button.dataset.caption;
      dialogCaption.textContent = button.dataset.caption;
      dialog.showModal();
      document.body.classList.add("dialog-open");
      document.getElementById("close-dialog").focus();
    });
  });
  document.getElementById("close-dialog").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
  dialog.addEventListener("close", () => {
    document.body.classList.remove("dialog-open");
    if (dialogTrigger) dialogTrigger.focus();
  });

  let snapshot;
  let selectedSuite = "spatial";
  const resultRows = document.getElementById("result-rows");
  const resultPanel = document.getElementById("results-panel");
  const formatCount = value => Number(value).toLocaleString("en-US");
  const backendMarks = { openvla: "OV", pi0: "π₀", pi05: "π₀.₅", "pi0.5": "π₀.₅", xvla: "XV" };

  function renderResults() {
    if (!snapshot) return;
    const groups = snapshot.groups.filter(group => group.suite.toLowerCase() === selectedSuite);
    resultRows.replaceChildren();
    if (groups.length === 0) {
      const message = document.createElement("p");
      message.className = "results-loading";
      message.textContent = "This suite has no records in the released snapshot. Inspect the source data below.";
      resultRows.append(message);
      return;
    }
    groups.forEach(group => {
      const complete = Number(group.completed);
      const expected = Number(group.expected);
      const successes = Number(group.successes);
      const rate = complete > 0 ? successes / complete * 100 : 0;
      const row = document.createElement("div");
      row.className = "result-row";
      row.setAttribute("role", "group");
      const label = document.createElement("div");
      label.className = "backend-label";
      const mark = document.createElement("span");
      mark.className = "backend-mark";
      mark.setAttribute("aria-hidden", "true");
      mark.textContent = backendMarks[group.key.toLowerCase()] || group.backend.slice(0, 2);
      const nameWrap = document.createElement("div");
      const name = document.createElement("span");
      name.className = "backend-name";
      name.textContent = group.backend;
      nameWrap.append(name);
      if (complete < expected) {
        const incomplete = document.createElement("span");
        incomplete.className = "backend-incomplete";
        incomplete.textContent = `${formatCount(expected - complete)} episodes pending`;
        nameWrap.append(incomplete);
      }
      label.append(mark, nameWrap);
      const count = document.createElement("div");
      count.className = "result-count";
      count.textContent = `${formatCount(successes)} / ${formatCount(complete)}`;
      const completion = document.createElement("small");
      completion.textContent = `${formatCount(complete)} / ${formatCount(expected)} episodes`;
      count.append(completion);
      const rateElement = document.createElement("div");
      rateElement.className = "result-rate";
      rateElement.textContent = `${rate.toFixed(1)}%`;
      const track = document.createElement("div");
      track.className = "result-bar-track";
      track.setAttribute("aria-hidden", "true");
      const bar = document.createElement("span");
      bar.className = "result-bar";
      bar.style.width = `${rate}%`;
      track.append(bar);
      rateElement.append(track);
      row.append(label, count, rateElement);
      row.setAttribute("aria-label", `${group.backend}: ${rate.toFixed(1)} percent success; ${successes} successful out of ${complete} completed episodes; ${expected} expected.`);
      resultRows.append(row);
    });
  }

  bindTabs(".suite-tabs button", tab => {
    selectedSuite = tab.dataset.suite;
    resultPanel.setAttribute("aria-labelledby", tab.id);
    renderResults();
  });

  fetch("results.json")
    .then(response => {
      if (!response.ok) throw new Error(`Snapshot request failed: ${response.status}`);
      return response.json();
    })
    .then(data => {
      if (!Array.isArray(data.groups)) throw new Error("Snapshot does not contain evaluation groups");
      snapshot = data;
      renderResults();
    })
    .catch(() => {
      resultRows.replaceChildren();
      const message = document.createElement("p");
      message.className = "results-loading";
      message.textContent = "The snapshot could not be loaded. Open the source data below to inspect the released evaluation records.";
      resultRows.append(message);
    })
    .finally(() => resultRows.setAttribute("aria-busy", "false"));

  const copyButton = document.getElementById("copy-command");
  const copyStatus = document.getElementById("copy-status");
  let copyTimeout;
  copyButton.addEventListener("click", async () => {
    const command = document.getElementById("setup-command").textContent;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(command);
      } else {
        const field = document.createElement("textarea");
        field.value = command;
        field.style.position = "fixed";
        field.style.opacity = "0";
        document.body.append(field);
        field.select();
        const copied = document.execCommand("copy");
        field.remove();
        copyButton.focus();
        if (!copied) throw new Error("Clipboard copy unavailable");
      }
      copyButton.querySelector("span").textContent = "Copied";
      copyStatus.textContent = "Setup commands copied to clipboard.";
    } catch {
      copyStatus.textContent = "Copy is unavailable in this browser. Select the commands above to copy them.";
    }
    window.clearTimeout(copyTimeout);
    copyTimeout = window.setTimeout(() => {
      copyButton.querySelector("span").textContent = "Copy";
      copyStatus.textContent = "";
    }, 5000);
  });

  const canvas = document.getElementById("hero-particles");
  const context = canvas.getContext("2d");
  if (!context) return;
  const hero = document.querySelector(".hero");
  const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  const motionButton = document.getElementById("motion-toggle");
  let motionPaused = motionQuery.matches;
  function syncMotionButton() {
    motionButton.setAttribute("aria-pressed", String(motionPaused));
    motionButton.querySelector("span:last-child").textContent = motionPaused ? "Resume motion" : "Pause motion";
    motionButton.querySelector(".motion-icon").textContent = motionPaused ? "▷" : "Ⅱ";
  }
  syncMotionButton();
  let width = 0;
  let height = 0;
  let particles = [];
  let animationId;
  let visible = true;
  let lastTime = 0;
  const pointer = { x: -1000, y: -1000 };

  function resizeCanvas() {
    const bounds = hero.getBoundingClientRect();
    width = bounds.width;
    height = bounds.height;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    const count = Math.min(85, Math.max(32, Math.round(width / 18)));
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - .5) * .11,
      vy: (Math.random() - .5) * .11,
      radius: Math.random() * 1.2 + .35,
      opacity: Math.random() * .45 + .15
    }));
    if (motionPaused) drawFrame(0);
  }

  function drawFrame(time) {
    const step = lastTime ? Math.min((time - lastTime) / 16.67, 2) : 1;
    lastTime = time;
    context.clearRect(0, 0, width, height);
    particles.forEach((particle, index) => {
      if (!motionPaused) {
        particle.x += particle.vx * step;
        particle.y += particle.vy * step;
        if (particle.x < -5) particle.x = width + 5;
        if (particle.x > width + 5) particle.x = -5;
        if (particle.y < -5) particle.y = height + 5;
        if (particle.y > height + 5) particle.y = -5;
      }
      const distanceToPointer = Math.hypot(particle.x - pointer.x, particle.y - pointer.y);
      const glow = Math.max(0, 1 - distanceToPointer / 190);
      context.beginPath();
      context.arc(particle.x, particle.y, particle.radius + glow * .75, 0, Math.PI * 2);
      context.fillStyle = `rgba(174, 160, 240, ${Math.min(particle.opacity + glow * .4, .9)})`;
      context.fill();
      particles.slice(index + 1).forEach(other => {
        const distance = Math.hypot(particle.x - other.x, particle.y - other.y);
        if (distance > 130) return;
        context.beginPath();
        context.moveTo(particle.x, particle.y);
        context.lineTo(other.x, other.y);
        context.strokeStyle = `rgba(128, 111, 186, ${(1 - distance / 130) * (.13 + glow * .12)})`;
        context.lineWidth = .5;
        context.stroke();
      });
    });
    if (!motionPaused && visible && !document.hidden) animationId = requestAnimationFrame(drawFrame);
  }

  function updateAnimation() {
    cancelAnimationFrame(animationId);
    lastTime = 0;
    if (visible && !document.hidden) drawFrame(0);
  }
  hero.addEventListener("pointermove", event => {
    const bounds = hero.getBoundingClientRect();
    pointer.x = event.clientX - bounds.left;
    pointer.y = event.clientY - bounds.top;
  }, { passive: true });
  hero.addEventListener("pointerleave", () => { pointer.x = -1000; pointer.y = -1000; });
  new ResizeObserver(resizeCanvas).observe(hero);
  new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    updateAnimation();
  }, { threshold: 0 }).observe(hero);
  motionButton.addEventListener("click", () => {
    motionPaused = !motionPaused;
    syncMotionButton();
    updateAnimation();
  });
  motionQuery.addEventListener("change", () => {
    motionPaused = motionQuery.matches;
    syncMotionButton();
    updateAnimation();
  });
  document.addEventListener("visibilitychange", updateAnimation);
  resizeCanvas();
})();
