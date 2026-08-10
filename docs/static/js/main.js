// ---- RAMI project page interactions ----------------------------------------
document.addEventListener('DOMContentLoaded', function () {

  /* 0. Global playback speed — the flythroughs are short; slow them down so
        viewers can read the difference. */
  const PLAYBACK_RATE = 0.5;
  function setRate(v) { try { v.playbackRate = PLAYBACK_RATE; } catch (e) {} }
  document.querySelectorAll('video').forEach(v => {
    setRate(v);
    v.addEventListener('loadedmetadata', () => setRate(v));
    v.addEventListener('play', () => setRate(v));   // some browsers reset on (re)play
  });

  /* 1. Viewport-aware autoplay for all videos (save CPU/bandwidth offscreen). */
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        const v = e.target;
        if (e.isIntersecting) { v.play().catch(() => {}); } else { v.pause(); }
      });
    }, { threshold: 0.15 });
    document.querySelectorAll('video').forEach(v => io.observe(v));
  }

  /* 2. Drag-to-compare wipe — works for both <img> and <video> layers. */
  function playWipeVideos(el) {
    const vids = el.querySelectorAll('video');
    if (!vids.length) return;
    vids.forEach(v => v.play().catch(() => {}));
  }
  function initVCompare(el) {
    const before = el.querySelector('.vc-before');   // clipped top layer (left)
    const after  = el.querySelector('.vc-after');    // full background (right)
    const divider = el.querySelector('.vc-divider');
    let pos = parseFloat(el.dataset.start || '50');

    function setPos(p) {
      pos = Math.max(0, Math.min(100, p));
      before.style.clipPath = `inset(0 ${100 - pos}% 0 0)`;
      divider.style.left = pos + '%';
    }
    setPos(pos);

    function fromEvent(ev) {
      const r = el.getBoundingClientRect();
      const x = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
      setPos((x / r.width) * 100);
    }
    let dragging = false;
    const down = (ev) => { dragging = true; fromEvent(ev); ev.preventDefault(); };
    const move = (ev) => { if (dragging) fromEvent(ev); };
    const up   = () => { dragging = false; };
    el.addEventListener('mousedown', down);
    el.addEventListener('touchstart', down, { passive: false });
    window.addEventListener('mousemove', move);
    window.addEventListener('touchmove', move, { passive: false });
    window.addEventListener('mouseup', up);
    window.addEventListener('touchend', up);
    el.addEventListener('mousemove', (ev) => { if (!dragging) fromEvent(ev); });

    // video layers only: keep the two clips frame-synced
    if (before.tagName === 'VIDEO' && after.tagName === 'VIDEO') {
      function sync() {
        if (before.readyState >= 2 && after.readyState >= 2 &&
            Math.abs(before.currentTime - after.currentTime) > 0.06) {
          before.currentTime = after.currentTime;
        }
        requestAnimationFrame(sync);
      }
      requestAnimationFrame(sync);
    }
  }
  document.querySelectorAll('.vcompare').forEach(initVCompare);

  /* 3. Still / Video mode toggle (all scenes shown as rows). */
  function inViewport(el) {
    const r = el.getBoundingClientRect();
    return r.bottom > 0 && r.top < (window.innerHeight || document.documentElement.clientHeight);
  }
  document.querySelectorAll('[data-tabgroup]').forEach(group => {
    const modeBtns = group.querySelectorAll('.mode-toggle button');
    modeBtns.forEach(btn => btn.addEventListener('click', () => {
      modeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.dataset.mode;          // 'still' | 'video'
      group.classList.toggle('show-video', mode === 'video');
      group.classList.toggle('show-still', mode === 'still');
      if (mode === 'video') {
        // only kick off videos currently on screen; the IntersectionObserver
        // starts/pauses the rest as the user scrolls (avoids loading all at once)
        group.querySelectorAll('.video-wipe').forEach(w => { if (inViewport(w)) playWipeVideos(w); });
      }
    }));
  });

  /* 5. Scroll-reveal. */
  if ('IntersectionObserver' in window) {
    const ro = new IntersectionObserver((entries, obs) => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); obs.unobserve(e.target); } });
    }, { threshold: 0.12 });
    document.querySelectorAll('.reveal').forEach(el => ro.observe(el));
  } else {
    document.querySelectorAll('.reveal').forEach(el => el.classList.add('in'));
  }
});
