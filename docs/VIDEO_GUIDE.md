# Recorded-video component

Reusable video-focused presentation for verified task recordings. These component files require page integration and a recording manifest. Keep source provenance, configuration, playback speed, and outcomes grounded in the supplied records.

Once actual recordings and their provenance are supplied:

1. Add `video.css` after `site.css`, and `video.js` after `site.js` with `defer`.
2. Add an appropriately titled section containing `<div id="video-experience" data-manifest="videos.json"><p class="video-loading" role="status">Loading recorded demonstrations…</p></div>`.
3. Supply `videos.json` with this structure. Values below are field descriptions, **not real media or outcome records**:

```text
{
  "title": "Collection title",
  "context": "Verified setting, policy configuration, and capture context",
  "sourceUrl": "Collection provenance URL (optional)",
  "note": "Collection limitations (optional; a selection-limit note is shown by default)",
  "featuredIds": ["IDs of clips to show in the featured playlist (optional)"],
  "clips": [{
    "id": "Unique stable identifier",
    "title": "Verified task description",
    "backend": "Verified policy name (optional)",
    "category": "Gallery filter category (optional; falls back to backend)",
    "suite": "Verified task suite (optional)",
    "outcome": "success | failure | unknown",
    "outcomeLabel": "Optional evidence-backed outcome label",
    "playbackLabel": "Optional verified speed label, such as Original speed",
    "src": "Actual video URL or relative path",
    "poster": "Actual still from this video (optional)",
    "width": "Actual pixel width as a positive number (optional)",
    "height": "Actual pixel height as a positive number (optional)",
    "description": "Evidence-backed recording context (optional)",
    "sourceUrl": "Original recording provenance URL (optional)"
  }]
}
```

Optional hero collage: add `<div id="hero-reel" class="hero-reel" aria-label="Recorded task previews"></div>` beside the hero copy. It shows the first three verified `featuredIds` (or the first three clips if no IDs are supplied), with one wide player and two smaller players. One- and two-clip collections adapt automatically. The hero uses the same media manager as the main gallery; both global pause buttons remain synchronized and control all videos. The outer hero grid remains the page's responsibility. Keep the hero paired with the gallery so the collection context and recording provenance remain readily available. `playbackLabel` is displayed alongside clip metadata without altering playback speed.

Prefer browser-compatible H.264 MP4 (with fast-start metadata), and real frame posters. Keep task labels, outcomes, playback speed, and configuration faithful to the original recordings. Omit unknown fields; missing outcomes are shown as unverified. Do not convert still images to videos and label them as rollouts.

The component provides native controls, a featured playlist, category filters (falling back to backend), a responsive three-column gallery, source links, a global pause button, and muted loops. Supplied positive `width` and `height` values preserve the recording's aspect ratio, including dual-camera layouts. It hydrates video sources only when entering view. Videos pause offscreen and when the tab is hidden. Reduced-motion and data-saving preferences disable autoplay initially; users may explicitly opt in. Manual pause remains respected until the user resumes playback. Browser autoplay rejection leaves native controls available.

Before publication, validate each asset URL, source provenance, natural dimensions, native playback, visible-only autoplay, global pause, filters, playlist changes, keyboard controls, reduced-motion behavior, and mobile layout using the supplied recordings.
