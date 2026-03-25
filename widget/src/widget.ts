/**
 * Athena DevDocs Widget
 *
 * Embeddable search widget for documentation sites.
 * Uses Shadow DOM for complete style isolation.
 *
 * Usage:
 *   <script src="widget.js" data-api-key="pk_..." async></script>
 */

import type { WidgetConfig, SourceChunk } from "./types";

declare const __WIDGET_CSS__: string;

// ── Config ──────────────────────────────────────────────────────────────

function getConfig(): WidgetConfig {
  const script =
    document.currentScript ??
    document.querySelector("script[data-api-key]");

  return {
    apiKey: script?.getAttribute("data-api-key") ?? "",
    apiUrl: script?.getAttribute("data-api-url") ?? "https://api.devdocsai.com",
    theme: (script?.getAttribute("data-theme") as "light" | "dark") ?? "light",
    accentColor: script?.getAttribute("data-accent-color") ?? "#6366f1",
    position:
      (script?.getAttribute("data-position") as "bottom-right" | "bottom-left") ??
      "bottom-right",
    placeholder: script?.getAttribute("data-placeholder") ?? "Search docs…",
  };
}

// ── Widget ──────────────────────────────────────────────────────────────

class AthenaWidget extends HTMLElement {
  private shadow: ShadowRoot;
  private config: WidgetConfig;
  private isOpen = false;
  private abortController: AbortController | null = null;

  constructor() {
    super();
    this.config = getConfig();
    this.shadow = this.attachShadow({ mode: "open" });

    // Inject CSS
    const style = document.createElement("style");
    style.textContent = __WIDGET_CSS__;
    this.shadow.appendChild(style);

    // Apply theme
    if (this.config.theme === "dark") {
      this.setAttribute("data-theme", "dark");
    }

    // Apply accent color override
    if (this.config.accentColor !== "#6366f1") {
      this.style.setProperty("--dd-primary", this.config.accentColor);
    }

    this.render();
    this.bindKeyboard();
  }

  private render() {
    // Trigger button
    const trigger = document.createElement("button");
    trigger.className = "dd-trigger";
    if (this.config.position === "bottom-left") {
      trigger.style.left = "20px";
      trigger.style.right = "auto";
    }
    trigger.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>`;
    trigger.addEventListener("click", () => this.open());
    this.shadow.appendChild(trigger);
  }

  private open() {
    if (this.isOpen) return;
    this.isOpen = true;

    const backdrop = document.createElement("div");
    backdrop.className = "dd-modal-backdrop";
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) this.close();
    });

    const modal = document.createElement("div");
    modal.className = "dd-modal";

    // Search bar
    const bar = document.createElement("div");
    bar.className = "dd-search-bar";

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = this.config.placeholder;
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && input.value.trim()) {
        this.search(input.value.trim(), results);
      }
      if (e.key === "Escape") this.close();
    });

    const kbd = document.createElement("kbd");
    kbd.textContent = "ESC";

    bar.appendChild(input);
    bar.appendChild(kbd);

    // Results container
    const results = document.createElement("div");
    results.className = "dd-results";
    results.innerHTML = `<div class="dd-empty">Type a question and press Enter</div>`;

    modal.appendChild(bar);
    modal.appendChild(results);
    backdrop.appendChild(modal);
    this.shadow.appendChild(backdrop);

    requestAnimationFrame(() => input.focus());
  }

  private close() {
    this.isOpen = false;
    this.abortController?.abort();
    const backdrop = this.shadow.querySelector(".dd-modal-backdrop");
    if (backdrop) backdrop.remove();
  }

  private bindKeyboard() {
    document.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (this.isOpen) this.close();
        else this.open();
      }
    });
  }

  private async search(query: string, container: HTMLElement) {
    this.abortController?.abort();
    this.abortController = new AbortController();

    container.innerHTML = `
      <div class="dd-loading">
        <div class="dd-loading-dot"></div>
        <div class="dd-loading-dot"></div>
        <div class="dd-loading-dot"></div>
        <span>Searching…</span>
      </div>`;

    try {
      const response = await fetch(`${this.config.apiUrl}/api/widget/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Widget-Key": this.config.apiKey,
        },
        body: JSON.stringify({ query }),
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Request failed" }));
        container.innerHTML = `<div class="dd-empty">${err.detail ?? "Something went wrong"}</div>`;
        return;
      }

      const data = await response.json();
      this.renderAnswer(container, data, query);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      container.innerHTML = `<div class="dd-empty">Failed to connect</div>`;
    }
  }

  private renderAnswer(
    container: HTMLElement,
    data: { answer: string; sources: SourceChunk[]; verified?: boolean; confidence?: number; query_id?: string },
    _query: string,
  ) {
    let html = `<div class="dd-answer">${this.renderMarkdown(data.answer)}</div>`;

    if (data.verified) {
      html += `<div class="dd-verified">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>
        Verified against ${data.sources?.length ?? 0} sources
      </div>`;
    }

    if (data.sources?.length) {
      html += `<div class="dd-sources"><div class="dd-sources-title">Sources</div>`;
      for (const src of data.sources) {
        const href = src.url ? ` href="${src.url}" target="_blank" rel="noopener"` : "";
        const tag = src.url ? "a" : "span";
        html += `<${tag} class="dd-source"${href}>${src.title || src.snippet?.slice(0, 80) || "Source"}</${tag}>`;
      }
      html += `</div>`;
    }

    html += `<div class="dd-feedback">
      <button data-vote="up">👍 Helpful</button>
      <button data-vote="down">👎 Not helpful</button>
    </div>`;

    container.innerHTML = html;

    // Bind feedback
    container.querySelectorAll(".dd-feedback button").forEach((btn) => {
      btn.addEventListener("click", () => {
        const vote = (btn as HTMLElement).dataset.vote;
        this.sendFeedback(data.query_id ?? "", vote as string);
        const parent = btn.parentElement!;
        parent.innerHTML = `<span style="color: var(--dd-text-secondary); font-size: 13px;">Thanks for your feedback!</span>`;
      });
    });
  }

  private renderMarkdown(text: string): string {
    // Minimal markdown → HTML for code blocks and paragraphs
    return text
      .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\n\n/g, "</p><p>")
      .replace(/^/, "<p>")
      .replace(/$/, "</p>");
  }

  private async sendFeedback(queryId: string, rating: string) {
    try {
      await fetch(`${this.config.apiUrl}/api/widget/feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Widget-Key": this.config.apiKey,
        },
        body: JSON.stringify({ query_id: queryId, rating }),
      });
    } catch {
      // Best-effort, don't block UI
    }
  }
}

// ── Bootstrap ───────────────────────────────────────────────────────────

customElements.define("athena-docs-widget", AthenaWidget);

// Auto-mount if script tag has data-api-key
if (getConfig().apiKey) {
  const el = document.createElement("athena-docs-widget");
  document.body.appendChild(el);
}
