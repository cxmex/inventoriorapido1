/**
 * Cloudflare Worker — Automatic Failover Proxy
 * =============================================
 * Routes traffic to PRIMARY (Railway). If Railway is down, automatically
 * falls back to BACKUP (Render). Both hit the same Supabase database.
 *
 * Setup:
 * 1. Go to dash.cloudflare.com → Workers & Pages → Create Worker
 * 2. Paste this code
 * 3. Set environment variables:
 *    - PRIMARY_URL = "https://web-production-c0d6.up.railway.app"
 *    - BACKUP_URL  = "https://your-app.onrender.com"
 * 4. Add a custom domain or route (e.g. app.fundastock.com)
 *
 * How it works:
 * - Every request goes to PRIMARY first with a 5s timeout
 * - If PRIMARY fails or times out → request goes to BACKUP
 * - /health endpoint is used for fast checks
 * - Adds X-Served-By header so you know which server responded
 */

export default {
  async fetch(request, env) {
    const PRIMARY = env.PRIMARY_URL || "https://web-production-c0d6.up.railway.app";
    const BACKUP  = env.BACKUP_URL  || "https://your-app.onrender.com";

    const url = new URL(request.url);
    const path = url.pathname + url.search;

    // Try PRIMARY first
    try {
      const primaryUrl = PRIMARY + path;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000); // 5s timeout

      const resp = await fetch(primaryUrl, {
        method: request.method,
        headers: request.headers,
        body: request.method !== "GET" && request.method !== "HEAD"
          ? await request.clone().arrayBuffer()
          : undefined,
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (resp.ok || resp.status < 500) {
        // PRIMARY responded — forward its response
        const newResp = new Response(resp.body, resp);
        newResp.headers.set("X-Served-By", "primary-railway");
        return newResp;
      }
      // 5xx error — fall through to backup
    } catch (e) {
      // Timeout or network error — fall through to backup
    }

    // PRIMARY failed — try BACKUP
    try {
      const backupUrl = BACKUP + path;
      const resp = await fetch(backupUrl, {
        method: request.method,
        headers: request.headers,
        body: request.method !== "GET" && request.method !== "HEAD"
          ? await request.clone().arrayBuffer()
          : undefined,
      });

      const newResp = new Response(resp.body, resp);
      newResp.headers.set("X-Served-By", "backup-render");
      return newResp;
    } catch (e) {
      return new Response(
        JSON.stringify({ error: "Both primary and backup servers are down", detail: e.message }),
        { status: 503, headers: { "Content-Type": "application/json" } }
      );
    }
  },
};
