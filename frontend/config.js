// Runtime API configuration for Kryon split deployment:
// - Localhost / 127.0.0.1 / local Docker: uses same-origin relative "/api/v1/..." paths.
// - Firebase Hosting (*.web.app / *.firebaseapp.com): targets the Render FastAPI backend.
(function () {
  const host = (window.location && window.location.hostname) || "";
  const isLocal = host === "localhost" || host === "127.0.0.1" || host === "";
  const DEFAULT_PROD_API = "https://kryon-rag-api.onrender.com";

  window.KRYON_API_BASE = isLocal ? "" : DEFAULT_PROD_API.replace(/\/+$/, "");

  window.kryonApiUrl = function (path) {
    const normalized = path.startsWith("/") ? path : "/" + path;
    return window.KRYON_API_BASE ? `${window.KRYON_API_BASE}${normalized}` : normalized;
  };

  if (window.KRYON_API_BASE && typeof window.fetch === "function") {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
      if (typeof input === "string" && input.startsWith("/api/")) {
        return nativeFetch(window.kryonApiUrl(input), init);
      }
      return nativeFetch(input, init);
    };
  }
})();
