// Server-side Basic Auth gate for the Caridence preview.
// Any username is accepted; the password must match.
const PASSWORD = "Piglet1";
const REALM = "Caridence preview";

export const onRequest = async (context) => {
  const { request, next } = context;
  const header = request.headers.get("Authorization") || "";

  if (header.startsWith("Basic ")) {
    try {
      const decoded = atob(header.slice(6)); // "user:pass"
      const sep = decoded.indexOf(":");
      const pass = sep === -1 ? "" : decoded.slice(sep + 1);
      if (pass === PASSWORD) {
        return next();
      }
    } catch (_) {
      // fall through to 401
    }
  }

  return new Response("Authentication required.", {
    status: 401,
    headers: {
      "WWW-Authenticate": `Basic realm="${REALM}", charset="UTF-8"`,
      "Cache-Control": "no-store",
    },
  });
};
