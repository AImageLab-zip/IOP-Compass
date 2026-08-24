import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is proxied rather than called cross-origin, so the browser sees one
// origin and the backend's CORS allowlist stays narrow.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.IOPC_API ?? "http://127.0.0.1:5000",
        changeOrigin: true,
      },
    },
  },
});
