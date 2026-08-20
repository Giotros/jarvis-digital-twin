import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Use Docker service names when running in container, localhost otherwise
const apiHost = process.env.JARVIS_API_HOST || "localhost";
const n8nHost = process.env.JARVIS_N8N_HOST || "localhost";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001,
    host: "0.0.0.0",
    proxy: {
      "/orchestration": {
        target: `http://${apiHost}:8000`,
        changeOrigin: true,
      },
      "/health": {
        target: `http://${apiHost}:8000`,
        changeOrigin: true,
      },
      "/webhook": {
        target: `http://${n8nHost}:5678`,
        changeOrigin: true,
      },
    },
  },
});
