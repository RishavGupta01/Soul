import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // GitHub Pages serves project sites from /<repo>/, local dev from /.
  // The deploy workflow sets BASE_PATH accordingly.
  base: process.env.BASE_PATH || "/",
  plugins: [react()],
  server: { port: 5173, open: false },
  build: { target: "es2022", chunkSizeWarningLimit: 1200 },
  test: {
    environment: "node",
    include: ["src/**/*.test.{ts,tsx}", "scripts/**/*.test.mjs"],
  },
});