import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build the admin SPA into ../ui/dist so the Rust binary can embed it via
// rust-embed. Absolute base ("/") means assets are requested from /assets/*,
// which the auth middleware allow-lists as public.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../ui/dist",
    emptyOutDir: true,
  },
});
