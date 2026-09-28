import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/syne/700.css";
import "@fontsource/syne/800.css";
import "@fontsource/schibsted-grotesk/400.css";
import "@fontsource/schibsted-grotesk/500.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "./styles/base.css";
import "./styles/app.css";
import "./styles/scenes.css";
import { App } from "./App";
import { initRangeFill } from "./motion/rangeFill";

initRangeFill();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
