"use client";

import { useEffect } from "react";

const suppressedWarnings = [
  "THREE.THREE.Clock: This module has been deprecated. Please use THREE.Timer instead.",
];

export default function ConsoleGuard() {
  useEffect(() => {
    const originalWarn = console.warn;

    console.warn = (...args: unknown[]) => {
      const first = typeof args[0] === "string" ? args[0] : "";
      if (suppressedWarnings.some((message) => first.includes(message))) {
        return;
      }

      originalWarn(...args);
    };

    return () => {
      console.warn = originalWarn;
    };
  }, []);

  return null;
}
