"use client";

import { useEffect, useState, type CSSProperties } from "react";

import { useTheme } from "@/components/theme/useTheme";

type ThemeToggleProps = {
  className?: string;
  style?: CSSProperties;
};

export default function ThemeToggle({ className, style }: ThemeToggleProps) {
  const { isDark, toggleTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className={className ?? "btn btn-default"}
      title={mounted && isDark ? "Switch to light theme" : "Switch to dark theme"}
      aria-label={mounted && isDark ? "Switch to light theme" : "Switch to dark theme"}
      style={{
        minWidth: 40,
        width: 40,
        height: 34,
        padding: 0,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
        ...style,
      }}
    >
      {mounted && isDark ? (
        <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="5" />
          <path
            d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"
            strokeLinecap="round"
          />
        </svg>
      ) : (
        <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
    </button>
  );
}
