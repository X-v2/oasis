"use client";

import type { CSSProperties, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

import styles from "./RevealOnScroll.module.css";

type RevealTag = "article" | "div" | "header" | "li" | "section";

type RevealOnScrollProps = {
  as?: RevealTag;
  children: ReactNode;
  className?: string;
  delay?: number;
  threshold?: number;
};

export default function RevealOnScroll({
  as = "div",
  children,
  className,
  delay = 0,
  threshold = 0.18,
}: RevealOnScrollProps) {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node || visible) {
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      {
        threshold,
        rootMargin: "0px 0px -10% 0px",
      },
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, [threshold, visible]);

  const Component = as;
  const combinedClassName = [styles.reveal, visible ? styles.visible : "", className ?? ""].filter(Boolean).join(" ");
  const inlineStyle = { "--reveal-delay": `${delay}ms` } as CSSProperties;
  const assignRef = (node: HTMLElement | null) => {
    ref.current = node;
  };

  return (
    <Component ref={assignRef as never} className={combinedClassName} style={inlineStyle}>
      {children}
    </Component>
  );
}
