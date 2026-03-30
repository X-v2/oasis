import type { Metadata } from "next";
import ConsoleGuard from "@/components/ConsoleGuard";
import "./globals.css";

export const metadata: Metadata = {
  title: "Structural Reasoning Viewer",
  description: "3D structural validation, material recommendations, and demo-safe fallbacks.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">
        <ConsoleGuard />
        {children}
      </body>
    </html>
  );
}
