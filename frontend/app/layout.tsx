import type { Metadata } from "next";
import { Cormorant_Garamond, IBM_Plex_Mono, Manrope } from "next/font/google";

import ConsoleGuard from "@/components/ConsoleGuard";
import "./globals.css";

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-sans",
});

const cormorant = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
});

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: {
    default: "Oasis Structures",
    template: "%s | Oasis Structures",
  },
  description: "Formal structural review workspace with 3D validation, material guidance, and heuristic optimisation.",
  applicationName: "Oasis Structures",
  keywords: ["structural review", "3D validation", "material recommendation", "layout optimisation"],
  openGraph: {
    title: "Oasis Structures",
    description: "A formal front door and technical workspace for structural plan review.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className={`h-full antialiased ${manrope.variable} ${cormorant.variable} ${ibmPlexMono.variable}`}>
      <head>
        <script
          id="oasis-theme-init"
          dangerouslySetInnerHTML={{
            __html: `
              (function () {
                try {
                  var key = "oasis-theme";
                  var saved = window.localStorage.getItem(key);
                  var isDark = saved ? saved === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
                  document.documentElement.classList.toggle("theme-dark", isDark);
                } catch (error) {}
              })();
            `,
          }}
        />
      </head>
      <body className="min-h-full flex flex-col">
        <ConsoleGuard />
        {children}
      </body>
    </html>
  );
}
