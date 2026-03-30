import type { Metadata } from "next";

import AnalysisClient from "./pageClient";

export const metadata: Metadata = {
  title: "Analysis",
  description: "Browse saved analyses and open them in read-only workspace mode.",
};

export default function AnalysisPage() {
  return <AnalysisClient />;
}

