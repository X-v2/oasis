"use client";

type ExplanationPanelProps = {
  explanation: string;
  darkMode?: boolean;
};

export default function ExplanationPanel({ explanation, darkMode = false }: ExplanationPanelProps) {
  return (
    <section
      className={`rounded-[24px] border p-5 shadow-sm ${
        darkMode ? "border-violet-900/70 bg-[#1e1635]/90" : "border-stone-200 bg-white/90"
      }`}
    >
      <h3 className={`text-base font-semibold ${darkMode ? "text-violet-50" : "text-stone-900"}`}>Explanation</h3>
      <p className={`mt-1 text-xs font-semibold tracking-[0.16em] uppercase ${darkMode ? "text-violet-300/85" : "text-stone-500"}`}>Selection Brief</p>
      <p className={`mt-3 text-sm leading-7 ${darkMode ? "text-violet-100/88" : "text-stone-700"}`}>{explanation}</p>
    </section>
  );
}
