import type { Metadata } from "next";
import Link from "next/link";
import ThemeToggle from "@/components/theme/ThemeToggle";
import RevealOnScroll from "@/components/landing/RevealOnScroll";
import styles from "./page.module.css";

const workflowSteps = [
  {
    step: "01",
    title: "Groundwork",
    body: "Ingest structured JSON or image-derived plan data natively, maintaining precision from the foundation up.",
  },
  {
    step: "02",
    title: "Geometry Mapping",
    body: "Resolve architectural coordinates, surface inferred volumes, and validate strict spatial rules prior to modeling.",
  },
  {
    step: "03",
    title: "Volumetric Inspection",
    body: "Interact with raw walls, structural slabs, load-bearing nodes, and spatial openings in a pure 3D construct.",
  },
  {
    step: "04",
    title: "Material Logic",
    body: "Cross-reference material tolerances directly against selected structural elements, rendering risk visible in context.",
  },
  {
    step: "05",
    title: "Structural Optimisation",
    body: "Apply severity-tagged heuristics to adapt the layout. Preview physical impact before committing structural changes.",
  },
];

const capabilities = [
  {
    title: "Contextual Validation",
    body: "Oasis doesn't just parse shapes. It tags readiness, calculates potential structural clashes, and details the physical risk profile of the geometry presented.",
    highlights: ["Formal issue logs", "Uncompromising normalization", "Preserved spatial logic"],
  },
  {
    title: "Precision Material Alignment",
    body: "Materials require specific conditions. Suggestions strictly adapt to the current structural entity under review.",
    highlights: ["Targeted recommendation matrices", "Explicit tradeoff documentation", "Core safety summaries"],
  },
  {
    title: "Defensible Logic",
    body: "Space shouldn't be guessed. Optimisations are surfaced with evidence, severity tiers, and previewable confidence ratings.",
    highlights: ["Real-time 3D layout projection", "Intent-driven confirmations", "Granular physical interpretations"],
  },
];

export const metadata: Metadata = {
  title: "Oasis Structures | Home",
  description: "A solid, unyielding workspace for formal structural review, material planning, and architectural optimization.",
};

export default function HomePage() {
  return (
    <main className={styles.page}>
      <div className={styles.themeDock}>
        <ThemeToggle />
      </div>

      <section className={styles.hero}>
        <div className={styles.heroGrid}>
          <div className={styles.heroContent}>
            <RevealOnScroll as="div" delay={0}>
              <div className={styles.brandTag}>Oasis Structures</div>
              <h1 className={styles.heroTitle}>
                Built on solid ground.
                <span className={styles.heroTitleAccent}>Engineered for reality.</span>
              </h1>
              <p className={styles.heroBody}>
                An unyielding workspace replacing fragile abstractions with concrete spatial validation. Evaluate geometry,
                map physical materials, and optimize structural layouts with absolute confidence. No templated assumptions,
                just verifiable architecture.
              </p>

              <div className={styles.heroActions}>
                <Link href="/workspace" className={styles.buttonSolid}>
                  Enter the Workspace
                </Link>
                <Link href="/analysis" className={styles.buttonOutline}>
                  View Analysis
                </Link>
              </div>

              <div className={styles.statusGrid}>
                <div className={styles.statusBlock}>
                  <div className={styles.statusLabel}>Mode</div>
                  <div className={styles.statusValue}>Spatial Processing</div>
                </div>
                <div className={styles.statusBlock}>
                  <div className={styles.statusLabel}>Integrity</div>
                  <div className={styles.statusValue}>Strict Validation</div>
                </div>
                <div className={styles.statusBlock}>
                  <div className={styles.statusLabel}>Physics</div>
                  <div className={styles.statusValue}>Material Locked</div>
                </div>
              </div>
            </RevealOnScroll>
          </div>

          <div className={styles.heroVisualizer}>
             <RevealOnScroll as="div" delay={150} className={styles.visualizerContainer}>
                {/* Structural abstract representation */}
                <div className={styles.architecturalBlock}>
                  <div className={styles.blockFaceTop} />
                  <div className={styles.blockFaceLeft} />
                  <div className={styles.blockFaceRight} />
                  
                  <div className={styles.measureLineH}>
                     <span className={styles.measureText}>12.4m</span>
                  </div>
                  <div className={styles.measureLineV}>
                     <span className={styles.measureText}>4.8m</span>
                  </div>
                  
                  <div className={styles.structuralGrid} />
                  <div className={styles.loadPoint} />
                  <div className={styles.loadPointAlt} />
                </div>
                
                <div className={styles.overlayData}>
                  <div className={styles.dataRow}>
                    <span>Tolerance</span>
                    <span>±0.05</span>
                  </div>
                  <div className={styles.dataRow}>
                    <span>Load Path</span>
                    <span className={styles.statusGood}>Verified</span>
                  </div>
                  <div className={styles.dataRow}>
                     <span>Material Form</span>
                     <span>Concrete</span>
                  </div>
                </div>
             </RevealOnScroll>
          </div>
        </div>
      </section>

      <section id="analysis" className={styles.methodology}>
        <div className={styles.shell}>
          <RevealOnScroll as="header" className={styles.sectionHeader}>
            <h2 className={styles.sectionTitle}>The Oasis Analysis</h2>
            <p className={styles.sectionBody}>A brutal, honest path from flat plan input to a fully validated spatial model. Built to respect the physical constraints of structural engineering.</p>
          </RevealOnScroll>

          <div className={styles.workflowGrid}>
            {workflowSteps.map((item, index) => (
              <RevealOnScroll key={item.step} as="article" className={styles.workflowBlock} delay={index * 50}>
                <div className={styles.blockHead}>
                   <span className={styles.stepMarker}>{item.step}</span>
                   <h3 className={styles.blockTitle}>{item.title}</h3>
                </div>
                <p className={styles.blockBody}>{item.body}</p>
              </RevealOnScroll>
            ))}
          </div>
        </div>
      </section>

      <section className={styles.capabilities}>
        <div className={styles.shell}>
          <RevealOnScroll as="header" className={styles.sectionHeader}>
             <h2 className={styles.sectionTitle}>Structural Capabilities</h2>
             <p className={styles.sectionBody}>Beyond drawing lines. A platform that reasons about mass, space, and material necessity.</p>
          </RevealOnScroll>

          <div className={styles.capGrid}>
             {capabilities.map((item, index) => (
               <RevealOnScroll key={item.title} as="article" className={styles.capBlock} delay={index * 100}>
                  <h3 className={styles.capTitle}>{item.title}</h3>
                  <p className={styles.capBody}>{item.body}</p>
                  <ul className={styles.capList}>
                    {item.highlights.map(hl => (
                      <li key={hl}>
                        <div className={styles.capBullet} />
                        {hl}
                      </li>
                    ))}
                  </ul>
               </RevealOnScroll>
             ))}
          </div>
        </div>
      </section>

      <section className={styles.terminal}>
        <RevealOnScroll as="div" className={styles.terminalContainer}>
           <div className={styles.terminalInner}>
             <h2 className={styles.terminalTitle}>Initiate the Build</h2>
             <p className={styles.terminalBody}>Upload your spatial data. Let the architecture speak for itself.</p>
             <Link href="/workspace" className={styles.buttonTerminal}>
                Deploy Workspace
             </Link>
           </div>
        </RevealOnScroll>
      </section>
    </main>
  );
}
