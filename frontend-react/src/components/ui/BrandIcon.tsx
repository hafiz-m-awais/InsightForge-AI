/**
 * InsightForge Brand Icon — "The Data Prism"
 *
 * Concept: Three raw data streams enter a forge-prism from the left.
 * The prism transforms and converges them into a single focused
 * insight beam that exits to the right — raw data → intelligence.
 *
 * Anatomy:
 *  ● ─── ┐                 ← top data source
 *  ● ─── ┤ ◆ (prism) ─── → ● spark   ← middle (heaviest)
 *  ● ─── ┘                 ← bottom data source
 */

export function BrandIcon({
  size = 32,
  className,
}: {
  size?: number
  className?: string
}) {
  // Use static IDs — this icon is rendered once per page.
  const gid  = 'if-prism-grad'
  const bgid = 'if-beam-grad'
  const fid  = 'if-glow'

  return (
    <svg
      width={size}
      height={Math.round(size * 0.8)}   // 40 × 32 aspect ratio
      viewBox="0 0 40 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="InsightForge"
    >
      <defs>
        {/* Prism body: blue → indigo diagonal */}
        <linearGradient id={gid} x1="12" y1="3" x2="30" y2="29" gradientUnits="userSpaceOnUse">
          <stop offset="0%"   stopColor="#60a5fa" />
          <stop offset="55%"  stopColor="#818cf8" />
          <stop offset="100%" stopColor="#a78bfa" />
        </linearGradient>

        {/* Output beam: indigo → sky */}
        <linearGradient id={bgid} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stopColor="#a5b4fc" />
          <stop offset="100%" stopColor="#e0f2fe" />
        </linearGradient>

        {/* Glow for the output spark */}
        <filter id={fid} x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="1.4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* ── Input data nodes (left) ─────────────────────────── */}
      <circle cx="3"   cy="10" r="2"   fill="#93c5fd" opacity="0.65" />
      <circle cx="3"   cy="16" r="2.5" fill="#60a5fa" />
      <circle cx="3"   cy="22" r="2"   fill="#93c5fd" opacity="0.65" />

      {/* ── Connector lines: nodes → prism left face ────────── */}
      <line x1="5"  y1="10" x2="12" y2="10"
            stroke="#93c5fd" strokeWidth="1.5" strokeLinecap="round" opacity="0.55" />
      <line x1="5"  y1="16" x2="12" y2="16"
            stroke="#60a5fa" strokeWidth="2"   strokeLinecap="round" />
      <line x1="5"  y1="22" x2="12" y2="22"
            stroke="#93c5fd" strokeWidth="1.5" strokeLinecap="round" opacity="0.55" />

      {/* ── Prism body ──────────────────────────────────────── */}
      {/* Triangle: flat left face (x=12, y 3→29), sharp right tip (30, 16) */}
      <path d="M12 3 L12 29 L30 16 Z" fill={`url(#${gid})`} />

      {/* Left-face highlight (bright edge = "forge mouth") */}
      <line x1="12" y1="3" x2="12" y2="29"
            stroke="white" strokeWidth="1.5" strokeLinecap="round" opacity="0.32" />

      {/* Interior convergence lines (data flowing to insight point) */}
      <line x1="12" y1="10" x2="30" y2="16"
            stroke="white" strokeWidth="1" opacity="0.22" />
      <line x1="12" y1="22" x2="30" y2="16"
            stroke="white" strokeWidth="1" opacity="0.22" />

      {/* ── Output beam ─────────────────────────────────────── */}
      {/*<line x1="30" y1="16" x2="37.5" y2="16"
            stroke={`url(#${bgid})`} strokeWidth="2.5" strokeLinecap="round" />

      {/* ── Output spark (insight) ──────────────────────────── */}
      {/*<circle cx="38" cy="16" r="2" fill="#e0f2fe" filter={`url(#${fid})`} />*/}
    </svg>
  )
}
