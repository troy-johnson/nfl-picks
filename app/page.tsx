import picksJson from "@/public/data/current.json";
import {GameCard} from "@/components/GameCard";
import {PoolSummary} from "@/components/PoolSummary";
import type {PicksPayload} from "@/lib/types";
const picks=picksJson as PicksPayload;
function fmt(v:string){if(!v)return "Not generated yet";return new Intl.DateTimeFormat("en-US",{dateStyle:"medium",timeStyle:"short",timeZone:"America/Denver"}).format(new Date(v));}
export default function Home() {
  const games = [...picks.games].sort((a,b) => `${a.gameday}T${a.gametime ?? "23:59"}`.localeCompare(`${b.gameday}T${b.gametime ?? "23:59"}`));
  return <main className="page-shell">
    <header className="page-header"><div><div className="eyebrow">{picks.season || 2026} pick&apos;em</div><h1>NFL Picks</h1><p>Market-informed, stats-driven straight-up win probabilities.</p></div><div className="week-badge">Week {picks.week || "—"}</div></header>
    <section className="summary-bar">
      <div><span>Updated</span><strong>{fmt(picks.generatedAt)}</strong></div>
      <div><span>Market weight</span><strong>{Math.round((picks.model?.marketWeight ?? 0)*100)}%</strong></div>
      <div><span>Stat model validation</span><strong>{picks.model?.validationAccuracy == null ? "—" : `${Math.round(picks.model.validationAccuracy*100)}%`}</strong></div>
    </section>
    <PoolSummary pool={picks.pool}/>
    {games.length ? <section className="game-grid">{games.map(game => <GameCard key={game.gameId} game={game}/>)}</section> : <section className="empty-state"><h2>No generated picks yet</h2><p>Run the prediction workflow to populate this week.</p></section>}
    <section className="legend"><h2>How to read a card</h2><dl>
      <dt>Pick</dt><dd>The team with the higher blended win probability. This is the accuracy recommendation.</dd>
      <dt>Stat model</dt><dd>Logistic regression on rolling eight-game EPA, success rate, rest days, and neutral-site status. It uses no betting lines.</dd>
      <dt>Market</dt><dd>The average moneyline probability after removing each bookmaker&apos;s margin. Fresh consensus requires at least two books updated within 30 minutes.</dd>
      <dt>Odds age</dt><dd>The card shows the oldest included price and its age when the forecast was generated. Fallback odds have unknown age.</dd>
      <dt>Schedule spread</dt><dd>The recorded nflverse spread. Its update time can differ from the live moneyline.</dd>
      <dt>Crowd</dt><dd>The national ESPN pick share. It does not represent this group&apos;s actual picks or change the forecast.</dd>
      <dt>Confidence</dt><dd>High means 72% or more. Medium means 62% to 71%. Low means under 62%.</dd>
      <dt>Pool strategy</dt><dd>Current recommendations maximize expected correct picks. The group panel shows the actual standings and dropped-week rule. Older locked games can show an archived pool pick.</dd>
      <dt>Submitted picks</dt><dd>The check compares visible ESPN picks with the recommendation. ESPN can hide open picks, so an unavailable pick is not a confirmed missing pick.</dd>
    </dl></section>
    <footer>Statistical probabilities blended with market odds. Forecasts and their market inputs are archived together before kickoff.</footer>
  </main>;
}
