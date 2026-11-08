import type {GamePick} from "@/lib/types";

export function ResultsSummary({games, updatedAt}:{games:GamePick[];updatedAt?:string}) {
  if (!updatedAt) return null;
  const finished = games.filter(game => game.result?.status === "final");
  const count = (outcome:string) => finished.filter(game => game.result?.pickResult === outcome).length;
  const checked = new Intl.DateTimeFormat("en-US", {month:"short", day:"numeric", hour:"numeric", minute:"2-digit", timeZone:"America/New_York", timeZoneName:"short"}).format(new Date(updatedAt));
  return <section className="pool-summary results-summary">
    <div className="pool-heading"><h2>Your results this week</h2><strong>{count("win")} won · {count("loss")} lost · {count("tie")} tied</strong></div>
    <p>{finished.length} of {games.length} games final · {count("missing")} missing picks · {count("unavailable")} unavailable picks in final games.</p>
    <p>Results use your submitted ESPN picks. Scores checked <time dateTime={updatedAt}>{checked}</time>.</p>
  </section>;
}
