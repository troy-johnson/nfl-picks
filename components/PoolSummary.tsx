import type {PoolSummary as PoolData} from "@/lib/types";

export function PoolSummary({pool}:{pool:PoolData|undefined}) {
  if (!pool) return null;
  if (pool.status === "unavailable") {
    return <section className="pool-summary"><h2>ESPN group</h2><p>Group data is unavailable. Submitted picks could not be checked.</p></section>;
  }
  const counts = pool.pickCheck.counts;
  const checked = new Intl.DateTimeFormat("en-US", {month:"short", day:"numeric", hour:"numeric", minute:"2-digit", timeZone:"America/New_York", timeZoneName:"short"}).format(new Date(pool.pickCheck.checkedAt));
  return <section className="pool-summary">
    <div className="pool-heading"><h2><a href={pool.url}>{pool.name}</a></h2><strong>Accuracy priority</strong></div>
    <p>{pool.entryName}: #{pool.rank} of {pool.entries} · {pool.points} points · {pool.correctPicksBehind} correct picks behind first
      {pool.remainingGames != null && ` · ${pool.remainingGames} games remain`}</p>
    <p>{pool.pointsPerPick} points per correct pick. {pool.dropLowestWeek ? "The lowest-scoring week is dropped." : "Every week counts."}
      {pool.tiebreakers && " Tiebreakers are enabled."}</p>
    {pool.droppedWeeks.length > 0 && <p>Excluded week: {pool.droppedWeeks.join(", ")}.</p>}
    <p>{pool.strategyReason}</p>
    <div className="pick-check-summary">
      <strong>Submitted picks</strong>
      <span>{counts.matched} match · {counts.different} differ · {counts.missing} missing · {counts.unavailable} unavailable</span>
      <span>{pool.pickCheck.visibilityNote} Source: {pool.pickCheck.source}.</span>
      <span>Checked data as of <time dateTime={pool.pickCheck.checkedAt}>{checked}</time>.</span>
    </div>
  </section>;
}
