import type {GamePick} from "@/lib/types";

export function MarketFreshness({game}:{game:GamePick}) {
  if (!game.marketCapturedAt || !game.marketUpdatedAt) {
    return <div className="market-freshness">{game.marketSource ?? "Recorded odds"} · age unknown</div>;
  }
  const updated = new Intl.DateTimeFormat("en-US", {
    month:"short", day:"numeric", hour:"numeric", minute:"2-digit", timeZone:"America/New_York", timeZoneName:"short",
  }).format(new Date(game.marketUpdatedAt));
  return <div className="market-freshness">
    <div>{game.marketSource} · {game.marketBookCount} books</div>
    <div>Oldest price: <time dateTime={game.marketUpdatedAt}>{updated}</time>
      {game.marketAgeMinutes != null && ` · ${Math.ceil(game.marketAgeMinutes)}m old at refresh`}
    </div>
  </div>;
}
