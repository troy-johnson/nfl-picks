import type {GamePick} from "@/lib/types";

const outcomes = {win:"Won", loss:"Lost", tie:"Tie", missing:"No submitted pick", unavailable:"Submitted pick unavailable", pending:"Pending"};

export function GameResult({game}:{game:GamePick}) {
  const result = game.result;
  if (!result || result.status === "scheduled") return null;
  return <div className={`game-result game-result-${result.pickResult}`}>
    <div className="result-heading"><strong>{result.status === "final" ? "Final" : result.status === "in_progress" ? "In progress" : "Status update"}</strong><span>{result.detail}</span></div>
    {result.awayScore != null && result.homeScore != null && <div className="result-score">{game.awayTeam} {result.awayScore} <span>–</span> {game.homeTeam} {result.homeScore}</div>}
    <div className="your-pick">Your pick: {result.submittedPick ? `${result.submittedPick} · ${outcomes[result.pickResult]}` : result.pickResult === "missing" ? "No submitted pick" : "Unavailable"}</div>
  </div>;
}
