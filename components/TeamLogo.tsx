import styles from "./TeamLogo.module.css";

const espnTeamIds: Record<string, string> = {LA: "lar", WAS: "wsh"};

export function TeamLogo({team, size = 28}: {team: string; size?: number}) {
  const teamId = espnTeamIds[team] ?? team.toLowerCase();
  return <img alt={`${team} logo`} className={styles.logo} height={size} src={`https://a.espncdn.com/i/teamlogos/nfl/500/${teamId}.png`} width={size}/>;
}
