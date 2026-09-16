# Historical NFL Odds Archive Options

## Decision

Use **The Odds API historical odds endpoint** for a time-correct backtest from the 2020 season onward. It is the only candidate verified here that returns a dated NFL snapshot, several named bookmakers, and a per-book `last_update`. No free source verified here meets all three requirements.

Request the `h2h` market once for every NFL game day at `first_kickoff_on_day - 90 minutes`. Keep the returned snapshot only when its `timestamp` is at or before the target. Store the requested time, returned `timestamp`, every bookmaker's `last_update`, the selected bookmakers, and the raw response. Average de-vigged implied probabilities across a fixed, documented book set. Do not fill a missing bookmaker from another time.

The provider returns the closest snapshot at or before the requested time. The maximum expected age is 10 minutes before September 2022 and five minutes after that, because its documented snapshot intervals changed then. This gives a conservative pre-kickoff observation, not an asserted exact-to-the-second 90-minute quote. [Historical data documentation](https://the-odds-api.com/historical-odds-data/)

## Fit Criteria

The required row has all of these properties:

| Requirement | Meaning |
| --- | --- |
| Day snapshot | One snapshot for every NFL calendar day that has a game. The target is 90 minutes before that day's first kickoff. |
| Historical timestamp | The data source identifies the observed snapshot time and does not return a later quote. |
| Book identity and time | The response identifies each sportsbook and gives a timestamp for its quote. |
| Market average | More than one sportsbook can be selected at the same snapshot, so the project can calculate its own average. |
| Permitted acquisition | A documented API or explicit written permission permits the planned download, storage, and model use. |

## Exact Fit

### The Odds API historical odds

**Classification: exact fit, subject to paid-access confirmation.**

- The historical endpoint accepts an NFL `date` and returns the nearest snapshot at or before it. The response contains `timestamp`, neighbouring timestamps, game kickoff times, named `bookmakers`, market outcomes, and each bookmaker's `last_update`. Its NFL example includes moneyline, spread, and total markets. [Historical endpoint and NFL example](https://the-odds-api.com/historical-odds-data/)
- NFL historical coverage begins at `2020-06-06T10:05:00Z`. Featured markets have 10-minute snapshots from June 2020 and five-minute snapshots from September 2022. Validate book coverage for each target season, especially 2020, rather than assuming a current US book existed then. [Coverage and interval table](https://the-odds-api.com/historical-odds-data/)
- A single `americanfootball_nfl` request for `regions=us&markets=h2h` retrieves the available US bookmakers for that point in time. Historical usage costs 10 credits per region per market. This is 10 credits per game day for the minimum one-region, moneyline-only backtest. [Historical cost rule](https://the-odds-api.com/historical-odds-data/)
- The marketing plan page advertises 500 free credits per month and labels historical odds as included. The historical-data page says historical data is available only on paid subscriptions. These first-party pages conflict. Budget it as paid until The Odds API confirms the free plan's historical entitlement in writing. The stated paid entry plan is 20,000 credits for USD 30 per month. [Plans](https://the-odds-api.com/#get-access) and [historical access statement](https://the-odds-api.com/historical-odds-data/)
- The provider terms permit indefinite storage and use in statistical and machine-learning models. They prohibit reselling or redistributing raw data as a standalone data product. [Terms and conditions](https://the-odds-api.com/terms-and-conditions.html)
- The source is suitable for the requested average because the project controls the included book list. Save that list by season and do not describe the result as a market-wide average.

## Partial Fit

These sources can support a closing/opening-line baseline or a cross-check. They cannot recreate the required snapshot.

### nflverse schedules

**Classification: partial fit. Free and already used by this project.**

- `load_schedules()` returns past and future game information, including `away_moneyline`, `home_moneyline`, `spread_line`, and related line fields. [nflreadr schedule documentation](https://nflreadr.nflverse.com/reference/load_schedules.html)
- The published schema supplies no observation time, sportsbook identity, or per-book quote timestamp. Thus, its lines cannot prove availability 90 minutes before kickoff or form a multiple-sportsbook average.
- Use it only as the present historical market baseline and retain the existing timing warning. Do not relabel it as a 90-minute snapshot.

### Covers public odds and Sports Odds History pages

**Classification: partial fit for manual research; not an approved programmatic archive.**

- Covers publishes current NFL odds pages and a public page called "Sports Odds History." [NFL odds](https://www.covers.com/sport/football/nfl/odds) and [Sports Odds History](https://www.covers.com/sportsoddshistory/)
- The reviewed public pages do not document a public historical-odds API, an export, a per-book historical timestamp schema, or a rate limit. They therefore do not establish a reproducible 90-minute snapshot.
- Covers' `robots.txt` lists crawl restrictions, but it is not permission to copy or store data. [Covers robots.txt](https://www.covers.com/robots.txt)
- Do not scrape or call undocumented endpoints. Use only manual citations or written permission that expressly covers bulk historical retrieval, storage, and model use. This option cannot supply the required backtest unless Covers provides that permission and a timestamped data export.

### Pro-Football-Reference (PFR)

**Classification: partial as data shape; not usable for this project.**

- PFR game pages can provide a game-level Vegas line, but they do not provide a documented per-book historical quote time. Therefore the line is at most a game-level reference, not the requested multi-book, 90-minute observation.
- Sports Reference prohibits creating a competing data store or service from site content. It also prohibits using site content to support machine-learning methods that predict or score inputs. Its data-use page says custom downloads require a minimum USD 5,000 request. [Sports Reference data-use terms](https://www.sports-reference.com/data_use.html)
- The project's prediction model and stored historical archive fall within those restrictions. Do not scrape, download in bulk, or use PFR odds as model inputs without a written licence that covers this use.

### Football-Data.co.uk-style public CSV archives

**Classification: partial in general; no verified NFL archive in this review.**

- The publisher's original field notes show the useful shape of many free odds CSVs: named-book opening and closing fields plus market average fields. However, they state that weekend odds are collected on Friday afternoons, which is not the required game-day 90-minute time. [Publisher field notes](https://www.football-data.co.uk/notes.txt)
- The reviewed documentation concerns football CSVs and did not verify an NFL download. Do not treat a search result, a mirror, or a Kaggle repost as an NFL source of record. A verified original NFL release would still be partial unless it includes an observation timestamp.

## Not Usable

### SportsDataIO free trial and historical product

**Classification: not usable without a paid agreement and schema confirmation.**

- SportsDataIO says that odds older than 30 days are in its historical warehouse. It also states that its free trial provides only UEFA Champions League access, not NFL. [NFL API documentation](https://sportsdata.io/developers/api-documentation/nfl)
- Its public pages reviewed here do not verify the historical NFL response cadence or a per-book historical timestamp. Contact sales only if The Odds API cannot meet coverage needs. Require a sample containing the exact snapshot timestamp, book identifier, quote timestamp, and NFL 2020-plus coverage before purchase.

### Betfair historical data

**Classification: not usable for the required market average.**

- Betfair advertises an official Historical Data Services API, but its official historical-data pages required authenticated access during this review. [Historical Data Services API](https://developer.betfair.com/historical-data-services-api/)
- Even if obtained, Betfair is a single exchange. Its traded prices are valuable as a separate exchange benchmark, but they cannot form an average across multiple sportsbooks. Do not substitute exchange prices for sportsbook quotes.

### Other exchange APIs and public odds sites

**Classification: not usable unless their owner supplies a documented, licensed, multi-book archive.**

- A prediction-market or exchange API reports one venue's contract price. It may have a timestamped history, but it cannot produce the required multiple-sportsbook average.
- A public odds-comparison page is not an API contract. Without official historical endpoint documentation, retention terms, and per-book timestamps, it fails the reproducibility and permission requirements.
- Do not use browser automation, page scraping, undocumented JSON endpoints, cached pages, or third-party mirrors as a backtest source. A public page and an indexable `robots.txt` file do not grant data rights.

## Implementation Guardrails

1. Use the schedule's timezone-aware kickoff timestamps to group games by game day and calculate each target time.
2. Query `americanfootball_nfl`, `regions=us`, and `markets=h2h` at each target time. Record the provider's returned snapshot time.
3. Reject any bookmaker whose `last_update` is later than the returned snapshot or whose two moneyline outcomes are incomplete.
4. Use a fixed whitelist of at least two books with sufficient historical coverage. Report the count and names used for every game.
5. Convert each book's two moneylines to implied probabilities, de-vig each pair, then average probabilities. Never average American odds directly.
6. Preserve raw responses and a manifest with provider, query, retrieval time, target time, returned time, and parser version.
7. Keep the existing nflverse line separately as a partial baseline. Do not mix it into the new market average.

## Source Method

Research completed 2026-09-15. Sources above are provider documentation, provider terms, original repository documentation, or publisher-owned public pages. Where a provider page was inaccessible or did not document a required property, this note records that limit instead of inferring it from an unofficial client, a scraper, or a reposted dataset.
