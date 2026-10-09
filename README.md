# Social Development Tracker

Block → district → state tracker for Jeevika's Social Development vertical:
Gender (DAK), VRF, VPRP, Didi ki Nursery, VanMitra Plantation, Disability SHGs,
and Second Chance (placeholder until programme data arrives). The theme, layout,
map, ranking tables and weight panel are taken from `../Synthesised CLF Tracker`.
The KPIs follow the agreed list in "Social Development Tracker KPIs" (Oct 2026).

## Rebuild

```
python build_sd_inputs.py   # slow (~2-5 min): LokOS members/SHGs/VOs + VPRP -> 2_Data/Processed/sd_tracker/*.pkl
python build_sd_data.py     # ~2 min: everything -> data/*.json
```

Run `build_sd_inputs.py` only when the LokOS or VPRP `.dta` files change. Nursery,
plantation, DAK and VRF sources are read directly by `build_sd_data.py`. Paths are
relative to this folder, so the scripts work on any machine with the same Dropbox
layout. DAK sources come from `Dropbox/DAK_MIS_Scrape`.

## View

The page fetches JSON, so it must be served over http, not opened as a file:

```
python -m http.server 8765
```

Then open http://localhost:8765/dasboard.html. Double-clicking `dasboard.html` also works, because it falls back to `data/bundle.js`.

## Layout

Tabs: Overview, one tab per component (Gender (DAK), VRF, VPRP, Nursery, Plantation, Disability SHGs, Second Chance, Budget), then **SD Index** and **Discrepancy Report**. Budget is a placeholder with weight 0 until its district data and weight are agreed. Each component tab opens with that component's score and the score of each metric behind it, followed by the detail and a map and ranking. The SD Index tab combines the components. Click any component there to see its metrics.

## Period selector

The selector at the top offers **Cumulative**, **2026 (so far)** and **2025**. These apply to every tab. Months are picked separately, inside the DAK and Nursery tabs only. These are calendar years, like the DAK tracker. The build computes every metric, score and rank for each of the three periods (`periods` in each JSON file; cumulative is the top level).
- **DAK cases and alternate services** are counted by month. **Nursery** sales, payments and stock come from the monthly reports.
- **VPRP** is yearly: each calendar year shows that year's plan, and 2026 shows the 2025 plan until 2026 plans exist.
- **Plantation** is by season: 2025 shows the 2025-26 season, and 2026 shows 2025-26 until 2026-27 data exists.
- **VRF, disability and Second Chance** have no history, so they are the same in every period.
- **Months:** the DAK and Nursery tabs have a Month dropdown, listing that year's months, or all months under Cumulative. It shows that month's own figures (`monthly`). Scores stay at the year or cumulative level.

## Discrepancy Report tab and "last updated"

The header shows when the newest source was last scraped, with a per-source list (taken from file dates when the build runs).

The **Discrepancy Report** tab lists entries that look wrong, at block and district level, for follow-up with field teams (`data/checks.json`):
- **Error**: a value that cannot be right. Examples: a case resolved before it was filed, a nursery sale with no plants, the same sale value re-entered, over Rs 5,000 per plant, a VRF corpus below the grant received.
- **Check**: a value far from other places, high or low. Measured as a robust z-score above 3.5 (distance from the median in median-absolute-deviation units). It is used only for metrics where an extreme value suggests a recording problem: case reach, VRF savings discipline, corpus multiplier and interest yield, plants sold per nursery, plantation survival and reach. Also flagged here: more plants alive than given (a check when last season's surviving plants could explain the excess, an error otherwise), DAKs with no cases, resolved cases with no details, VRF savings over 3 times the expected amount, sales larger than the previous month's stock, species not adding up to the total, more VOs filing than are active, and disability SHGs where no member is recorded as disabled.

## Scoring

- **Default weights** (agreed Oct 2026): DAK 30, VRF 10, VPRP 20, Nursery 5, Plantation 5, Disability 10, Second Chance 20. Until Second Chance has data, its share is spread across the others in proportion to their weights. Viewers can change the weights on the page for their own session.

- **"vs target" KPIs** score as % of target, capped at 100: nursery target, VPRP coverage, social issues filed, disability SHG target. Alternate services is scored as a percentile on the number of different services offered in the selected period.
- **"percentile" KPIs** score as a state percentile: blocks are compared with blocks, districts with districts. Ties share the lowest rank in the tie, so hundreds of blocks with zero nursery sales don't all get a high score.
- **DAK case scores** follow the DAK tracker's method: the average of the percentiles of resolution rate, median days to resolve and average age of pending cases.
- **Component score** is the mean of the component's KPI scores. **Overall score** is the mean of the components a unit has. Missing components (no DAK, no VRF data) are left out, not counted as 0. Viewers can re-weight the components on the page.
- **District and state values** are computed from summed numerators and denominators, never averaged from block values.

## Known data issues (as of Oct 2026)

- **Nursery money:** implausible entries are left out of the money figures (plants sold still count): a sale value with no plants sold and under 1% paid (including a Rs 3 crore entry for Akorhi Gola, May 2025); the same sale value re-entered by a block in a later month (a running total carried forward, as in Gaya's Rs 4,80,000 entries); and more than Rs 5,000 per plant. The portal records payment only in the month of sale, so later payments never show up and the payment rate understates real collection.

- **VPRP 2025:** the entitlement export has about a quarter as many requests per VO as 2023, and social category is missing for almost all 2025 requesters. SC/ST reach therefore uses 2024.
- **Alternate services (DAK Specta):** only 10 of the last 12 months were scraped, and only 37 blocks have any activity. Most DAK blocks score 0 on this KPI.
- **Nursery:** the nursery-wise report lists 636 of the 893 nurseries in the monthly report. Its "dried plants" columns are empty, and the monthly "sold till date" column just repeats "sold this month".
- **Block-name aliases:** a few names differ between sources (Rahika = Madhubani, Danapur = Dinapur-Cum-Khagaul, Mokama = Mokameh, Chhauradano = Narkatia). They are listed in `BLOCK_ALIAS` in `build_sd_data.py`. The build log prints any names that fail to match.
