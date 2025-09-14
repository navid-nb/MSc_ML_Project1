-- IBES Summary (statsumu_epsus): identifiers first, then model features for return forecasting
# noinspection SqlType
SELECT ticker      AS ibes_ticker,     -- IBES ticker (stable within IBES)
       cusip       AS cusip8,          -- 8-char CUSIP/SEDOL (mapping only)
       oftic       AS official_ticker, -- official exchange ticker
       cname       AS company_name,    -- firm name (human-readable)
       statpers    AS stat_date,       -- statistical period (as-of date for the summary)
       curcode     AS currency,        -- estimate currency (e.g., USD)

       -- Target/measure
       measure,                        -- which metric: EPS, EBIT, SALES, etc.
       fiscalp     AS periodicity,     -- A/Q/S (annual, quarterly, semi)
       fpi         AS fpi,             -- forecast period indicator (e.g., 1, 2, 3 = horizon)
       estflag     AS est_flag,        -- P/S = primary/secondary (preferred: P)

       -- Core consensus features
       numest      AS n_analysts,      -- number of contributing analysts
       meanest     AS cons_mean,       -- consensus mean estimate
       medest      AS cons_median,     -- consensus median estimate
       stdev       AS cons_stdev,      -- cross-analyst dispersion (std dev)
       high        AS cons_high,       -- highest estimate
       low         AS cons_low,        -- lowest estimate

       -- Realization / timing
       actual      AS actual_val,      -- reported actual for the measure/period
       fpedats     AS fpe_date,        -- fiscal period end date of the forecasted period
       anndats_act AS ann_date,        -- announcement date of the actual (earnings date)

       -- Derived features (consensus shape, dispersion, and surprise)
       CASE
           WHEN meanest IS NOT NULL AND meanest <> 0 AND stdev IS NOT NULL
               THEN stdev / NULLIF(ABS(meanest), 0)
           END     AS cons_cv,         -- coefficient of variation (dispersion / level)
       CASE
           WHEN high IS NOT NULL AND low IS NOT NULL AND meanest IS NOT NULL AND meanest <> 0
               THEN (high - low) / NULLIF(ABS(meanest), 0)
           END     AS cons_range_pct,  -- normalized consensus range
       CASE
           WHEN actual IS NOT NULL AND meanest IS NOT NULL
               THEN actual - meanest
           END     AS surprise_abs,    -- earnings surprise (absolute)
       CASE
           WHEN actual IS NOT NULL AND meanest IS NOT NULL AND meanest <> 0
               THEN (actual - meanest) / NULLIF(ABS(meanest), 0)
           END     AS surprise_pct     -- earnings surprise (% of consensus)

FROM ibes.statsumu_epsus
WHERE statpers < % (end)s::date
  AND statpers >= %(start)s::date
    AND estflag = 'P'             -- keep primary estimates (official forecast used in consensus)
    AND measure = 'EPS'           -- we are interested in Earnings per Share (EPS) forecasts. possible values (based on docs, I haven't tested yet (to-do)): EPS, Sales, EBIT, EBITDA, Cash Flow, etc...)
ORDER BY ibes_ticker, stat_date, measure, periodicity, fpi;