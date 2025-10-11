-- Institutional Brokers' Estimate System (IBES) statistical summary (unadjusted) of U.S. EPS forecasts (statsumu_epsus)
SELECT ticker   AS ibes_ticker     -- IBES ticker (stable within IBES)
     , cusip    AS cusip8          -- 8-char CUSIP/SEDOL (mapping only)
     , oftic    AS official_ticker -- official exchange ticker
     , cname    AS company_name    -- firm name (human-readable)
     , statpers AS stat_date       -- statistical period (as-of date for the summary)
     , curcode  AS currency        -- estimate currency (e.g., USD)

     -- Target/measure
     , measure                     -- which metric: EPS, EBIT, SALES, etc.
     , fiscalp  AS periodicity     -- A/Q/S (annual, quarterly, semi)
     , fpi      AS fpi             -- forecast period indicator (e.g., 1, 2, 3 = horizon)
     , estflag  AS est_flag        -- P/S = primary/secondary (preferred: P)

     -- Core consensus features
     , numest   AS n_analysts      -- number of contributing analysts
     , numup    AS n_up            -- number of upward revisions
     , numdown  AS n_down          -- number of downward revisions
     , meanest  AS cons_mean       -- consensus mean estimate
     , medest   AS cons_median     -- consensus median estimate
     , stdev    AS cons_stdev      -- cross-analyst dispersion (std dev)
     , highest  AS cons_high       -- highest estimate
     , lowest   AS cons_low        -- lowest estimate

     -- Forecast horizon
     , fpedats  AS fpe_date        -- fiscal period end date of the forecasted period

     -- Derived features (consensus shape, dispersion)
     , CASE
           WHEN meanest IS NOT NULL AND meanest <> 0 AND stdev IS NOT NULL
               THEN stdev / NULLIF(ABS(meanest), 0)
    END         AS cons_cv         -- coefficient of variation (dispersion / level)
     , CASE
           WHEN highest IS NOT NULL AND lowest IS NOT NULL AND meanest IS NOT NULL AND meanest <> 0
               THEN (highest - lowest) / NULLIF(ABS(meanest), 0)
    END         AS cons_range_pct  -- normalized consensus range

FROM ibes.statsumu_epsus
WHERE statpers <  %(end)s::date
  AND statpers >= %(start)s::date
  AND estflag = 'P'       -- keep primary estimates
  AND measure = 'EPS'     -- restrict to EPS forecasts
  AND usfirm = 1          -- keep U.S. firms only
ORDER BY ibes_ticker, stat_date, measure, periodicity, fpi;