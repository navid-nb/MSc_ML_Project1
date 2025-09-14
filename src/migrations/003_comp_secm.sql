-- Compustat Security Monthly Descriptor (SECM) - feature-rich for return forecasting
# noinspection SqlType
SELECT gvkey,                                                     -- firm identifier
       datadate,                                                  -- month-end snapshot date
       tic,                                                       -- ticker
       cusip,                                                     -- CUSIP (for linking/ID)

       -- Prices / returns
       prccm,                                                     -- month-end close price
       prchm,                                                     -- high price in month
       prclm,                                                     -- low price in month
       trt1m,                                                     -- total return (this month)
       trfm,                                                      -- total return factor (for compounding)

       -- Dividends / adjustments
       dvpsxm,                                                    -- dividends per share (ex-date, monthly)
       dvpspm,                                                    -- dividends per share (pay-date, monthly)
       dvrate,                                                    -- dividend rate (monthly)
       cheqvm,                                                    -- cash equivalent distributions
       ajexm,                                                     -- cumulative adjustment factor (ex-date)
       ajpm,                                                      -- cumulative adjustment factor (pay-date)

       -- Shares & volume
       cshoq,                                                     -- common shares outstanding
       cshom,                                                     -- shares outstanding (issue-level)
       cshtrm,                                                    -- trading volume
       adrrm,                                                     -- ADR ratio (if ADR)

       -- Derived features
       (prchm - prclm)                          AS px_range_m,    -- intra-month price range
       NULLIF(cshtrm, 0)                        AS vol_raw_m,     -- raw monthly volume
       CASE
           WHEN cshoq IS NOT NULL AND cshoq <> 0
               THEN cshtrm / cshoq END          AS turnover_m,    -- share turnover ratio
       CASE
           WHEN prccm IS NOT NULL AND cshoq IS NOT NULL
               THEN prccm * cshoq END           AS mktcap_m,      -- market cap proxy
       CASE
           WHEN prclm IS NOT NULL AND prclm > 0
               THEN (prchm - prclm) / prclm END AS range_pct_m,   -- normalized price range
       CASE
           WHEN ajexm IS NOT NULL AND prccm IS NOT NULL
               THEN prccm * ajexm END           AS prccm_adj_ex,  -- adjusted price (ex-date)
       CASE
           WHEN dvpsxm IS NOT NULL AND prccm IS NOT NULL AND prccm <> 0
               THEN dvpsxm / prccm END          AS div_yield_ex_m -- dividend yield (approx.)
FROM comp.secm
WHERE datadate <  %(end)s::date
  AND datadate >= %(start)s::date;