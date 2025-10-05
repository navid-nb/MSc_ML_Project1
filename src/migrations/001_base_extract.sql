-- Center for Research in Security Prices (CRSP) daily stock file (dsf)
SELECT cusip,                                                -- 8-digit CUSIP identifier
       permno,                                               -- CRSP permanent security number (unique ID)
       "date",                                               -- trading date (YYYY-MM-DD)
       bidlo,                                                -- daily low (bid/low price)
       askhi,                                                -- daily high (ask/high price)
       prc,                                                  -- raw closing price (can be bid/ask avg, negative if bid)
       vol,                                                  -- trading volume (raw, unadjusted)
       ret,                                                  -- daily return (with dividends, adjusted for splits)
       bid,                                                  -- bid price
       ask,                                                  -- ask price
       shrout,                                               -- shares outstanding (in thousands, raw)
       cfacpr,                                               -- cumulative factor to adjust prices
       cfacshr,                                              -- cumulative factor to adjust shares/volume
       openprc,                                              -- opening price
       numtrd,                                               -- number of trades (Nasdaq only)
       retx,                                                 -- daily return (excluding dividends)
       -- manual derivations
       prc * cfacpr                           AS adj_prc,    -- adjusted close price (split/dividend adjusted)
       shrout * cfacshr                       AS adj_shrout, -- adjusted shares outstanding (useful for adj. market cap)
       abs(prc * cfacpr) * (shrout * cfacshr) AS adj_mktcap  -- adjusted market cap
FROM crsp.dsf
WHERE "date" >= %(start)s::date
  AND "date" <  %(end)s::date
  AND hexcd = 3; -- nasdaq