-- Chicago Board Options Exchange (CBOE) volatility Indexes
SELECT "date", -- Date
       vix,    -- S&P 500 VIX Close
       vxo,    -- S&P 100 VXO Close
       vxn,    -- NASDAQ 100 VIX Close
       vxd     -- Dow Jones Industrial Average VIX Close
FROM cboe.cboe
WHERE "date" >= % (start)s::date
  AND "date" <  %(end)s::date;