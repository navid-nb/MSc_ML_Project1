-- Fama-French daily factors
SELECT "date",
       mktrf, -- Excess Return on the Market
       smb,   -- Small-Minus-Big Return
       hml,   -- High-Minus-Low Return
       rf,    -- Risk-Free Return Rate (One Month Treasury Bill Rate)
       umd    -- Momentum
FROM ff.factors_daily
WHERE "date" >= % (start)s::date
  AND "date" <  %(end)s::date;
