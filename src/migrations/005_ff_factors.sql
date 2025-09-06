-- Fama-French daily factors
SELECT
  "date",
  mktrf,
  smb,
  hml,
  rf,
  umd
FROM ff.factors_daily
WHERE "date" >= %(start)s::date
  AND "date" <  %(end)s::date;
