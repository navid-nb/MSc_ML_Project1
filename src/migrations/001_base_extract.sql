-- CRSP daily (bounded by date)
SELECT
  permno,
  "date",
  ret,
  vol,
  prc,
  shrout,
  dsf.cfacpr -- cumulative factor to adjust prices for splits and dividends
FROM crsp.dsf
WHERE "date" >= %(start)s::date
  AND "date" <  %(end)s::date;